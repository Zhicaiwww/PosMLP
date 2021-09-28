import collections.abc
import logging
import math
from functools import partial
from timm.models import layers

import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import DropPath,create_conv2d, create_pool2d, to_ntuple,trunc_normal_,create_classifier
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.parameter import Parameter
from .helpers import build_model_with_cfg, named_apply
from . import patch_emb as peb
from .registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': [14, 14],
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # (weights from official Google JAX impl)
    'nest_gmlp_b': _cfg(),
    'nest_gmlp_s': _cfg(),
    'nest_gmlp_t': _cfg(),
    'nest_scgmlp_b': _cfg(),
    'nest_scgmlp_s': _cfg(),
    'nest_scgmlp_t': _cfg(),
}



class SCGatingUnit(nn.Module):
    
    def __init__(self, dim, seq_len,chunks=2,norm_layer=nn.LayerNorm,gamma= 16,splat = True,**kwargs):
        super().__init__()
        self.chunks = 3
        self.seq_len=seq_len
        self.with_att= False
        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        self.proj_s = nn.Linear(self.seq_len,self.seq_len)
        self.splat= splat
        if splat :
            self.proj_c = SplAtConv1d(self.gate_dim,self.gate_dim,1,stride=1,groups=4,radix=1,reduction_factor=4)
        else:
            self.proj_c = nn.Sequential(*[
                nn.Linear(self.gate_dim,self.gate_dim//gamma),
                nn.Linear(self.gate_dim//gamma,self.gate_dim)
            ])
        self.proj_c_s = nn.Linear(self.gate_dim,self.gate_dim//gamma)
        self.proj_c_e = nn.Linear(self.gate_dim//gamma,self.gate_dim)

        self.norm_s = norm_layer(self.gate_dim)
        self.norm_c = norm_layer(self.seq_len)
        self.act = nn.GELU()
        self.sft = nn.Softmax(-1)
        self.init_weights()
        # self.se = Spatial_SE(self.seq_len)
    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj_s.weight, std=1e-6)
        nn.init.ones_(self.proj_s.bias)
    def forward(self, x):
        # B N C
        att = self.att(x) if self.with_att else 0
        if self.pad_dim:
            x = x[:,:,:-self.pad_dim]
        u,v,w = x.chunk(3, dim=-1)
        # b,sq,gd
        u = self.norm_s(u)
        u = self.act(self.proj_s(u.transpose(-1, -2)))
        uv = u.transpose(-1, -2)*v


        if self.splat:
            w = self.proj_c(self.norm_c(w.transpose(-1,-2)))
            w = self.act(w.transpose(-1,-2))
        else:
            w = self.norm_c(w.transpose(-1,-2)).transpose(-1,-2)
            w = self.act(self.proj_c_e(self.act(self.proj_c_s(w))))
        
        wv = w*v
        uwv=(uv+wv)/2

        # uwv=self.se(w)*uv
        return uwv
class SGatingUnit(nn.Module):
    
    def __init__(self, dim, seq_len,chunks=2, norm_layer=nn.LayerNorm,pos_emb=True,num_blocks=1,quadratic=False,blockwise=False,**kwargs):
        super().__init__()
        self.chunks = 2
        self.wh =int(math.pow(seq_len,0.5))
        self.pos=pos_emb
        self.blocks=num_blocks if blockwise else 1
        if blockwise: 
            print("using blockwise method..")
        self.seq_len=seq_len
        self.quadratic = quadratic
        if self.pos:
            if self.quadratic:
                self.local_init(self.blocks)
                self.get_rel_indices(self.seq_len)
            else:
                self.rel_locl_init(self.blocks)

        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        # self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len) for i in range(chunks-1)])
        self.norm= norm_layer(self.gate_dim)
        
        self.block_proj_n_weight = nn.Parameter(torch.zeros(self.blocks, seq_len, seq_len))
        self.block_proj_n_bias = nn.Parameter(torch.ones(self.blocks, seq_len,1))
        trunc_normal_(self.block_proj_n_weight,std=1e-6)

    # def init_weights(self):
    #     # special init for the projection gate, called as override by base model init
    #     for proj in self.proj_list:
    #         nn.init.normal_(proj.weight, std=1e-6)
    #         nn.init.ones_(proj.bias)

    def rel_locl_init(self,num_blocks):
                                # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.wh - 1) * (2 * self.wh - 1), num_blocks))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.wh)
        coords_w = torch.arange(self.wh)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.wh - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.wh - 1
        relative_coords[:, :, 0] *= 2 * self.wh - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def local_init(self,blocks):
        

        # self.attention_centers = nn.Parameter(
        #     torch.zeros(1, 2).normal_(0.0, 1e-4)
        # )
        self.pos_proj = nn.Parameter(torch.randn([3,blocks]))
        # self.pos_proj = torch.cat([-torch.ones([1,1]),self.attention_centers],dim=1)
        # self.spread = 1+ nn.Parameter(torch.zeros(1).normal_(0.0, 1e-4))

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices   = torch.zeros(num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,0] = indx.unsqueeze(0)
        self.register_buffer("rel_indices", rel_indices)

    def forward(self, x):
        # B W N C
        B,W,N,C = x.size()
        x_chunks = x.chunk(2, dim=-1)
        u = x_chunks[0]
        if self.pos:
            if self.quadratic:

                # W,N,N
                pos_score = torch.einsum('mnc,cw->wmn',self.rel_indices,self.pos_proj)
                relative_position_bias = pos_score
            else:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.wh * self.wh, self.wh * self.wh, -1)  # Wh*Ww,Wh*Ww,block
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # block, Wh*Ww, Wh*Ww

        v = x_chunks[1]
        u = self.norm(u)
        pos_bias = torch.einsum('wmn,bwnc->bwmc',relative_position_bias,u) if self.pos else 0 
        
        # bwmc
        u = torch.einsum('wmn,bwnc->bwmc',self.block_proj_n_weight,u) + self.block_proj_n_bias.unsqueeze(0) + pos_bias
        u = u * v
        return u

class GmlpLayer(nn.Module):
    def __init__(self, dim, seq_length,gate_unit=SGatingUnit, num_blocks=1,
    chunks = 2, mlp_ratio=4., drop=0., drop_path_rates=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()  
        chunks = chunks if gate_unit is SGatingUnit else 3 # 因为我们限制了SCG的splits数为3
        self.dim =dim
        self.norm = norm_layer(dim)
        self.hidden_dim = int(mlp_ratio * dim)
        self.proj_c_e = nn.Linear(self.dim,self.hidden_dim)
        self.gate_unit = gate_unit(dim=self.hidden_dim, seq_len = seq_length, num_blocks=num_blocks,chunks=chunks, norm_layer=norm_layer,**kwargs)

        self.hidden_dim = self.hidden_dim // chunks

        self.proj_c_s = nn.Linear(self.hidden_dim,self.dim)
        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0. else nn.Identity()
        self.act = act_layer()

    def forward_unit(self,x):
        x = self.gate_unit(x)
        return x

    def forward(self,x):
        # Input : x (1,b,n,c)
        residual = x
        x = self.act(self.proj_c_e(self.norm(x)))
        x = self.forward_unit(x)
        x = self.drop(self.proj_c_s(x))
        return self.drop_path(x) + residual
        
class ConvPool(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, pad_type='',depth_conv = True,**kwargs):
        super().__init__()
        if depth_conv:
            print("using depth_con instead!")
            self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type,depthwise=True, bias=True)
        else:
            self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = norm_layer(out_channels)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        assert x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims'
        assert x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims'
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)

def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, H, W, C)
        block_size (int): edge length of a single square block in units of H, W
    """
    B, H, W, C  = x.shape
    assert H % block_size == 0, '`block_size` must divide input height evenly'
    assert W % block_size == 0, '`block_size` must divide input width evenly'
    grid_height = H // block_size
    grid_width = W // block_size
    x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    return x  # (B, T, N, C)

def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired H, W
    """
    B, T, _, C = x.shape
    grid_size = int(math.sqrt(T))
    height = width = grid_size * block_size
    x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x = x.transpose(2, 3).reshape(B, height, width, C)
    return x  # (B, H, W, C)


class NestLevel(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """
    def __init__(
            self, num_blocks, block_size,gate_unit, seq_length, depth, embed_dim, prev_embed_dim=None,
            mlp_ratio=4., drop_rate=0., drop_path_rates=[],
            norm_layer=None, act_layer=None, pad_type='',**kwargs):
        super().__init__()
        self.block_size = block_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type,**kwargs)
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.encoder = nn.Sequential(*[
            GmlpLayer(dim= embed_dim,num_blocks=num_blocks,seq_length=seq_length,gate_unit=gate_unit,mlp_ratio=mlp_ratio,
            drop = drop_rate,drop_path_rates=drop_path_rates[i],
            norm_layer=norm_layer,act_layer=act_layer,**kwargs)
            # TransformerLayer(
            #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rates[i],
            #     norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, C, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = x + self.pos_embed
        x = self.encoder(x)  # (B, T, N, C')
        x = deblockify(x, self.block_size)  # (B, H', W', C')
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 3, 1, 2)  # (B, C, H', W')


class Nest(nn.Module):
    """ Nested Transformer (NesT)
    A PyTorch impl of : `Aggregating Nested Transformers`
        - https://arxiv.org/abs/2105.12723
    """

    def __init__(self, img_size=224, in_chans=3, patch_size=4, num_levels=3, embed_dims=(128, 256, 512),
                  depths=(2, 2, 20), num_classes=1000, mlp_ratio=4., gate_unit=SGatingUnit,
                 drop_rate=0.,  drop_path_rate=0.5, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 pad_type='', weight_init='', global_pool='avg',stem_name = "Nest_ConvolutionalEmbed",**kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            patch_size (int): patch size
            num_levels (int): number of block hierarchies (T_d in the paper)
            embed_dims (int, tuple): embedding dimensions of each level
            depths (int, tuple): number of transformer layers for each level
            num_classes (int): number of classes for classification head
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim for MLP of transformer layers
            drop_rate (float): dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer for transformer layers
            act_layer: (nn.Module): activation layer in MLP of transformer layers
            pad_type: str: Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init: (str): weight init scheme
            global_pool: (str): type of pooling operation to apply to final feature map
            
            optional(SCguit):
            gamma= 16, 
            splat = True,
        Notes:
            - Default values follow NesT-B from the original Jax code.
            - `embed_dims`, `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - For those following the paper, Table A1 may have errors!
                - https://github.com/google-research/nested-transformer/issues/2
        """
        super().__init__()

        for param_name in ['embed_dims', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f'Require `len({param_name}) == num_levels`'

        embed_dims = to_ntuple(num_levels)(embed_dims)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.feature_info = []
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], 'Model only handles square inputs'
            img_size = img_size[0]
        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size

        # Number of blocks at each level
        self.num_blocks = (4 ** torch.arange(num_levels)).flip(0).tolist()
        assert (img_size // patch_size) % math.sqrt(self.num_blocks[0]) == 0, \
            'First level blocks don\'t fit evenly. Check `img_size`, `patch_size`, and `num_levels`'

        # Block edge size in units of patches
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(self.num_blocks[0]) is the
        #  number of blocks along edge of image
        self.block_size = int((img_size // patch_size) // math.sqrt(self.num_blocks[0]))
        
        # Patch embedding
        
        stem = getattr(peb,stem_name)
        self.patch_embed = stem(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], flatten=False,**kwargs)
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]

        # Build up each hierarchical level
        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(NestLevel(
                self.num_blocks[i], self.block_size, gate_unit,self.seq_length,depths[i], dim, prev_embed_dim=prev_dim,
                mlp_ratio=mlp_ratio,  drop_rate=drop_rate, drop_path_rates = dp_rates[i], norm_layer=norm_layer, act_layer=act_layer, pad_type=pad_type,**kwargs))
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f'levels.{i}')]
            prev_dim = dim
            curr_stride *= 2
        self.levels = nn.Sequential(*levels)

        # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        # Classifier
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {f'level.{i}.pos_embed' for i in range(len(self.levels))}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.patch_embed(x)
        x = self.levels(x)
        # Layer norm done over channel dim only (to NHWC and back)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.head(x)


def _init_nest_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new):
    """
    Rescale the grid of position embeddings when loading from state_dict
    Expected shape of position embeddings is (1, T, N, C), and considers only square images
    """
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    seq_length_old = posemb.shape[2]
    num_blocks_new, seq_length_new = posemb_new.shape[1:3]
    size_new = int(math.sqrt(num_blocks_new*seq_length_new))
    # First change to (1, C, H, W)
    posemb = deblockify(posemb, int(math.sqrt(seq_length_old))).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=[size_new, size_new], mode='bicubic', align_corners=False)
    # Now change to new (1, T, N, C)
    posemb = blockify(posemb.permute(0, 2, 3, 1), int(math.sqrt(seq_length_new)))
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ resize positional embeddings of pretrained weights """
    pos_embed_keys = [k for k in state_dict.keys() if k.startswith('pos_embed_')]
    for k in pos_embed_keys:
        if state_dict[k].shape != getattr(model, k).shape:
            state_dict[k] = resize_pos_embed(state_dict[k], getattr(model, k))
    return state_dict



def _create_nest(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    model = build_model_with_cfg(
        Nest, variant, pretrained,
        default_cfg=default_cfg,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


@register_model
def nest_gmlp_b(pretrained=False, **kwargs):
    """ Nest-B @ 224x224
    """
    model_kwargs = dict(
        embed_dims=(128, 256, 512), depths=(2, 2, 20), **kwargs)
    model = _create_nest('nest_gmlp_b', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_gmlp_s(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20),chunks=2,**kwargs)
    model = _create_nest('nest_gmlp_s', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_gmlp_t(pretrained=False, **kwargs):
    """ Nest-T @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384), depths=(2, 2, 8), **kwargs)
    model = _create_nest('nest_gmlp_t', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def nest_scgmlp_b(pretrained=False, **kwargs):
    """ Nest-B @ 224x224
    """
    model_kwargs = dict(
        embed_dims=(128, 256, 512), depths=(2, 2, 20), gamma= 8,splat = False,gate_unit=SCGatingUnit, **kwargs)
    model = _create_nest('nest_scgmlp_b', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_scgmlp_s(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20), gamma= 8,splat = False,gate_unit=SCGatingUnit, **kwargs)
    model = _create_nest('nest_scgmlp_s', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_scgmlp_t(pretrained=False, **kwargs):
    """ Nest-T @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384), depths=(2, 2, 8), gamma= 8,splat = False,gate_unit=SCGatingUnit,**kwargs)
    model = _create_nest('nest_scgmlp_t', pretrained=pretrained, **model_kwargs)
    return model
