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
from einops import rearrange

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
    'nest_gmlp_s4':_cfg()
}


class QuaMap(nn.Module):
    def __init__(self,dim, seq_len, blocks,gamma=4, use_softmax=True,att_std = 1e-4,**kwargs):
        super().__init__()
        self.dim = dim
        self.att_std = att_std 
        self.seq_len = seq_len
        self.blocks = blocks
        self.gamma = gamma
        self.use_softmax = use_softmax
        self.channel_split = self.dim//self.gamma
        self.get_quadratic_rel_indices(self.seq_len,self.blocks,gamma=self.gamma)
        self.token_proj_n_bias = nn.Parameter(torch.zeros(self.blocks*self.gamma, seq_len,1))

    def get_quadratic_rel_indices(self, num_patches,blocks,gamma=1):
        img_size = int(num_patches**.5)
        rel_indices   = torch.zeros(num_patches, num_patches,5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indxx = indx**2 
        indyy = indy**2
        indxy = indx * indy
        rel_indices[:,:,4] = torch.sigmoid(indxy.unsqueeze(0))
        rel_indices[:,:,3] = indyy.unsqueeze(0)      
        rel_indices[:,:,2] = indxx.unsqueeze(0)
        rel_indices[:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,0] = indx.unsqueeze(0)
        self.register_buffer("rel_indices", rel_indices)
        self.attention_centers = nn.Parameter(
            torch.zeros(blocks*gamma, 2).normal_(0.0,self.att_std)
        )
        attention_spreads = torch.eye(2).unsqueeze(0).repeat(blocks*gamma, 1, 1)
        attention_spreads += torch.zeros_like(attention_spreads).normal_(0,self.att_std)
        self.attention_spreads = nn.Parameter(attention_spreads)



    def forward_pos(self):
        relative_position_bias = 0
        # B,D
        mu_1, mu_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]
        inv_covariance = torch.einsum('hij,hkj->hik', [self.attention_spreads, self.attention_spreads])
        a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]

        # bs,5
        pos_proj =-1/2 * torch.stack([
            -2*(a*mu_1 + b*mu_2),
            -2*(c*mu_2 + b*mu_1),
            a,
            c,
            2 * b
        ], dim=-1)

        # bs m n
        pos_score = torch.einsum('mnd,bd->bmn',self.rel_indices,pos_proj)
        if self.use_softmax:
            relative_position_bias = nn.Softmax(-1)(pos_score)
        else: 
            relative_position_bias = pos_score
        return relative_position_bias

    def forward(self,x):
        assert len(x.size()) == 4
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma)
        win_weight = self.forward_pos()
        win_weight = rearrange(win_weight,'(s b) m n  -> b m n s', s = self.gamma)
        x = torch.einsum('wmns,bwnvs->bwmvs',win_weight,x) +  rearrange(self.token_proj_n_bias,'(w s) m  v-> w m v s', s = self.gamma).unsqueeze(0)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
        return x


class LearnedPosMap(nn.Module):
    def __init__(self,dim, seq_len, blocks,gamma=4,**kwargs):
        super().__init__()
        self.blocks = blocks
        self.gamma = gamma
        self.wh =int(math.pow(seq_len,0.5))
        self.token_proj_n_bias = nn.Parameter(torch.zeros(self.blocks * self.gamma, seq_len,1))
        self.rel_locl_init(self.wh,self.blocks * self.gamma,register_name='window')
        self.init_bias_table()
        self.lamb=nn.Parameter(torch.Tensor([1]))
        self.sig = nn.Sigmoid()


    def rel_locl_init(self,wh, num_blocks, register_name='window'):
        # define a parameter table of relative position bias
        self.register_parameter(f'{register_name}_relative_position_bias_table' ,nn.Parameter(
            torch.zeros((2 * wh - 1) * (2 * wh - 1), num_blocks)))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(wh)
        coords_w = torch.arange(wh)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += wh - 1  # shift to start from 0
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 0] *= 2 * wh - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(f"{register_name}_relative_position_index", relative_position_index)
 
    def init_bias_table(self):
        for k,v in self.named_modules():
            if 'relative_position_bias_table' in k:
                trunc_normal_(v.weight, std=.02) 

    def forward_pos(self):
        relative_position_bias = 0
        relative_position_bias = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
            self.wh * self.wh, self.wh * self.wh, -1)  # Wh*Ww,Wh*Ww,block
        relative_position_bias =  relative_position_bias.permute(2, 0, 1).contiguous()  # block, Wh*Ww, Wh*Ww
        return relative_position_bias

    def forward(self,x,weight=None):
        relative_position_bias = self.forward_pos()
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma)
        win_weight = rearrange(relative_position_bias,'(s b) m n  -> b m n s', s = self.gamma)
        if weight:
            # b x N x N 
            self.lamb = self.lamb.view(1,1,1,1)
            win_weight = self.sig(1-self.lamb)*win_weight + (1-self.lamb)*weight.unsqueeze(-1)
        else:
            win_weight = win_weight
        x = torch.einsum('wmns,bwnvs->bwmvs',win_weight,x) +  rearrange(self.token_proj_n_bias,'(w s) m v-> w m v s', s = self.gamma).unsqueeze(0)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
 
        return x

class SGatingUnit(nn.Module):
    
    def __init__(self, dim, seq_len,chunks=2, norm_layer=nn.LayerNorm,pos_emb=True,num_blocks=1,quadratic=True,blockwise=False,
    blocksplit=False,gamma=16,pos_only=True,**kwargs):
        super().__init__()
        self.chunks = chunks
        self.wh =int(math.pow(seq_len,0.5))
        self.pos=pos_emb
        self.blocksplit=blocksplit if num_blocks>1 else False
        self.blocks=num_blocks 
        self.win_blocks=num_blocks if blockwise else 1
        self.gate_dim = dim // chunks
        self.seq_len=seq_len
        self.quadratic = quadratic
        self.pos_only=pos_only
        self.norm= norm_layer(self.gate_dim)

        if self.quadratic:
            self.pos=QuaMap(self.gate_dim,seq_len,num_blocks,gamma=gamma,**kwargs)
        else:
            self.pos=LearnedPosMap(self.gate_dim,seq_len,num_blocks,gamma=gamma,**kwargs)

        # if self.blocksplit:
        #     self.rel_locl_init(int(math.pow(self.blocks,0.5)),1,register_name='block')
        #     self.block_lamb=nn.Parameter(torch.Tensor([1]))
        #     self.split_lamb=nn.Parameter(torch.Tensor([1]))
        #     self.pool = nn.AvgPool2d
        #     self.block_proj_n_weight = nn.Parameter(torch.zeros(1, self.blocks, self.blocks))
        #     self.block_proj_n_bias = nn.Parameter(torch.zeros(1, self.blocks,1))
        #     trunc_normal_(self.block_proj_n_weight,std=1e-6)

        if not self.pos_only:
            self.sig = nn.Sigmoid()
            self.token_proj_n_weight = nn.Parameter(torch.zeros(self.win_blocks, seq_len, seq_len))
            trunc_normal_(self.token_proj_n_weight,std=1e-6)

    def forward(self, x):
        # B W N C
        B,W,N,C = x.size()
        if self.chunks==1:
            u = x
            v = x
        else:
            x_chunks = x.chunk(2, dim=-1)
            u = x_chunks[0]
            v = x_chunks[1]
        u = self.pos(self.norm(u))
        
        u = u * v
        # if self.blocksplit:
        #     u_2 = u.mean(2).unsqueeze(2)
        #     # block_pos_bias = torch.einsum('lmn,bnlc->bmlc', block_relative_position_bias,u_2) if self.pos else torch.zeros_like(u_2)
        #     block_weight = torch.einsum('w,wmn->wmn',1-self.sig(self.block_lamb),self.block_proj_n_weight)+\
        #     torch.einsum('w,wmn->wmn',1-self.sig(self.block_lamb),block_relative_position_bias)
        #     u_2 = torch.einsum('lmn,bnlc->bmlc',block_weight,u_2) + self.block_proj_n_bias.unsqueeze(-1)
        #     # b w 1 c
        #     u_2 = u_2 * v.mean(2).unsqueeze(2)
        #     # print(u_2.size())
        #     gating = self.split_lamb.view(1,-1,1,1)
        #     u =  torch.sigmoid(gating) * u_1 +(1.-torch.sigmoid(gating)) * u_2
 
        #       return u
        return u

class GmlpLayer(nn.Module):
    def __init__(self, dim, seq_length,gate_unit=SGatingUnit, num_blocks=1,
    chunks = 2, mlp_ratio=4., drop=0., drop_path_rates=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()  
        chunks = chunks if gate_unit is SGatingUnit else 3 # 因为我们限制了SCG的splits数为3
        self.dim =dim
        self.norm = norm_layer(dim)
        self.hidden_dim = int(mlp_ratio * dim)
        self.split_dim = self.hidden_dim // chunks
        # if not quadratic:
        self.proj_c_e = nn.Linear(self.dim,self.hidden_dim)
        self.proj_c_s = nn.Linear(self.split_dim ,self.dim)
        # else:
        #     self.proj_c_e=create_conv2d(self.dim,self.hidden_dim,kernel_size=1, groups=channel_split, bias=True)
        #     # self.proj_c_s=nn.Conv1d(self.split_dim ,self.dim,kernel_size=1, groups=channel_split, bias=True)
        #     self.proj_c_s = nn.Linear(self.split_dim ,self.dim)
        self.gate_unit = gate_unit(dim=self.hidden_dim, seq_len = seq_length, num_blocks=num_blocks,chunks=chunks, norm_layer=norm_layer,**kwargs)



        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0. else nn.Identity()
        self.act = act_layer()


    def forward(self,x):
        # Input : x (1,b,n,c)
        residual = x
        x = self.act(self.proj_c_e(self.norm(x)))
        # 1 c b n 
        x = self.gate_unit(x)
        x = self.drop(self.proj_c_s(x))
        return self.drop_path(x) + residual



# class MixerBlock(nn.Module):
#     """ Residual Block w/ token mixing and channel MLPs
#     Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
#     """
#     def __init__(
#             self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
#         super().__init__()
#         tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
#         self.norm1 = norm_layer(dim)
#         self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
#         x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
#         return x

class MixerLayer_2(nn.Module):
    def __init__(self, dim, seq_length, num_blocks=1,
         mlp_ratio=4., drop=0., drop_path_rates=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()  
        self.dim =dim
        self.norm = norm_layer(dim)
        self.hidden_dim = int(mlp_ratio * dim)
        self.mlp_channels=nn.Sequential(*[ 
            nn.Linear(self.dim,self.hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim,self.dim)])
        self.mlp_tokens = QuaMap(dim,seq_len=seq_length,blocks=num_blocks,**kwargs)
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0. else nn.Identity()

    def forward(self, x):
        #1,B,N,C
        x = x + self.drop_path(self.mlp_tokens(x))
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return self.drop_path(x) 


class MixerLayer(nn.Module):
    def __init__(self, dim, seq_length, num_blocks=1,
         mlp_ratio=4., drop=0., drop_path_rates=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()  
        self.dim =dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.hidden_dim = int(mlp_ratio * dim)
        self.mlp_channels=nn.Sequential(*[ 
            nn.Linear(self.dim,self.hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim,self.dim)])
        self.mlp_tokens = QuaMap(dim,seq_len=seq_length,blocks=num_blocks,**kwargs)
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0. else nn.Identity()
        self.mlp_tokens=nn.Sequential(*[ 
            nn.Linear(seq_length,seq_length),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(seq_length,seq_length)])

    def forward(self, x):
        #1,B,N,C
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(-1,-2))).transpose(-1,-2)
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return self.drop_path(x) 

class GmlpLayer_2(nn.Module):
    def __init__(self, dim, seq_length,gate_unit=SGatingUnit, num_blocks=1,
    chunks = 2, mlp_ratio=4., drop=0., drop_path_rates=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()  
        chunks = chunks if gate_unit is SGatingUnit else 3 # 因为我们限制了SCG的splits数为3
        self.dim =dim
        self.norm = norm_layer(dim)
        self.hidden_dim = int(mlp_ratio * dim)
        self.split_dim = self.hidden_dim // chunks

        self.proj_c_e = nn.Linear(self.dim,self.hidden_dim)
        self.proj_c_s = nn.Linear(self.hidden_dim,self.dim)
  
        self.gate_unit = gate_unit(dim=self.dim, seq_len = seq_length, num_blocks=num_blocks,chunks=chunks, norm_layer=norm_layer,**kwargs)



        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0. else nn.Identity()
        self.act = act_layer()


    def forward(self,x):
        # Input : x (1,b,n,c)
        residual = x
        x = self.act(self.proj_c_e(self.norm(x)))
        # 1 c b n 
        x = self.drop(self.proj_c_s(x))
        x = self.gate_unit(x)
        return self.drop_path(x) + residual


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape
        #assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, C, H, W)

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2 

        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.reduction(x)

        return x

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
            self, num_blocks, block_size,gate_unit, seq_length, depth, embed_dim, gate_layer=GmlpLayer,prev_embed_dim=None,
            mlp_ratio=4., drop_rate=0., drop_path_rates=[],
            norm_layer=None, act_layer=None, pad_type='',**kwargs):
        super().__init__()
        self.block_size = block_size
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:
            # self.pool=PatchMerging(prev_embed_dim)
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type,**kwargs)
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.encoder = nn.Sequential(*[
            gate_layer(dim= embed_dim,num_blocks=num_blocks,seq_length=seq_length,gate_unit=gate_unit,mlp_ratio=mlp_ratio,
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
        # print(x.size(),self.pos_embed.size())
        # x = x + self.pos_embed
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
                  depths=(2, 2, 20), num_classes=1000, mlp_ratio=4., gate_unit=SGatingUnit,gate_layer=GmlpLayer,
                 drop_rate=0.,  drop_path_rate=0.5, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 pad_type='', weight_init='', global_pool='avg',stem_name = "PatchEmbed",**kwargs):
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
                self.num_blocks[i], self.block_size, gate_unit,self.seq_length,depths[i], dim, gate_layer=gate_layer,prev_embed_dim=prev_dim,
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
        # for level in self.levels:
        #     trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
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




def _create_nest(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    model = build_model_with_cfg(
        Nest, variant, pretrained,
        default_cfg=default_cfg,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        **kwargs)

    return model


@register_model
def nest_gmlp_l(pretrained=False, **kwargs):
    """ Nest-B @ 224x224
    """
    model_kwargs = dict(
        embed_dims=(192, 384, 768), depths=(2, 2, 20),chunks=2, **kwargs)
    model = _create_nest('nest_gmlp_b', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def nest_gmlp_b(pretrained=False, **kwargs):
    """ Nest-B @ 224x224
    """
    model_kwargs = dict(
        embed_dims=(128, 256, 512), depths=(2, 2, 20),chunks=2, **kwargs)
    model = _create_nest('nest_gmlp_b', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_gmlp_s(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20),**kwargs)
    model = _create_nest('nest_gmlp_s', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_gmlp_s_v2(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20),chunks=1,gate_layer=GmlpLayer_2,mlp_ratio=3,**kwargs)
    model = _create_nest('nest_gmlp_s', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_gmlp_s_v3(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20),chunks=1,gate_layer=MixerLayer,mlp_ratio=3,**kwargs)
    model = _create_nest('nest_gmlp_s', pretrained=pretrained, **model_kwargs)
    return model

def nest_gmlp_s_v4(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20),chunks=1,gate_layer=MixerLayer_2,mlp_ratio=3,**kwargs)
    model = _create_nest('nest_gmlp_s', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def nest_gmlp_s4_p2(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(48, 96, 192, 384),  depths=(2, 2, 4, 16),num_levels=4,chunks=2,patch_size=2,**kwargs)
    model = _create_nest('nest_gmlp_s4', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def nest_gmlp_s4_p4(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384,768),  depths=(2, 4, 8, 10),num_levels=4,chunks=2,patch_size=4,**kwargs)
    model = _create_nest('nest_gmlp_s4', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def nest_gmlp_s4_p4(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384,768),  depths=(2, 4, 12, 4),num_levels=4,chunks=2,patch_size=4,**kwargs)
    model = _create_nest('nest_gmlp_s4', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def nest_gmlp_b4(pretrained=False, **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(64, 128, 256, 512),  depths=(2, 2, 2, 16),num_levels=4,chunks=2,patch_size=2,**kwargs)
    model = _create_nest('nest_gmlp_s4', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def nest_gmlp_t(pretrained=False, **kwargs):
    """ Nest-T @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384), depths=(2, 2, 8), **kwargs)
    model = _create_nest('nest_gmlp_t', pretrained=pretrained, **model_kwargs)
    return model



