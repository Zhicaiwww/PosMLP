
import math
from functools import partial
import einops
import torch
from torch._C import ErrorReport, NoneType
import torch.nn as nn
import math
from torch.nn.modules.linear import Identity

from .layers import PatchEmbed,ConvolutionalEmbed
from .helpers import build_model_with_cfg, named_apply
from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, lecun_normal_, to_ntuple

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = dict(
    gmlp_ti16_224=_cfg(),
    Mgmlp_ti16_224=_cfg(),
    gmlp_s16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
    ),
    Mgmlp_b16_224=_cfg(),
    Mgmlp_s16_224=_cfg(),
)

class SpatialGatingUnit(nn.Module):

    def __init__(self, dim, seq_len,chunks=2,with_att = True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.chunks = chunks
        self.with_att= with_att
        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len) for i in range(chunks-1)])
        self.norm_list = nn.Sequential(*[norm_layer(self.gate_dim)for i in range(chunks-1)])
        if with_att:
            self.att = Tiny_Att(dim,self.gate_dim)
        # self.init_weights()

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        for proj in self.proj_list:
            nn.init.normal_(proj.weight, std=1e-6)
            nn.init.ones_(proj.bias)

    def forward(self, x):
        # B N C
        att = self.att(x) if self.with_att else 0
        if self.pad_dim:
            x = x[:,:,:-self.pad_dim]
        x_chunks = x.chunk(self.chunks, dim=-1)
        u = x_chunks[0]
        for i in range(self.chunks-1):
            v = x_chunks[i+1]
            u = self.norm_list[i](u)
            u = self.proj_list[i](u.transpose(-1, -2))
            if i == self.chunks -1:
                u = u.transpose(-1, -2) + att
            else:
                u = u.transpose(-1, -2)
            u = u * v
        return u

class SpaticialChannelGU(nn.Module):
    
    def __init__(self, dim, seq_len,chunks=2,with_att = True, gamma= 16,norm_layer=nn.LayerNorm):
        super().__init__()
        self.chunks = 3
        self.seq_len=seq_len
        self.with_att= False
        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        self.proj_s = nn.Linear(self.seq_len,self.seq_len)
        self.proj_c_s = nn.Linear(self.gate_dim,self.gate_dim//gamma)
        self.proj_c_e = nn.Linear(self.gate_dim//gamma,self.gate_dim)    
        self.norm_s = norm_layer(self.gate_dim)
        self.norm_c = norm_layer(self.seq_len)
        self.act = nn.ReLU()
        self.sft = nn.Softmax(-1)
        if with_att:
            self.att = Tiny_Att(dim,self.gate_dim)
        self.init_weights()
        # self.se = Spatial_SE(self.seq_len)
    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj_s.weight, std=1e-6)
        nn.init.ones_(self.proj_s.bias)
        nn.init.normal_(self.proj_c_s.weight, std=1e-6)
        nn.init.ones_(self.proj_c_s.bias)
        nn.init.normal_(self.proj_c_e.weight, std=1e-6)
        nn.init.ones_(self.proj_c_e.bias)
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

        w = self.norm_c(w.transpose(-1,-2)).transpose(-1,-2)
        w = self.act(self.proj_c_e(self.proj_c_s(w)))
        
        wv = w*v
        uwv=(uv+wv)/2

        # uwv=self.se(w)*uv
        return uwv

class Spatial_SE(nn.Module):
    def __init__(self,in_dim,out_dim=None,pool_way = 'mean',time_pool = False,div=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.fc_1 = nn.Linear(self.in_dim,self.out_dim//div,bias=False)
        self.fc_2 = nn.Linear(self.out_dim//div,self.out_dim,bias=False)
        if pool_way=='mean':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_way=='max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        self.time_pool = time_pool
        self.sig = nn.Sigmoid()
        self.act_fun = nn.GELU()


    def forward(self,input):
        att = self.pool(input).squeeze(-1)
        att = self.fc_2(self.act_fun(self.fc_1(att)))
        att = einops.rearrange(att,'b n -> b n 1 ')
        return self.sig(att)

class Indentity(nn.Module):
    def __init__(self,out) -> None:
        super().__init__()
        self.out =out 
    def forward(self,input):
        return self.out

class MultiHeadAttentionUnit(nn.Module):
    
    def __init__(self, dim, seq_len,chunks=4,with_att = True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.chunks = chunks
        self.seq_len=seq_len
        self.with_att= False
        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        self.mlat=MultiHeadAttention(4,self.gate_dim)
        # self.init_weights()
        # self.se = Spatial_SE(self.seq_len)
    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        # TODO
        pass

    def forward(self, x):
        # B N C
        q,k,v=x.chunk(self.chunks,dim=-1)
        x =self.mlat(q,k,v)

        # uwv=self.se(w)*uv
        return x

class PrepareForMultiHeadAttention(nn.Module):
    """
    ## Prepare for multi-head attention
    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def __call__(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        # Output has shape `[batch_size, seq_len, heads, d_k]` or `[batch_size, d_model]`
        return x

class MultiHeadAttention(nn.Module):


    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True,
                 mask_type: str = 'softmax'):
        super().__init__()
        # Number of features per head
        self.d_k = d_model // heads
        self.heads = heads
        #(B,m,heads,d_k)
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        if mask_type == 'softmax':
            self.selector = nn.Softmax(dim=1)
        else:
            raise NotImplemented()

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        # Calculate $Q K^\top$
        return torch.einsum('bihd,bjhd->bijh', query, key)

    def forward(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask = None):

        # `query`, `key` and `value`  have shape `[batch_size, seq_len, d_model]`
        batch_size, seq_len, _ = query.shape

        if mask is not None:
            assert mask.shape[1] == 1 or mask.shape[1] == mask.shape[2]
            mask = mask.unsqueeze(-1)

        # Prepare `query`, `key` and `value` for attention computation
        # These will then have shape `[batch_size, seq_len, heads, d_k]`
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.selector(scores)
        attn = self.dropout(attn)
        # Multiply by values
        x = torch.einsum("bijh,bjhd->bihd", attn, value)
        # Save attentions for any other calculations
        self.attn = attn.detach()
        x = x.reshape(batch_size, seq_len, -1)
        # x = x.mean(-2)
        # (b,n,c)
        return self.output(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfGatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features,hidden_features=None,seq_len=196, out_features=None,with_att=True, act_layer=nn.GELU,
                 gate_layer=SpatialGatingUnit, chunks=2,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop=0.):
        super().__init__()
        self.chunks= chunks
        out_features = out_features or in_features
        self.norm = norm_layer(out_features)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.gate_dim = hidden_features // chunks  
        self.pad_dim = hidden_features % chunks
        self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len) for i in range(chunks-1)])
        self.norm_list = nn.Sequential(*[norm_layer(self.gate_dim)for i in range(chunks-1)])
        self.reweight = Mlp(self.gate_dim, self.gate_dim // 4, self.gate_dim *self.chunks)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward_gate(self, x):
        # B N C

        if self.pad_dim:
            x = x[:,:,:-self.pad_dim]
        x_chunks = x.chunk(self.chunks, dim=-1)
        u = x_chunks[0]
        out =[]
        out.append(u)
        for i in range(self.chunks-1):
            v = x_chunks[i+1]
            u = self.norm_list[i](u)
            u = self.proj_list[i](u.transpose(-1, -2))
            u = u.transpose(-1, -2)
            u = u * v
            out.append(u)
        return torch.cat(out,dim=-1)

    def forward(self, x):
        # x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.forward_gate(x)
        x = self.fc2(x)
        x = self.drop(x)
        # print(x.size())
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features,hidden_features=None, out_features=None,with_att=True, act_layer=nn.GELU,
                 gate_layer=SpatialGatingUnit, chunks=2,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            self.gate = gate_layer(hidden_features,chunks=chunks,with_att=with_att,norm_layer=norm_layer)
            hidden_features = hidden_features // chunks  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gate(x)
        # print(x.size())
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Tiny_Att(nn.Module):
    def __init__(self,in_features, out_features=None,d_attn=64):
        super().__init__()
        out_features = out_features or in_features
        self.d_attn = d_attn
        self.proj1 = nn.Linear(in_features,3*d_attn)
        self.softmax = nn.Softmax(dim=-1)
        self.proj2=nn.Linear(d_attn,out_features)
    def forward(self,input):
        # input = einops.rearrange(input,'b s n c')
        q,k,v=self.proj1(input).chunk(3,dim=-1)
        w = torch.einsum("bsnd,bsmd->bsnm", q, k)
        a = self.softmax(w/pow(self.d_attn,0.5))
        x = torch.einsum("bsnm,bsmd->bsnd", a, v)
        x = self.proj2(x)
        # print(x)
        return x 

class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=6, chunks=2,  with_att=True,mlp_layer=GatedMlp,gate_unit=SpatialGatingUnit, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()

        if isinstance(mlp_ratio, int):
            gate_dim = [int(mlp_ratio*dim)]
        elif isinstance(mlp_ratio,(list,tuple,)):
            to_tuple = to_ntuple(len(mlp_ratio))
            gate_dim = [int(x * dim) for x in to_tuple(mlp_ratio)]
        else:
            NotImplementedError
        # gate_dim = [1024,256]
        self.norm = norm_layer(dim)
        gate_unit = partial(gate_unit, seq_len=seq_len)
        # tgu = partial(TimeGatingUnit, segments=segments)
        self.mlp_channels = mlp_layer(dim, gate_dim[0], chunks=chunks, act_layer=act_layer,  with_att= with_att, gate_layer=gate_unit,norm_layer=norm_layer,drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x

class MlpMixer(nn.Module):
    
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=SpatialGatingBlock, 
            mlp_layer=GatedMlp,
            gate_unit=SpatialGatingUnit,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem = PatchEmbed,
            stem_norm=partial(nn.BatchNorm2d, eps=1e-6),
            chunks = 2,
            with_att = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
    
        self.stem =stem(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer= stem_norm )
        self.seq_len = self.stem.num_patches
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.seq_len , mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer,chunks= chunks,with_att=with_att,gate_unit=gate_unit, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def freeze_part_param(self):
        for name,para in self.named_parameters():
            print(name,para.requires_grad)
    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        if 'gate.norm' in k:
            norm_dict = {}
            norm_dict[k.replace('gate.norm','gate.norm_list')]=v
            norm_dict[k.replace('gate.norm','gate.norm_list.0')]=v
            norm_dict[k.replace('gate.norm','gate.norm_list.1')]=v
            out_dict.update(norm_dict)
        if 'gate.proj' in k:
            proj_dict = {}
            proj_dict[k.replace('gate.proj','gate.proj_list')]=v
            proj_dict[k.replace('gate.proj','gate.proj_list.0')]=v
            proj_dict[k.replace('gate.proj','gate.proj_list.1')]=v
            out_dict.update(proj_dict)
        return out_dict
    return state_dict

def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()

def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model

@register_model
def gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, chunks=2,block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def SCG3gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128,chunks=3, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpaticialChannelGU,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def ATT4gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128,chunks=3, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=MultiHeadAttentionUnit,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def SCG3gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256,chunks=3, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpaticialChannelGU,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def SCG3gmlp_s16_224_2(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256,chunks=3, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpaticialChannelGU,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def SCG3gmlp_convstem_s16_224_2(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=29, stem =ConvolutionalEmbed,stem_norm=partial(nn.BatchNorm2d, eps=1e-6), embed_dim=256,chunks=3, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpaticialChannelGU,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model
@register_model
def M4gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128,chunks=4, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def M3gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,chunks =3,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained,pretrained_strict=False, **model_args)
    return model

# multi sgu-stage
@register_model
def M3gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,chunks =3,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained,pretrained_strict=False, **model_args)
    return model
@register_model
def SelfGate3_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=3, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,chunks =3,
        mlp_layer=SelfGatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained,pretrained_strict=False, **model_args)
    return model

@register_model
def M4gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small with 4 chunks
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit, chunks =4,
        mlp_layer=GatedMlp,pretrained_strict=False, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model
@register_model
def M3Agmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small with 3 chunks and att 
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,chunks =3,with_att =True,
        mlp_layer=GatedMlp,pretrained_strict=False, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def M4Agmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small with 4 chunks and att 
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,chunks =4, with_att =True,
        mlp_layer=GatedMlp,pretrained_strict=False, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit,    chunks = 2,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def Mgmlp_b16_224(pretrained=False, **kwargs):
    """ gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict( block_layer=SpatialGatingBlock, gate_unit=SpatialGatingUnit, mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
    return model
