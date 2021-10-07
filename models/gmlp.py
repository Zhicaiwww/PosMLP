
import math
from functools import partial
import einops
import torch
from torch._C import ErrorReport, NoneType
import torch.nn as nn
import math
from torch.nn.modules.linear import Identity

from .patch_emb import *
from .helpers import build_model_with_cfg, named_apply
from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, lecun_normal_, to_ntuple,to_2tuple,trunc_normal_
from einops import rearrange

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

        
class SGatingUnit(nn.Module):
    
    def __init__(self, dim, seq_len,chunks=2, norm_layer=nn.LayerNorm,pos_emb=True,num_blocks=1,quadratic=True,blockwise=False,
    blocksplit=False,channel_split=24,pos_only=True,**kwargs):
        super().__init__()
        self.chunks = 2
        self.wh =int(math.pow(seq_len,0.5))
        self.pos=pos_emb
        self.blocksplit=blocksplit if num_blocks>1 else False
        self.blocks=num_blocks 
        self.win_blocks=num_blocks if blockwise else 1
        self.seq_len=seq_len
        self.quadratic = quadratic
        self.channel_split = 1
        self.pos_only=pos_only

        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        # self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len) for i in range(chunks-1)])
        self.token_proj_n_bias = nn.Parameter(torch.zeros(self.win_blocks, seq_len,1))
        if self.pos:
            if self.quadratic:
                self.att_std = 1e-4
                self.channel_split = self.gate_dim//channel_split
                self.get_quadratic_rel_indices(self.seq_len,self.blocks,channel_split=self.channel_split)

            else:
                self.rel_locl_init(self.wh,self.win_blocks,register_name='window')
                
            if self.blocksplit:
                self.rel_locl_init(int(math.pow(self.blocks,0.5)),1,register_name='block')
                self.block_lamb=nn.Parameter(torch.Tensor([1]))
                self.split_lamb=nn.Parameter(torch.Tensor([1]))
                self.pool = nn.AvgPool2d
                self.block_proj_n_weight = nn.Parameter(torch.zeros(1, self.blocks, self.blocks))
                self.block_proj_n_bias = nn.Parameter(torch.zeros(1, self.blocks,1))
                trunc_normal_(self.block_proj_n_weight,std=1e-6)
            self.init_bias_table()

        if not self.pos_only:
            self.norm= norm_layer(self.gate_dim)
            self.window_lamb=nn.Parameter(2*torch.ones(self.win_blocks))
            self.sig = nn.Sigmoid()
            self.token_proj_n_weight = nn.Parameter(torch.zeros(self.win_blocks, seq_len, seq_len))
            trunc_normal_(self.token_proj_n_weight,std=1e-6)


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

    # def local_init(self,blocks):
        

        # self.attention_centers = nn.Parameter(
        #     torch.zeros(1, 2).normal_(0.0, 1e-4)
        # )
        # self.pos_proj = torch.cat([-torch.ones([1,1]),self.attention_centers],dim=1)
        # self.spread = 1+ nn.Parameter(torch.zeros(1).normal_(0.0, 1e-4))

    def get_quadratic_rel_indices(self, num_patches,blocks,channel_split=1):
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
            torch.zeros(blocks*channel_split, 2).normal_(0.0,self.att_std)
        )
        attention_spreads = torch.eye(2).unsqueeze(0).repeat(blocks*channel_split, 1, 1)
        attention_spreads += torch.zeros_like(attention_spreads).normal_(0,self.att_std)
        self.attention_spreads = nn.Parameter(attention_spreads)

    def get_quadratic_pos_proj(self):
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
        return pos_proj

    def forward_pos(self):
        relative_position_bias = 0
        block_relative_position_bias = 0
        if self.pos:
            if self.quadratic:
                # B,D
                pos_proj = self.get_quadratic_pos_proj()
                # bs m n
                pos_score = torch.einsum('mnd,bd->bmn',self.rel_indices,pos_proj)
                relative_position_bias = nn.Softmax(-1)(pos_score)
            else:
                relative_position_bias = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
                    self.wh * self.wh, self.wh * self.wh, -1)  # Wh*Ww,Wh*Ww,block
                relative_position_bias =  relative_position_bias.permute(2, 0, 1).contiguous()  # block, Wh*Ww, Wh*Ww
            if self.blocksplit:
                block_relative_position_bias = self.block_relative_position_bias_table[self.block_relative_position_index.view(-1)].view(
                    self.blocks ,self.blocks , -1)  # block*block,block*block,1
                block_relative_position_bias = nn.Softmax(-1)(block_relative_position_bias.permute(2, 0, 1).contiguous())  # 1,block^2,block^2
        return relative_position_bias,block_relative_position_bias

    def forward(self, x):
        # B W N C
        x = x.unsqueeze(1)
        B,W,N,C = x.size()
        x_chunks = x.chunk(2, dim=-1)
        u = x_chunks[0]
        v = x_chunks[1]
        # u = self.norm(u)
        # b n n , 1 b b
        u = rearrange(u,'b w n (v s) -> b w n v s', s = self.channel_split)
        relative_position_bias, block_relative_position_bias = self.forward_pos()
        relative_position_bias = rearrange(relative_position_bias,'(s b) m n  -> b m n s', s = self.channel_split)
        # pos_bias = torch.einsum('wmn,bwnc->bwmc',relative_position_bias,u) if self.pos else torch.zeros_like(u)  
        # wmns
        if self.pos_only:
            win_weight=relative_position_bias
        else:
            win_weight = torch.einsum('w,wmns->wmns',1-self.sig(self.window_lamb),self.token_proj_n_weight.unsqueeze(-1))+\
            torch.einsum('w,wmns->wmns',1-self.sig(self.window_lamb),relative_position_bias)
        win_weight = win_weight
        u_1 = rearrange(torch.einsum('wmns,bwnvs->bwmvs',win_weight,u),'b w n v s -> b w n (v s)') + self.token_proj_n_bias.unsqueeze(0)
        u_1 = u_1 * v
        if self.blocksplit:
            u_2 = u.mean(2).unsqueeze(2)
            # block_pos_bias = torch.einsum('lmn,bnlc->bmlc', block_relative_position_bias,u_2) if self.pos else torch.zeros_like(u_2)
            block_weight = torch.einsum('w,wmn->wmn',1-self.sig(self.block_lamb),self.block_proj_n_weight)+\
            torch.einsum('w,wmn->wmn',1-self.sig(self.block_lamb),block_relative_position_bias)
            u_2 = torch.einsum('lmn,bnlc->bmlc',block_weight,u_2) + self.block_proj_n_bias.unsqueeze(-1)
            # b w 1 c
            u_2 = u_2 * v.mean(2).unsqueeze(2)
            # print(u_2.size())
            gating = self.split_lamb.view(1,-1,1,1)
            u =  torch.sigmoid(gating) * u_1 +(1.-torch.sigmoid(gating)) * u_2
 
            return u
        else:
            return u_1.squeeze(1)


class SpatialGatingUnit(nn.Module):

    def __init__(self, dim, seq_len,chunks=2,with_att = True, norm_layer=nn.LayerNorm, layer_idx=None):
        super().__init__()
        self.chunks = chunks
        self.with_att= with_att
        self.gate_dim = dim // chunks
        self.pad_dim = dim % chunks # if cant divided by chunks, cut the residual term
        
        bias = layer_idx <=25 if layer_idx else True
        self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len,bias=bias) for i in range(chunks-1)])
        self.norm_list = nn.Sequential(*[norm_layer(self.gate_dim)for i in range(chunks-1)])
        # self.init_weights()

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        for proj in self.proj_list:
            nn.init.normal_(proj.weight, std=1e-6)
            if proj.bias is not None:
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
    
    def __init__(self, dim, seq_len,chunks=2,with_att = True, gamma= 8,norm_layer=nn.LayerNorm):
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

class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.,**kwargs):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class SelfGatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features,hidden_features=None,seq_len=196, out_features=None,with_att=True, act_layer=nn.GELU
                 , chunks=2,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop=0.):
        super().__init__()
        self.chunks= chunks
        out_features = out_features or in_features
        self.norm = norm_layer(out_features)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.gate_dim = hidden_features // chunks  
        # self.pad_dim = hidden_features % chunks
        # self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len) for i in range(chunks-1)])
        # self.norm_list = nn.Sequential(*[norm_layer(self.gate_dim)for i in range(chunks-1)])
        # self.reweight = Mlp(self.gate_dim, self.gate_dim // 4, self.gate_dim *self.chunks)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.gate= SGatingUnit(in_features,seq_len)

    def forward_gate(self, x):
        # B N C
        return self.gate(x)
        # if self.pad_dim:
        #     x = x[:,:,:-self.pad_dim]
        # x_chunks = x.chunk(self.chunks, dim=-1)
        # u = x_chunks[0]
        # out =[]
        # out.append(u)
        # for i in range(self.chunks-1):
        #     v = x_chunks[i+1]
        #     u = self.norm_list[i](u)
        #     u = self.proj_list[i](u.transpose(-1, -2))
        #     u = u.transpose(-1, -2)
        #     u = u * v
        #     out.append(u)
        # return torch.cat(out,dim=-1)

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
                 gate_layer=SpatialGatingUnit, chunks=2,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop=0.,**kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            self.gate = gate_layer(hidden_features,chunks=chunks,norm_layer=norm_layer,**kwargs)
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

class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=6, chunks=2,  with_att=True,mlp_layer=GatedMlp,gate_unit=SpatialGatingUnit, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.,**kwargs):
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
        self.mlp_channels = mlp_layer(dim, gate_dim[0], chunks=chunks, act_layer=act_layer,  with_att= with_att, gate_layer=gate_unit,norm_layer=norm_layer,drop=drop,**kwargs)
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
            with_att = False,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
    
        self.stem =stem(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer= norm_layer )
        self.seq_len = self.stem.num_patches
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.seq_len ,mlp_ratio= mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer,chunks= chunks,with_att=with_att,gate_unit=gate_unit, drop=drop_rate, drop_path=drop_path_rate,layer_idx=i,**kwargs)
            for i in range(num_blocks)])
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
def gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, chunks=2,block_layer=SpatialGatingBlock, gate_unit=SGatingUnit,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model
