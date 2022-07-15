import collections.abc
import logging
import math
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from .patch_downsample import *
from timm.models.layers import DropPath, trunc_normal_,create_classifier
from timm.models.registry import register_model
from einops import rearrange
_logger = logging.getLogger("train")

import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()



class QuaMap(nn.Module):
    def __init__(self,win_size,gamma=4,use_softmax=True,att_std = 1e-2,generalized=True,absolute_bias=True,symmetric=True,layer_idx=1,**kwargs):
        super().__init__()
        self.att_std = att_std 
        self.win_size = win_size
        self.layer_idx = layer_idx 
        self.symmetric = symmetric
        self.gamma = gamma 
        self.generalized = generalized
        self.absolute_bias = absolute_bias
        self.sft=nn.Softmax(-2) if use_softmax else nn.Identity()
        self.get_rel_indices(self.win_size,gamma=self.gamma,generalized=generalized)
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1, win_size*win_size,1))

    def get_rel_indices(self, win_size,gamma=1,generalized=True):
        
        h = win_size
        w = win_size
        num_patches = w * h

        ind = torch.arange(h).view(1,-1) - torch.arange(w).view(-1, 1)
        # hw hw
        indx = ind.repeat(h,w)
        indy = ind.repeat_interleave(h,dim=0).repeat_interleave(w,dim=1).T
        indxx = indx**2 
        indyy = indy**2
        indd = indx**2 + indy**2
        indxy = indx * indy
        if generalized:
            rel_indices   = torch.zeros(num_patches, num_patches,5)
            rel_indices[:,:,4] =  indxy.unsqueeze(0)
            rel_indices[:,:,3] = indyy.unsqueeze(0)      
            rel_indices[:,:,2] = indxx.unsqueeze(0)
            rel_indices[:,:,1] = indy.unsqueeze(0)
            rel_indices[:,:,0] = indx.unsqueeze(0)
            self.attention_centers = nn.Parameter( torch.zeros(gamma, 2).normal_(0.0,self.att_std))
            attention_spreads = torch.eye(2).unsqueeze(0).repeat(gamma, 1, 1)
            attention_spreads += torch.zeros_like(attention_spreads).normal_(0,self.att_std)
            self.attention_spreads = nn.Parameter(attention_spreads)
        else:
            rel_indices   = torch.zeros(num_patches, num_patches,3)
            rel_indices[:,:,2] = indd.unsqueeze(0)
            rel_indices[:,:,1] = indy.unsqueeze(0)
            rel_indices[:,:,0] = indx.unsqueeze(0)
            self.attention_centers = nn.Parameter( torch.zeros(gamma, 2).normal_(0.0,self.att_std))
            attention_spreads = 1 + torch.zeros(gamma).normal_(0, self.att_std)
            self.attention_spreads = nn.Parameter(attention_spreads)

        self.register_buffer("rel_indices", rel_indices)

    def forward_pos(self,mask=None):
        # B,D

        delta_1, delta_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]

        if self.generalized:
            if self.symmetric:
                inv_covariance = torch.einsum('hij,hkj->hik', [self.attention_spreads, self.attention_spreads])
            else:
                inv_covariance = self.attention_spreads

            a, b, c ,d = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1], inv_covariance[:, 1, 0]
            # bs,5
            pos_proj =-1/2 * torch.stack([
                -2*(a*delta_1 + b*delta_2),
                -2*(c*delta_2 + d*delta_1),
                a,
                c,
                b+d
            ], dim=-1)

        else :
            a = self.attention_spreads**2
            # bs,3
            pos_proj = torch.stack([ a*delta_1, a * delta_2 ,-1/2*torch.ones_like(delta_1)], dim=-1)

        # bs m n
        pos_score = torch.einsum('mnd,bd->bmn',self.rel_indices,pos_proj) 
        pos_score = rearrange(pos_score,'(s b) m n  -> b m n s ', s = self.gamma)
        pos_score = pos_score
        if mask is not None:
            pos_score = pos_score + mask.unsqueeze(-1)


        posmap = self.sft(pos_score)
        return posmap

    def forward(self,x,mask=None):
        # B, m , n
        assert len(x.size()) == 4
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma)
        posmap = self.forward_pos(mask=mask)
        x = torch.einsum('wmns,bwnvs->bwmvs',posmap,x) +  self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
        return x

class LearnedPosMap(nn.Module):
    def __init__(self, win_size,gamma=1,**kwargs):
        super().__init__()
        self.gamma = gamma
        self.win_size =win_size
        self.seq_len = win_size*win_size
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1 ,self.seq_len,1))
        self.rel_locl_init(self.win_size,register_name='window')
        self.init_bias_table()

    def rel_locl_init(self,win_size, register_name='window'):

        h= win_size
        w= win_size
        self.register_parameter(f'{register_name}_relative_position_bias_table' ,nn.Parameter(
            torch.zeros((2 * h - 1) * (2 * w - 1), self.gamma)))


        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += h - 1 
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer(f"{register_name}_relative_position_index", relative_position_index)
 
    def init_bias_table(self):
        for k,v in self.named_modules():
            if 'relative_position_bias_table' in k:
                trunc_normal_(v.weight, std=.02) 

    def forward_pos(self):
        posmap = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
            self.seq_len, self.seq_len, -1) 
        posmap =  posmap.permute(2, 0, 1).contiguous()
        return posmap

    def forward(self,x,weight=None,mask=None):
        posmap = self.forward_pos()
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma)
        win_weight = rearrange(posmap,'(s b) m n  -> b m n s', s = self.gamma)
        if weight is not None:
            # b x N x N 
            # lamb = self.lamb.view(1,1,1,1)
            win_weight = win_weight + weight.unsqueeze(-1)
        else:
            win_weight = win_weight
        x = torch.einsum('wmns,bwnvs->bwmvs',win_weight,x) + self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
 
        return x

class SGU(nn.Module):

    def __init__(self, dim, win_size,chunks=2, norm_layer=nn.LayerNorm, layer_idx=None,**kwargs):
        super().__init__()
        self.chunks = chunks
        self.gate_dim = dim // chunks
        seq_len = win_size*win_size
        self.pad_dim = dim % chunks
        
        bias = layer_idx <=25 if layer_idx else True
        self.proj_list = nn.Sequential(*[nn.Linear(seq_len, seq_len,bias=bias) for i in range(chunks-1)])
        self.norm_list = nn.Sequential(*[norm_layer(self.gate_dim)for i in range(chunks-1)])

    def init_weights(self):

        for proj in self.proj_list:
            nn.init.normal_(proj.weight, std=1e-6)
            if proj.bias is not None:
                nn.init.ones_(proj.bias)

    def forward(self, x,mask = None):

        if self.pad_dim:
            x = x[:,:,:-self.pad_dim]
        x_chunks = x.chunk(self.chunks, dim=-1)
        u = x_chunks[0]
        for i in range(self.chunks-1):
            v = x_chunks[i+1]
            u = self.norm_list[i](u)
            u = self.proj_list[i](u.transpose(-1, -2))
            u = u.transpose(-1, -2)
            u = u * v
        return u

class PoSGU(nn.Module):
    
    def __init__(self, dim, win_size,chunks=2, norm_layer=nn.LayerNorm,quadratic=True, gamma=16,pos_only=True,**kwargs):
        super().__init__()
        self.chunks = chunks
        self.gate_dim = dim // chunks
        self.seq_len=win_size*win_size
        self.quadratic = quadratic
        self.pos_only=pos_only

        if self.quadratic:
            self.pos=QuaMap(win_size,gamma=gamma,norm_layer=norm_layer,**kwargs)
        else:
            self.pos=LearnedPosMap(win_size,gamma=gamma,norm_layer=norm_layer,**kwargs)

        if not self.pos_only:
            self.token_proj_n_weight = nn.Parameter(torch.zeros(1, self.seq_len, self.seq_len))
            trunc_normal_(self.token_proj_n_weight,std=1e-6)

    def forward(self, x,mask = None):
        # B W N C
        if self.chunks==1:
            u = x
            v = x
        else:
            x_chunks = x.chunk(2, dim=-1)
            u = x_chunks[0]
            v = x_chunks[1]
        if not self.pos_only and not self.quadratic:
            u = self.pos(u,self.token_proj_n_weight)
        else:
            u =self.pos(u,mask=mask)
        
        u = u * v

        return u



class PosMLPLayer(nn.Module):
    def __init__(self, dim, win_size,shift = False, gate_unit=PoSGU, num_blocks=1,
    chunks = 2, mlp_ratio=4., drop=0., drop_path_rates=0., act_layer=nn.GELU,norm_layer=nn.LayerNorm,layer_idx=0,gamma=8,**kwargs):
        super().__init__()  
        chunks = chunks  
        shift_size = 0
        self.shift = shift
        if shift and num_blocks != 1 and layer_idx %2 ==1:
            shift_size = win_size//2
            _logger.info(f"shifted layer {layer_idx}")
        self.shift_size = shift_size
        self.dim =dim
        self.norm = norm_layer(dim)
        self.hidden_dim = int(mlp_ratio * dim)
        self.split_dim = self.hidden_dim // chunks
        self.window_size = win_size
        self.proj_c_e = nn.Linear(self.dim,self.hidden_dim)
        self.proj_c_s = nn.Linear(self.split_dim ,self.dim)
        self.gate_unit = gate_unit(dim=self.hidden_dim, win_size = win_size,chunks=chunks, norm_layer=norm_layer,layer_idx=layer_idx,gamma=gamma,**kwargs)

        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0. else nn.Identity()
        self.act = act_layer()

    def forward_gate(self,x,mask= None):
        residual = x
        x = self.act(self.proj_c_e(self.norm(x)))
        # 1 c b n 
        x = self.gate_unit(x,mask=mask)
        x = self.drop(self.proj_c_s(x))
        return self.drop_path(x) + residual
        
    def forward_noshift(self,x):
                # Input : x (1,b,n,c)
        return self.forward_gate(x)

    def forward_shift(self,x):
        # @swin transformer
        # @http://arxiv.org/abs/2103.14030

        _, H, W, C = x.size()
        # H = int(math.sqrt(B*N))
        # W = H
        # x = x.view(-1,H,W,C)
        if self.shift_size > 0:
            # print("shift,here")
            shifted_x= torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # 1,nW, window_size*window_size, 1
            mask_windows = self.window_partition(img_mask)
            attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-200.0)).masked_fill(
                                                  attn_mask == 0, float(0.0)).squeeze()
        else:
            shifted_x = x
            attn_mask =None
        # _,NM,N,C
        shifted_x = window_partition(shifted_x,self.window_size,self.window_size)   
        out = self.forward_gate(shifted_x,mask = attn_mask)
        shifted_x = window_reverse(out, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self,x):
        if self.shift:
            x = self.forward_shift(x)
        else:
            x = self.forward_noshift(x)
        return x

def window_reverse(x, H, W):

    B,_,N,C=x.size()
    Wh= int(math.sqrt(N))
    Ww = Wh
    x= x.view(B, -1, Wh,  Ww, C)   
    x = x.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition( x,Wh,Ww):

    B, H, W, C = x.shape
    x = x.view(B, H //Wh, Wh, W // Ww,Ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(B,-1, Wh*Ww, C)
    return windows

class PosMLPLevel(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """
    def __init__(
            self,gate_unit, win_size, depth, embed_dim, gate_layer=PosMLPLayer,pool_layer=ConvPatchMerging,prev_embed_dim=None,
            mlp_ratio=4., drop_rate=0., drop_path_rates=[],downsample_norm="LN",
            norm_layer=None, act_layer=None, layer_idx = 0, shift = False,**kwargs):

        super().__init__()
        self.win_size=win_size
        self.shift = shift
        if prev_embed_dim is not None:
            self.pool = pool_layer(prev_embed_dim, embed_dim,downsample_norm=downsample_norm, layer_idx=layer_idx,**kwargs)
        else:
            self.pool = nn.Identity()
        self.idx = layer_idx
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.encoder = nn.Sequential(*[
            gate_layer(dim= embed_dim,win_size=win_size,gate_unit=gate_unit,mlp_ratio=mlp_ratio,
            drop = drop_rate,drop_path_rates=drop_path_rates[i],layer_idx = layer_idx+i,shift =shift ,
            norm_layer=norm_layer,act_layer=act_layer,**kwargs)
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (1, C, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1) # (1,H,W,C)
        if self.shift :
             x = self.encoder(x)  # (1, H, W, C)
        else:
            _,H,W,_ = x.size()
            x= window_partition(x, self.win_size,self.win_size)  # (1, T, N, C)
            x = self.encoder(x)  # (1, T, N, C)
            x = window_reverse(x, H, W)  # (1, H, W, C)
        return x.permute(0, 3, 1, 2)  # (1, C, H, W)

class PosMLP(nn.Module):

    def __init__(self, 
                img_size=(224,224), 
                in_chans=3,
                patch_size=4, 
                num_levels=3, 
                embed_dims=(96,192,384,768),
                win_size =(14,14,14,7),
                gamma = (8,16,32,64),
                depths=(2,2,18,2), 
                mlp_ratio=(4,4,4,2),
                num_classes=1000, 
                gate_unit=PoSGU,
                gate_layer=PosMLPLayer,
                pool_layer=ConvPatchMerging,
                stem = ConvPatchEmbed,
                drop_rate=0.,
                drop_path_rate=0.1, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                act_layer=nn.GELU,
                global_pool='avg',
                stem_norm="BN",
                downsample_norm="LN",
                ape = False,
                pretrained = None,
                **kwargs):
        super().__init__()


    
        embed_dims = tuple(list(embed_dims))
        depths = tuple(depths)
        win_size =tuple(win_size)
        gamma =tuple(gamma)
        mlp_ratio=tuple(mlp_ratio)
        _logger.info(f"Using embed_dims{embed_dims}")

        assert len(mlp_ratio)==len(depths)==len(embed_dims)==num_levels
        self.num_classes = num_classes
        self.feature_info = []
        self.drop_rate = drop_rate
        self.num_levels = num_levels

        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 3136, 96))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.patch_size = patch_size # it is freezed

        self.win_size=win_size
        for i in self.win_size:
            _logger.info(f"each_stage with win_size:{i}") 

        self.patch_embed = stem(
            img_size=img_size, in_chans=in_chans,stem_norm=stem_norm, out_chan=embed_dims[0],act_layer=act_layer, flatten=False,**kwargs)

        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        layer_idx = 0
        for i in range(len(depths)):
            dim = embed_dims[i]
            levels.append(PosMLPLevel(
              gate_unit,self.win_size[i],depths[i], dim,pool_layer=pool_layer, downsample_norm=downsample_norm,gate_layer=gate_layer,prev_embed_dim=prev_dim,gamma=gamma[i],
                mlp_ratio=mlp_ratio[i],  drop_rate=drop_rate, drop_path_rates = dp_rates[i], norm_layer=norm_layer, act_layer=act_layer,layer_idx=layer_idx,**kwargs))
            layer_idx += depths[i]
            prev_dim = dim
        self.levels = nn.Sequential(*levels)

        # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        # Classifier
        self.global_pool, self.head = create_classifier(embed_dims[-1], self.num_classes, pool_type=global_pool)
        self.apply(self._init_weights)

        if pretrained:
            state_dict=self.filter_pretrain(pretrained)
            self.load_state_dict(state_dict,strict=True)
            _logger.info(f"successfully loaded pretrain from{pretrained}")

    def filter_pretrain(self,path):
        ckpt = torch.load(path,map_location='cpu')['state_dict_ema']
        out_ckpt={}
        # model_state =dict(self.named_parameters())
        for key,value in ckpt.items():
            out_ckpt[key] = value
        return out_ckpt


    def forward_features(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed.reshape(1,-1,56,56)
        x = self.levels(x)
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

    def _init_weights(self,module: nn.Module, name: str = '', head_bias: float = 0.):

        if isinstance(module,nn.BatchNorm2d):
                _logger.info("BN here")
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




@register_model
def Create_PosMLP( **kwargs):
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(**kwargs)
    model = PosMLP(**model_kwargs)
    return model

@register_model
def PosMLP_3T14_224(**kwargs):
    model_kwargs = dict(embed_dims=(96, 192, 384),  depths=(2, 2, 20),mlp_ratio=(4,4,4),num_levels=3,num_blocks=(64,4,1),gamma=(8,16,32),**kwargs)
    model = PosMLP( **model_kwargs)
    return model

@register_model
def PosMLP_T14_224(**kwargs):
    model_kwargs = dict(embed_dims=(96, 192, 384,768),  depths=(2, 2, 18,2),mlp_ratio=(4,4,4,2),num_levels=4,chunks=2,gamma=(8,16,32,64),**kwargs)
    model = PosMLP(**model_kwargs)
    return model

@register_model
def PosMLP_S14_224(**kwargs):
    model_kwargs = dict(embed_dims=(128,256,512,1024),  depths=(2, 2, 18,2),mlp_ratio=(4,4,4,2),num_levels=4,chunks=2,gamma=(8,16,32,64),**kwargs)
    model = PosMLP(**model_kwargs)
    return model


@register_model
def PosMLP_T14_384(**kwargs):
    model_kwargs = dict(img_size=(384,384),embed_dims=(96, 192, 384,768),  depths=(2, 2, 18,2),mlp_ratio=(4,4,4,2),num_levels=4,gamma=(8,16,32,64),**kwargs)
    model = PosMLP(**model_kwargs)
    return model

if __name__ == '__main__':
    import timm
    net = PosMLP_T14_384()
    input = torch.randn([1,3,224,224])
    net(input)
    print(timm.models.is_model("PosMLP_T14_224"))

