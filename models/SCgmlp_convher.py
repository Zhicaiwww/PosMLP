



import os
import torch
import torch.nn as nn
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, drop, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model




def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'SCgmlp_ti8': _cfg(),
    'SCgmlp_s8': _cfg(),

}


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

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class SCGateUnit(nn.Module):
    
    
    def __init__(self, dim, seq_len,chunks=2, gamma= 8,norm_layer=nn.LayerNorm):
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

# def atten3D(x,lambda):
#     n = x.shape[2]*x.shape[3]-1
#     d = (x-x.mean(dim=[2,3])).pow(2)
#     v = d.sum(dim=[2,3])/n
#     E_inv=d/(4*(v+lambda))+0.5
#     return x * nn.sigmoid(E_inv)

class Downsample(nn.Module):
    """ Downsample transition stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


class GatedMlpLayer(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features,hidden_features=None, out_features=None,act_layer=nn.GELU,
                 gate_layer=SCGateUnit, chunks=2,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            self.gate = gate_layer(hidden_features,chunks=chunks,norm_layer=norm_layer)
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
            self, dim, seq_len, mlp_ratio=6, chunks=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), gamma=8,act_layer=nn.GELU, drop=0., drop_path=0.,mlp_fn=GatedMlpLayer):
        super().__init__()

        gate_dim = int(mlp_ratio*dim)
        # gate_dim = [1024,256]
        self.norm = norm_layer(dim)
        gate_unit = partial(SCGateUnit, seq_len=seq_len,gamma = gamma)
        # tgu = partial(TimeGatingUnit, segments=segments)
        self.mlp_channels = mlp_fn(dim, hidden_features=gate_dim, chunks=chunks, act_layer=act_layer, gate_layer=gate_unit,norm_layer=norm_layer,drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x



def basic_blocks(dim, index, seq_len, layers, mlp_ratio=3., chunks=2, drop=0.,
                 drop_path_rate=0., gamma=8,mlp_fn=GatedMlpLayer, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append( SpatialGatingBlock(dim, seq_len, mlp_ratio=mlp_ratio,chunks=chunks, 
                      drop=drop, drop_path=block_dpr,mlp_fn=mlp_fn,gamma=gamma))
    blocks = nn.Sequential(*blocks)

    return blocks

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,  dim, norm_layer=nn.GroupNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, bias=False)
        self.norm = norm_layer(1,4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B,hw,C=x.shape
        H=W=int(hw**0.5)
        #assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.permute(0,2,1).view(B, C, H, W)

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2 

        x = self.norm(x)
        x = self.reduction(x).flatten(2).permute(0,2,1)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops



class ConvolutionalEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=None,kernel=None,stride = None, padding=None, norm_layer=partial(nn.BatchNorm2d, eps=1e-6), flatten=True):
        super().__init__()
        assert len(embed_dim) == len(kernel) and len(embed_dim)==len(padding) and len(embed_dim)==len(stride)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.chans=embed_dim
        self.chans.insert(0,in_chans)
        self.in_chan=in_chans
        self.out_chan = self.chans[-1]

        self.strides =stride
        self.kernels =kernel
        self.pads = padding
        self.projs = nn.Sequential(*[nn.Conv2d(int(in_chan), int(out_chan), kernel_size=kernel, stride=stride,padding=pad) for 
        in_chan, out_chan,kernel,stride,pad in zip(self.chans[:-1],self.chans[1:],self.kernels,self.strides,self.pads)]) 
        self.norms = nn.Sequential(*[norm_layer(out_chan) for out_chan in self.chans[1:]])
        self.feature_norm=nn.GroupNorm(1, self.out_chan//8,eps=1e-6)
        self.act = nn.ReLU()

    def forward_break_feature(self,x):
        # split (B,H*W,C) as (B,C/4,2*H,2*W)
        x = x.transpose(1,2)
        b,c,hw=x.shape
        h=w=int(hw**0.5)
        out = torch.zeros([b,c//4,2*h,2*w]).type_as(x)
        chunks = x.view(b,c,h,w).chunk(4,dim=1)
        out[:,:,0::2,0::2]=chunks[0]
        out[:,:,1::2,1::2]=chunks[1]
        out[:,:,0::2,1::2]=chunks[2]
        out[:,:,1::2,0::2]=chunks[3]
        out = self.feature_norm(out)
        return out

    def forward(self, x):
        # print("in",x.size())
        if len(x.shape)==3:
            x = self.forward_break_feature(x)

        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        for i ,(proj, norm) in enumerate(zip(self.projs,self.norms)):
            x = proj(x)
            x = norm(x)

            if i == len(self.kernels)-1:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                break
            x = self.act(x)
        # print("out",x.size())
        return x

class SCGNet(nn.Module):

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, 
        num_classes=1000, gamma = 8, embed_dims=None, kernels=None,
        paddings=None,strides=None,mlp_ratios=None, chunks=3,
        drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, 
        mlp_fn=GatedMlpLayer, stem_norm=True,fork_feat=False):

        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        stem = ConvolutionalEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dims[0], kernel=kernels[0],padding=paddings[0],
                stride=strides[0],norm_layer=partial(nn.BatchNorm2d, eps=1e-6) if stem_norm else norm_layer)

        network = []
        network.append(stem)
        dim = embed_dims[0][-1]
        seq_len = stem.num_patches
        for i in range(len(layers)):
            if i > 0:
                dim = embed_dims[i-1][-1]
                seq_len = seq_len//4
                PM = PatchMerging(dim=dim)
            # print("conv",seq_len,in_chans,patch_size,img_size)
                network.append(PM)
            stage = basic_blocks(embed_dims[i][-1], index=i,seq_len=seq_len, layers=layers, mlp_ratio=mlp_ratios[i],
                chunks=chunks[i], drop=drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,gamma=gamma, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1][-1])
            self.head = nn.Linear(embed_dims[-1][-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def forward_embeddings(self, x):
    #     x = self.stem(x)
    #     # B,C,H,W-> B,H,W,C
    #     return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        if self.fork_feat:
            return outs

        return x

    def forward(self, x):
        # x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


@register_model
def SCgmlp_ti8(pretrained=False, **kwargs):
    layers = [2, 14, 2]
    mlp_ratios = [6, 3, 3]
    chunks = [3, 3, 3]
    embed_dims = [  [32,64,128],
                    [64,128,256],
                    [128,256,512]]
    kernels=[[3,3,3],
            [2,2,1],
            [2,2,1]]

    strides=[[2,2,2],
            [2,2,1],
            [2,2,1]]
    paddings =  [[1,1,1],
                [0,0,0],
                [0,0,0]]
    gamma = 4
    model = SCGNet(layers, patch_size=8, chunks=chunks,embed_dims=embed_dims,kernels=kernels,paddings=paddings,strides=strides,
                     mlp_ratios=mlp_ratios,gamma=gamma, mlp_fn=GatedMlpLayer, **kwargs)
    model.default_cfg = default_cfgs['SCgmlp_ti8']
    return model



@register_model
def SCgmlp_s8(pretrained=False, **kwargs):
    layers = [4, 18, 2]
    mlp_ratios = [4, 4, 4]
    chunks = [3, 3, 3]
    embed_dims = [  [32,64,128,256],
                    [128,256,512],
                    [256,512,1024]]
    kernels=[[3,3,3,3],
            [3,3,3],
            [3,3,3]]

    strides=[[2,2,2,1],
            [2,2,1],
            [2,2,1]]
    paddings =  [[1,1,1,1],
                [1,1,1],
                [1,1,1]]
    model = SCGNet(layers, patch_size=8, chunks=chunks,embed_dims=embed_dims,kernels=kernels,paddings=paddings,strides=strides,
                     mlp_ratios=mlp_ratios, mlp_fn=GatedMlpLayer, **kwargs)
    model.default_cfg = default_cfgs['SCgmlp_s8']
    return model



