

import torch
from torch import nn as nn
from timm.models.layers import to_2tuple
from functools import partial

class PatchMerging(nn.Module):

    def __init__(self, in_chan ,out_chan, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs):
        super().__init__()
        self.out_chan = out_chan
        self.norm = norm_layer(in_chan)
        self.reduction = nn.Conv2d(in_chan,  out_chan, 1, 1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
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

class ConvPatchMerging(nn.Module):

    def __init__(self, in_chan, out_chan, downsample_norm="LN",depth_conv = True,**kwargs):
        super().__init__()
        self.downsample_norm = downsample_norm
        norm_layer = nn.BatchNorm2d if downsample_norm=='BN' else nn.LayerNorm
        groups = 1
        if depth_conv:
            groups = in_chan
        self.conv = nn.Conv2d(in_chan,out_chan,kernel_size=3,stride=2,groups=groups,padding=1,bias=True)
        self.norm = norm_layer(out_chan,eps=1e-6)

    def forward(self, x):
        """
        (B, C, H, W) -> (B, 2C, H/2, W/2)
        """
        x = self.conv(x)
        x = self.norm(x) if self.downsample_norm=="BN" else self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x  
        
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chan=768, norm_layer=nn.LayerNorm, flatten=True,**kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, out_chan, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(out_chan) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class ConvPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
        patch size is freezed to be 4
    """
    def __init__(self,img_size=224, patch_size=None, in_chan=3, out_chan=96,stem_norm="LN",act_layer=nn.GELU,**kwargs):
        super().__init__()
        norm_layer = nn.BatchNorm2d if stem_norm=='BN' else nn.LayerNorm
        self.in_chans=[in_chan,out_chan//2,out_chan]
        self.out_chans = [out_chan//2,out_chan,out_chan]
        self.strides=[2,2,1]
        self.kernels=[3,3,1]
        self.pads = [1,1,0]
        self.projs = nn.Sequential(*[nn.Conv2d(in_chan, out_chan,kernel_size=kernel, stride=stride,padding=pad) for 
        in_chan, out_chan,kernel,stride,pad in zip(self.in_chans,self.out_chans,self.kernels,self.strides,self.pads)]) 
        self.stem_norm = stem_norm
        self.norms = nn.Sequential(*[norm_layer(out_chan,eps=1e-6) for out_chan in self.out_chans])
        self.act = act_layer()

    def forward(self, x):
        for i ,(proj, norm) in enumerate(zip(self.projs,self.norms)):
            x = proj(x)
            x = norm(x) if self.stem_norm == "BN" else norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i == len(self.kernels)-1:
                break
            x = self.act(x)
        return x

