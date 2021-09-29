

""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import nn as nn
from timm.models.layers import to_2tuple
from functools import partial

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,**kwargs):
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


class ConvolutionalEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=partial(nn.BatchNorm2d, eps=1e-6), flatten=True,**kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans=[3,32,64,128,256]
        self.out_chans = [32,64,128,256,256]
        self.strides=[2,2,2,2,1]
        self.kernels=[3,3,3,3,1]
        self.pads = [1,1,1,1,0]
        self.projs = nn.Sequential(*[nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=stride,padding=pad) for 
        in_chan, out_chan,kernel,stride,pad in zip(self.in_chans,self.out_chans,self.kernels,self.strides,self.pads)]) 
        self.norms = nn.Sequential(*[norm_layer(out_chan) for out_chan in self.out_chans])
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        for i ,(proj, norm) in enumerate(zip(self.projs,self.norms)):
            x = proj(x)
            x = norm(x)

            if i == len(self.kernels)-1:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                break
            x = self.act(x)
        return x

class Nest_ConvolutionalEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, flatten=True,**kwargs):
        super().__init__()

        norm_layer=partial(nn.BatchNorm2d, eps=1e-6)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans=[3,48,96]
        self.out_chans = [48,96,96]
        self.strides=[2,2,1]
        self.kernels=[3,3,1]
        self.pads = [1,1,0]
        self.projs = nn.Sequential(*[nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=stride,padding=pad) for 
        in_chan, out_chan,kernel,stride,pad in zip(self.in_chans,self.out_chans,self.kernels,self.strides,self.pads)]) 
        self.norms = nn.Sequential(*[norm_layer(out_chan) for out_chan in self.out_chans])
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        for i ,(proj, norm) in enumerate(zip(self.projs,self.norms)):
            x = proj(x)
            x = norm(x)

            if i == len(self.kernels)-1:
                x = x.transpose(2, 3)  # BCHW -> BNC
                break
            x = self.act(x)
        return x