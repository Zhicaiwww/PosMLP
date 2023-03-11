import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from timm.models.layers import  trunc_normal_

class QuaMap(nn.Module):
    """_summary_
    An quadratic relativie positional encoding based fully connected layer under 2D (-1, H, W, C) input case
    :param win_size: the width and height of a 2D patch.
    :param gamma: the channel groupwise number.
    :param att_std: initialized noise adding to the indentity covariate matrix.
    :param generalized: set True if not only optimize diagnal terms in covariate matrix.
    :param absolute_bias: set True if add bias term after gating.
    :param symmetric: set True if set the covariate matrix in semi-definte form. 
    """
    
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
        x = torch.einsum('wmns,bwnvs->bwmvs',posmap,x)
        if self.absolute_bias:
            x = x +  self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
        return x

class LearnedPosMap(nn.Module):

    """_summary_
    An learnable relativie positional encoding based fully connected layer under 2D (-1, H, W, C) input case
    :param win_size: the height and width length
    :param gamma: the channel groupwise number.
    """ 
    
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
