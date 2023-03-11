import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from timm.models.layers import  trunc_normal_


class LearnedPosMap1D(nn.Module):

    """_summary_
    An learnable relativie positional encoding based fully connected layer under 1D (-1, N, C) input case
    :param win_size: the 1D sequence length.
    :param gamma: the channel groupwise number.
    """
    
    def __init__(self, win_size,gamma=1,**kwargs):
        super().__init__()
        self.gamma = gamma
        self.win_size =win_size
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1 ,self.win_size,1))
        self.rel_locl_init(self.win_size,register_name='window')
        self.init_bias_table()

    def rel_locl_init(self,win_size, register_name='window'):

        h= win_size
        self.register_parameter(f'{register_name}_relative_position_bias_table' ,nn.Parameter(
            torch.zeros((2 * h - 1) , self.gamma)))


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



class QuaMap1D(nn.Module):

    """_summary_
    An quadratic relativie positional encoding based fully connected layer under 1D (-1, N, C) input case
    :param win_size: the 1D sequence length.
    :param gamma: the channel groupwise number.
    :param att_std: Initialized noise adding to the indentity covariate matrix.
    :param generalized: set False if only optimizing diagnal terms in covariate matrix.
    :param absolute_bias: set True if add bias term after gating.
    
    """
    def __init__(self,win_size,gamma=4,use_softmax=True,att_std = 1e-2,generalized=True,absolute_bias=True,layer_idx=1,**kwargs):
        super().__init__()
        self.att_std = att_std 
        self.win_size = win_size
        self.layer_idx = layer_idx 
        self.gamma = gamma 
        self.generalized = generalized
        self.absolute_bias = absolute_bias
        self.sft=nn.Softmax(-2) if use_softmax else nn.Identity()
        self.get_rel_indices(self.win_size,gamma=self.gamma)
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1, win_size,1))

    def get_rel_indices(self, win_size,gamma=1):
        
        w = win_size
        num_patches = w 

        indx =  torch.arange(w).view(1,-1) - torch.arange(w).view(-1,1) 
        indxx = indx**2 
        if self.generalized:
            rel_indices   = torch.zeros((num_patches, num_patches,2))
            rel_indices[:,:,1] = indxx.unsqueeze(0)
            rel_indices[:,:,0] = indx.unsqueeze(0)
            self.attention_centers = nn.Parameter(torch.zeros((gamma, 1)).normal_(0.0,self.att_std))
            attention_spreads = torch.ones(1).unsqueeze(0).repeat(gamma, 1)
            attention_spreads += torch.zeros_like(attention_spreads).normal_(0,self.att_std)
            self.attention_spreads = nn.Parameter(attention_spreads)
        else:
            rel_indices   = torch.zeros((num_patches, num_patches,1))
            rel_indices[:,:,0] = indxx.unsqueeze(0)
            self.attention_centers = nn.Parameter( torch.zeros(gamma, 1).normal_(0.0,self.att_std))
            attention_spreads = 1 + torch.zeros(gamma).normal_(0, self.att_std).unsqueeze(-1)
            self.attention_spreads = nn.Parameter(attention_spreads)

        self.register_buffer("rel_indices", rel_indices)

    def forward_pos(self,mask=None):
        # gamma,1

        delta = self.attention_centers
        # gamma,1
        inv_covariance = self.attention_spreads

        if self.generalized:
            # gamma,2   
            pos_proj = -1/2 * torch.cat([- 2 * delta * inv_covariance, inv_covariance], dim = -1)
        else:
            # gamma,1
            pos_proj = -1/2 * inv_covariance
 
        # gamma w w
        os_score = torch.einsum('mnd,sd->smn',self.rel_indices,pos_proj) 
        # w w gamma 
        pos_score = rearrange(pos_score,'(b s) m n  ->b m n s ', s = self.gamma)
        # 1 w sft(w) gamma 
        posmap = self.sft(pos_score)
        
        return posmap

    def forward(self,x,mask=None):
        # b, w, n, c
        assert len(x.size()) == 4
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma)
        posmap = self.forward_pos()

        # ipdb.set_trace()
        x = torch.einsum('wmns,bwnvs->bwmvs',posmap,x) 
        if self.absolute_bias:
            x = x +  self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
        return x


if __name__ == '__main__':
    import timm
    x = torch.rand([1,128,8,64])
    quamap = QuaMap1D(8, gamma=4, generalized= True)
    out = quamap(x)
    print(out.size())
