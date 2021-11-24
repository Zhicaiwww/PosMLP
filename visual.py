import csv
import matplotlib.pyplot as plt 
import os
import numpy as np
import torch.nn as nn 
import torch
color_list = ['red','blue','green','orange','salmon']

def getdata(path):
    acc=[]
    loss_train=[]
    ema_acc=[]
    with open(path,'r') as f:
        cr = f.readlines()
        with_ema = len(cr) > 5
        for idx, row in enumerate(cr):
            if idx==0:
                continue
            row = row.strip().split(',')
            acc.append(float((row[3])))
            loss_train.append(float(row[1]))
            if with_ema:
                ema_acc.append(float((row[6])))
    return acc,ema_acc,loss_train

def draw_acc(summary_list,name_list):

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    for i,(path, name) in enumerate(zip(summary_list,name_list)):
        acc,ema_acc,loss=getdata(path)
        ax1.plot(acc,color = color_list[i], label=f'{name}')
        ax2.plot(loss,color = color_list[i],label=f'{name}')    

    ax1.set_ylabel('ACC1')
    ax1.set_xlabel('epochs')
    ax1.legend()
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('epochs')
    ax2.legend()
    plt.show()
    fig.savefig('acc.jpg', facecolor='grey', edgecolor='red')

def show_weight(path,indexes=[0,-1],is_all = True,save_path=None):
    ckpt = torch.load(path,map_location='cpu')
    to_kw=[]
    to_vw=[]
    to_kb=[]
    to_vb=[]
    to_kp=[]
    to_vp=[]
    to_kidx=[]
    to_vidx=[]


    for k,v in ckpt['state_dict_ema'].items():
        if "gate_unit.proj" in k and 'weight' in k:
            to_kw.append(k)
            to_vw.append(v.cpu())
        if "token_proj_n_bias" in k :
            to_kb.append(k)
            to_vb.append(v.squeeze(-1))
        if 'gate_unit.relative_position_bias_table' in k:
            to_kp.append(k)
            to_vp.append(v.squeeze(0))
        if 'relative_position_index' in k:
            to_kidx.append(k)
            to_vidx.append(v.cpu())
    print(f"number of figures is {len(to_kp)}")
    if len(to_kp) != 0:
        import math
        bias_map =[]
        for vp in to_vp:
            h,w = to_vidx[0].size()
            bias_map.append(vp[to_vidx[0].view(-1)][:,0].view(h,w))
    idxs = range(len(to_kp)) if is_all else indexes
    for idx in idxs:
        fig = plt.figure(figsize=(16, 16),tight_layout=True)
        ax = fig.add_subplot(2,2,1)
        ax.imshow(to_vw[idx])
        ax.axis('off')
        ax = fig.add_subplot(2,2,2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(bias_map[idx])
        ax.axis('off')
        ax = fig.add_subplot(2,2,3)
        ax.imshow(bias_map[idx]+to_vw[idx])
        ax.axis('off')
        ax = fig.add_subplot(2,2,4)
        ax.plot(to_vw[idx][50],color = color_list[0],label='TokenFC')
        ax.plot(bias_map[idx][50],color = color_list[1],label='LRPE')
        ax.plot(to_vw[idx][50]+bias_map[idx][50],color = color_list[2],label='TokenFC+LRPE')
        ax.set_ylabel('score')
        ax.set_xlabel('N')
        ax.legend(fontsize=16,loc='lower right')
        # ax.axis('off')
        if save_path:
            if os.path.exists(f'./figure/{save_path}'):
                pass
            else :
                os.makedirs(f'./figure/{save_path}')

            fig.savefig(f'./figure/{save_path}/save_img_{idx}.pdf',  edgecolor='red')
            plt.close()
        else:
            plt.show()


def show_bias(path,indexes=[0,-1],is_all = True,save_path=None):
    ckpt = torch.load(path,map_location='cpu')
    to_kidx=[]
    to_vidx=[]
    import math

    for k,v in ckpt['state_dict_ema'].items():
        if 'token_proj_n_bias' in k:
            to_kidx.append(k)
            to_vidx.append(v.cpu())
    print(f"number of figures is {len(to_vidx)}")
    idxs = range(len(to_vidx)) if is_all else indexes
    fig = plt.figure(figsize=(12,8),tight_layout=True)
    for idx in idxs:
        ax = fig.add_subplot(4,6,idx+1)
        bias = to_vidx[idx].squeeze()
        size = bias.size(0)
        h = int(math.sqrt(size))
        bias= bias.view(h,h)
        ax.imshow(bias)
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.legend(fontsize=16,loc='lower right')
        ax.axis('off')          
    fig.tight_layout()#调整整体空白
    # fig.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace=-0.6, hspace=-0.6)#调整子图间距
    save_fig(save_path,fig,'all')


def show_key(path):
    ckpt = torch.load(path,map_location='cpu')
    for i,v in ckpt.items():
        print(f"{i}")
def to_qua_att_map(rel_indices,attention_centers,attention_spreads,levels=[2,2,18,2],idx=0,for_weight=True,pixel_idx=100):

    mu_1, mu_2 = attention_centers[:, 0], attention_centers[:, 1]
    inv_covariance = torch.einsum('hij,hkj->hik', [attention_spreads, attention_spreads])
    # inv_covariance = attention_spreads
    a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]
    # b,5
    pos_proj =-1/2 * torch.stack([
        -2*(a*mu_1 + b*mu_2),
        -2*(c*mu_2 + b*mu_1),
        a,
        c,
        2 * b
    ], dim=-1)
    pos_score = nn.Softmax(-1)(torch.einsum('mnd,bd->bmn',rel_indices,pos_proj))
    if  idx < levels[0]+levels[1]+levels[2]:
        pass
    else:
        pixel_idx = 24
    import math
    bs,m,n=pos_score.size()
    h = int(math.sqrt(n))
    # if idx < levels[0]:
    #     blocks = 16
    # elif idx < levels[0]+levels[1]:
    #     blocks= 4
    # else:
    blocks= 1
    s = bs//blocks
    if for_weight:
        return pos_score.view(blocks,s,m,n)
    else:
        pos_score=pos_score.view(blocks,s,m,h,h)[:,:,pixel_idx] 
    # bs m n 
    return pos_score
    
def show_qua_weight(path,indexes=[0,-1], select=None,is_all = True,for_weight=False,levels=[2,2,18,2],save_path=None):
    ckpt = torch.load(path,map_location='cpu')
    to_kw=[]
    to_vw=[]
    to_kb=[]
    to_vb=[]
    to_kri=[]
    to_vri=[]
    to_krct=[]
    to_vrct=[]
    to_krcs=[]
    to_vrcs=[]


    for k,v in ckpt['state_dict'].items():
        if "token_proj_n_weight" in k:
            to_kw.append(k)
            to_vw.append(v.squeeze())
        if "token_proj_n_bias" in k :
            to_kb.append(k)
            to_vb.append(v.squeeze())
        if 'rel_indices' in k:
            to_kri.append(k)
            to_vri.append(v.cpu())
        if 'attention_centers' in k:
            to_krct.append(k)
            to_vrct.append(v.cpu())
        if 'attention_spreads' in k:
            to_krcs.append(k)
            to_vrcs.append(v.cpu())
    print(f"number of figures is {len(to_kb)}")
    if len(to_kri) != 0:
        import math
        bias_map =[]
        for idx,(vri,vrct,vrcs) in enumerate(zip(to_vri,to_vrct,to_vrcs)):
            bias_map.append(to_qua_att_map(vri,vrct,vrcs,levels,idx,for_weight=for_weight,))
    idxs = range(len(to_kb)) if is_all else indexes



    if select:
            fig = plt.figure(figsize=(2*select,2*len(idxs)),tight_layout=True)
            
            cnt = 1
            for cos,idx in enumerate(idxs):
                #bsmn
                layer_atts=bias_map[idx]
                for i in range(select):
                    ax = fig.add_subplot(len(idxs),select,cnt)
                    # if for_weight:
                    ax.imshow(layer_atts[0,cos+i])
                    # else:
                    #     ax.plot(layer_atts[0,i,50],color = color_list[1],label='vb')    
                    ax.set_xticks([])
                    ax.set_yticks([])

                    cnt+=1
            fig.tight_layout()#调整整体空白
            plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
            save_fig(save_path,fig,"select",type="pdf")
            return 
        
    for idx in idxs:
            # fig = plt.figure(figsize=(16,16),tight_layout=True)
            # ax = fig.add_subplot(2,2,1)
            # ax.imshow(bias_map[idx])
            # ax = fig.add_subplot(2,2,2)
            # ax.plot(to_vb[idx],color = color_list[1],label='vb')
            # ax = fig.add_subplot(2,2,3)
            # ax.imshow(bias_map[idx])
            # ax = fig.add_subplot(2,2,4)
            # ax.plot(bias_map[idx][40],color = color_list[1],label='vb')
            # ax.set_ylabel('score')
            # ax.set_xlabel('N')
            # ax.legend()

        layer_atts=bias_map[idx]
        b,s,m,n =layer_atts.size()
        fig = plt.figure(figsize=(2*s,2*b),tight_layout=True)
        
        cnt = 1
        for i in range(b):
            for j in range(s):
                ax = fig.add_subplot(b,s,cnt)
                if for_weight:
                    ax.plot(layer_atts[i,j,50],color = color_list[1],label='vb')
                else:
                    ax.imshow(layer_atts[i,j])
                ax.set_xticks([])
                ax.set_yticks([])

                cnt+=1
        fig.tight_layout()#调整整体空白
        plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
        save_fig(save_path,fig,idx)



def save_fig(save_path,fig,idx,type='pdf'):
    if os.path.exists(f'./figure/{save_path}'):
        pass
    else :
        os.makedirs(f'./figure/{save_path}')

    fig.savefig(f'./figure/{save_path}/save_img_{idx}.{type}', edgecolor='red')
    plt.close()

    

def show_para(path):
    ckpt = torch.load(path,map_location='cpu')
    for i,v in ckpt['state_dict'].items():
        print(f"{i}： {v.size()}")

def show_pos(path):
    ckpt = torch.load(path,map_location='cpu')
    a =[]
    for i,v in ckpt['state_dict'].items():
        if 'lamb' in i :
            a.append(v)
 
    print(torch.cat(a,dim=0))


# path ='/data/zhicai/ckpts/Mgmlp/train/20211122-011628-nest_gmlp_s_b4-224/checkpoint-110.pth.tar'
# # show_qua_weight(path,indexes=[1,3,8,12,16,20,23],select=6,is_all=False,levels=[2,2,18,2],for_weight= False,save_path='nest_gmlp_s_b4_sym_ATT_81.6_7(1,3,8,12,16,20,22)')
# # show_qua_weight(path)
# show_para(path)

# summary_list=['/data/zhicai/ckpts/Mgmlp/train/20211008-121343-gmlp_s16_224-224_74.0/GQPEsummary.csv',
# '/data/zhicai/ckpts/Mgmlp/train/20211008-121343-gmlp_s16_224-224_74.0/summary.csv']
# name_list = ['gmlp_s_GQPE',
# 'gmlp_s']
# draw_acc(summary_list,name_list)



# path = '/data/zhicai/ckpts/Mgmlp/train/20211005-052150-nest_gmlp_s-224/checkpoint-129.pth.tar'
# show_weight(path, save_path='gmlp_s_learpos_all_(wight_and_posBias)')
# show_qua_weight(path,save_path='gmlp_s_quaposonly_all')
# show_pos(path)
# path = '/data/zhicai/ckpts/pretrain/checkpoint-126.pth_conv.tar'
# show_para(path)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("m",default=1,type=int)
    parser.add_argument("ckpt",default=None,type=str)
    parser.add_argument("-save",default='nest_gmlp_s_PM_bias_ATT_81.6_7(1,3,8,12,16,20,22)',type=str)
    args = parser.parse_args()
    path = args.ckpt
    save_path = args.save
    if args.m == 1:
        show_para(path)
    elif args.m == 2:
        show_qua_weight(path,save_path=save_path)
    elif args.m == 3:
        show_bias(path,save_path=save_path)
