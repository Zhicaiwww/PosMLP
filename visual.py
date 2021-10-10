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
            to_vb.append(v.squeeze())
        if 'gate_unit.window_relative_position_bias_table' in k:
            to_kp.append(k)
            to_vp.append(v.squeeze(0))
        if 'relative_position_index' in k:
            to_kidx.append(k)
            to_vidx.append(v.cpu())
    print(f"number of figures is {len(to_kp)}")
    if len(to_kp) is not 0:
        import math
        bias_map =[]
        for vp in to_vp:
            h,w = to_vidx[0].size()
            bias_map.append(vp[to_vidx[0].view(-1)].view(h,w))
    idxs = range(len(to_kp)) if is_all else indexes
    for idx in idxs:
        fig = plt.figure(figsize=(16, 16),tight_layout=True)
        ax = fig.add_subplot(2,2,1)
        ax.imshow(bias_map[idx])
        ax = fig.add_subplot(2,2,2)
        ax.imshow(bias_map[idx])
        ax = fig.add_subplot(2,2,3)
        ax.imshow(bias_map[idx])
        ax = fig.add_subplot(2,2,4)
        # ax.plot(to_vw[idx][50],color = color_list[0],label='vw')
        ax.plot(bias_map[idx][50],color = color_list[1],label='vb')
        # ax.plot(to_vw[idx][50]+bias_map[idx][50],color = color_list[2],label='vw+vb')
        ax.set_ylabel('score')
        ax.set_xlabel('N')
        ax.legend()
        # ax1.axis('off')
        if save_path:
            if os.path.exists(f'./figure/{save_path}'):
                pass
            else :
                os.makedirs(f'./figure/{save_path}')

            fig.savefig(f'./figure/{save_path}/save_img_{idx}.jpg', facecolor='grey', edgecolor='red')
            plt.close()
        else:
            plt.show()

def show_para(path):
    ckpt = torch.load(path)
    for i,v in ckpt['state_dict_ema'].items():
        print(f"{i}： {v.size()}")


def show_pos(path):
    ckpt = torch.load(path,map_location='cpu')
    a = [[],[],[]]
    for i,v in ckpt['state_dict_ema'].items():
        if 'window_lamb' in i :
            a[0].append((i,v))
        elif 'block_lamb' in i :
            a[1].append((i,v))       
        elif 'split_lamb' in i :
            a[2].append((i,v))
    for i in a:
        print('\n')
        for j in i:
            print(j)
                # break

def to_qua_att_map(rel_indices,attention_centers,attention_spreads):

    mu_1, mu_2 = attention_centers[:, 0], attention_centers[:, 1]
    inv_covariance = torch.einsum('hij,hkj->hik', [attention_spreads, attention_spreads])
    a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]
    # b,5
    pos_proj =-1/2 * torch.stack([
        -2*(a*mu_1 + b*mu_2),
        -2*(c*mu_2 + b*mu_1),
        a,
        c,
        2 * b
    ], dim=-1)
    pos_score = torch.einsum('mnd,bd->bmn',rel_indices,pos_proj)
    return pos_score[0].squeeze(0)


def show_qua_weight(path,indexes=[0,-1], is_all = True,save_path=None):
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


    for k,v in ckpt['state_dict_ema'].items():
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
        for vri,vrct,vrcs in zip(to_vri,to_vrct,to_vrcs):
            bias_map.append(to_qua_att_map(vri,vrct,vrcs))
    idxs = range(len(to_kb)) if is_all else indexes
    splits = to_vri[-1].size(0)
    for idx in idxs:
        fig = plt.figure(figsize=(16,16),tight_layout=True)
        ax = fig.add_subplot(2,2,1)
        #ax.imshow(to_vw[idx])
        ax.imshow(bias_map[idx])
        ax = fig.add_subplot(2,2,2)
        # if len(to_kb) != 0:
        #     ax.plot(to_vb[idx])
        # else:
        ax.plot(bias_map[idx][50],color = color_list[1],label='vb')
        ax = fig.add_subplot(2,2,3)
        ax.imshow(nn.Softmax(-1)(bias_map[idx]))
        # ax.imshow(to_vw[idx]+bias_map[idx])
        ax = fig.add_subplot(2,2,4)
        # ax.plot(to_vw[idx][50],color = color_list[0],label='vw')
        ax.plot(nn.Softmax(-1)(bias_map[idx])[50],color = color_list[1],label='vb')
        # ax.plot(to_vw[idx][50]+bias_map[idx][50],color = color_list[2],label='vw+vb')
        ax.set_ylabel('score')
        ax.set_xlabel('N')
        ax.legend()
        # ax1.axis('off')
        if save_path:
            if os.path.exists(f'./figure/{save_path}'):
                pass
            else :
                os.makedirs(f'./figure/{save_path}')

            fig.savefig(f'./figure/{save_path}/save_img_{idx}.jpg', facecolor='grey', edgecolor='red')
            plt.close()
        else:
            plt.show()




path = '/data/zhicai/ckpts/Mgmlp/train/20211009-113958-nest_gmlp_s-224/checkpoint-50.pth.tar'
# show_weight(path,is_all = True, save_path='gmlp_s')
# summary_list=['/data/zhicai/ckpts/Mgmlp/train/20210924-223448-nest_gmlp_s-224/summary.csv',
# '/home/zhicai/Mglp/output/train/20210923-105647-nest_scgmlp_s-224/summary.csv']
# name_list = ['nest_gmlp_s_conv_pos',
# 'nest_gmlp_s_pos']
# draw_acc(summary_list,name_list)
show_para(path)
# show_weight(path, save_path='gmlp_s_pos_all(wight_and_posBias)')

# 




# from models.nest_gmlp import QuaMap,LearnedPosMap
# import torch

# gamma = 24

# model = QuaMap(dim=96,seq_len=196,blocks=16,gamma=gamma)
# model_2 = LearnedPosMap(dim=96,seq_len=196,blocks=16,gamma=gamma)


# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# # 创建resnet50网络
# # 创建输入网络的tensor
# tensor = (torch.randn([1,16,196,96]))

# # 分析FLOPs
# flops = FlopCountAnalysis(model, tensor)
# flops_2 = FlopCountAnalysis(model_2, tensor)
# print("FLOPs: ", flops.total())
# # 分析parameters
# print(parameter_count_table(model))

# print("FLOPs: ", flops_2.total())
# # 分析parameters
# print(parameter_count_table(model_2))