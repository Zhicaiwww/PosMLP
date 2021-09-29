import csv
import matplotlib.pyplot as plt 
import os
import numpy as np
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
    ckpt = torch.load(path)
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
        if "gate_unit.proj" in k and 'bias' in k:
            to_kb.append(k)
            to_vb.append(v.cpu())
        if 'relative_position_bias_table' in k:
            to_kp.append(k)
            to_vp.append(v.cpu())
        if 'relative_position_index' in k:
            to_kidx.append(k)
            to_vidx.append(v.cpu())
    print(f"number of figures is {len(to_kw)}")
    if len(to_kp) is not 0:
        import math
        bias_map =[]
        for vp in to_vp:
            h,w = to_vidx[0].size()
            bias_map.append(vp[to_vidx[0].view(-1)].view(h,w))
    idxs = range(len(to_kw)) if is_all else indexes
    for idx in idxs:
        fig = plt.figure(figsize=(16, 16),tight_layout=True)
        ax = fig.add_subplot(2,2,1)
        ax.imshow(to_vw[idx])
        ax = fig.add_subplot(2,2,2)
        if len(to_kp) is 0:
            ax.plot(to_vb[idx])
        else:
            ax.imshow(bias_map[idx])
        ax = fig.add_subplot(2,2,3)
        ax.imshow(to_vw[idx]+bias_map[idx])
        ax = fig.add_subplot(2,2,4)
        ax.plot(to_vw[idx][50],color = color_list[0],label='vw')
        ax.plot(bias_map[idx][50],color = color_list[1],label='vb')
        ax.plot(to_vw[idx][50]+bias_map[idx][50],color = color_list[2],label='vw+vb')
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
    ckpt = torch.load(path)
    for i,v in ckpt['state_dict_ema'].items():
        if 'block_proj_n_weight' in i :
            print(i,v)
            break

path = '/data/zhicai/ckpts/Mgmlp/train/20210928-084821-gmlp_s16_224-224/checkpoint-15.pth.tar'
# show_weight(path,is_all = True, save_path='gmlp_s')
# summary_list=['/data/zhicai/ckpts/Mgmlp/train/20210924-223448-nest_gmlp_s-224/summary.csv',
# '/home/zhicai/Mglp/output/train/20210923-105647-nest_scgmlp_s-224/summary.csv']
# name_list = ['nest_gmlp_s_conv_pos',
# 'nest_gmlp_s_pos']
# draw_acc(summary_list,name_list)

# show_weight(path, save_path='gmlp_s_pos_all(wight_and_posBias)')
show_para(path)