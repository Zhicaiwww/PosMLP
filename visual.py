import csv
import matplotlib.pyplot as plt 
import os
import numpy as np

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
    for path, name in zip(summary_list,name_list):
        acc,ema_acc,loss=getdata(path)
        ax1.plot(acc,label=f'{name}')
        ax2.plot(loss,label=f'{name}')    

    ax1.ylabel('ACC1')
    ax1.xlabel('epochs')
    ax1.legend()
    ax2.ylabel('Loss')
    ax2.xlabel('epochs')
    ax2.legend()
    plt.show()

import torch
def show_weight(path,indexes=[0,-1],is_all = False,save_path=None):
    ckpt = torch.load(path)
    to_kw=[]
    to_vw=[]
    to_kb=[]
    to_vb=[]
    # for k,v in ckpt['state_dict_ema'].items():
    #     if "gate.proj_list" in k and 'weight' in k:
    #         to_kw.append(k)
    #         to_vw.append(v)
    #     if "gate.proj_list" in k and 'bias' in k:
    #         to_kb.append(k)
    #         to_vb.append(v)
    for k,v in ckpt.items():
        if "gate.proj" in k and 'weight' in k:
            to_kw.append(k)
            to_vw.append(v)
        if "gate.proj" in k and 'bias' in k:
            to_kb.append(k)
            to_vb.append(v)
    print(f"number of figures is {len(to_kw)}")
    idxs = range(len(to_kw)) if is_all else indexes
    for idx in idxs:
        fig = plt.figure(figsize=(16, 8),tight_layout=True)
        ax = fig.add_subplot(1,2,1)
        ax.imshow(to_vw[idx].cpu())
        ax = fig.add_subplot(1,2,2)
        ax.plot(to_vb[idx].cpu())
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


# path = '/home/zhicai/Mgmlp/gmlp_s16_224_raa-10536d42.pth'
# show_weight(path,is_all = True, save_path='gmlp_s')