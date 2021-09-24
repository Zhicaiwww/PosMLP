#!/bin/bash

gpus=0,1,2,3
gpun=4
master_port=2953
shift
CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port main.py "$@" -c '/home/zhicai/Mgmlp/config/gmlp/nest_gmlp_s_24_0.5data.yaml'
