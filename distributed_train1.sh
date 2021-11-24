#!/bin/bash

gpus=0,1,2,3,4,5,6,7
gpun=8
master_port=2955
shift
CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port main.py "$@" -c '/home/zhicai/Mglp/config/gmlp/nest_b.yaml'
