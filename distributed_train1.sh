#!/bin/bash

gpus=4,5,6,7
gpun=4
master_port=2955
shift
CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port main_2.py "$@" -c '/home/zhicai/Mglp/config/gmlp/mixer_2.yaml'
