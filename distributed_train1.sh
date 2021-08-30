#!/bin/bash

gpus=4,5,6
gpun=3
master_port=2955
shift
CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port main.py "$@" -c '/home/zhicai/Mgmlp/config/gmlp/SCG3gmlp_convstem_s_24_0.5data.yaml'