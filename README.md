# Parameterization of Cross-Token Relations with Relative Positional Encoding for Vision MLP

This is a Pytorch implementation of our paper. We have presented a new gating unit PoSGU 
which  replace the FC layer in SGU of [gMLP](https://proceedings.neurips.cc/paper/2021/hash/4cc05b35c2f937c5bd9e7d41d3686fff-Abstract.html) with relative positional encoding methods (Spercifically, LRPE and GQPE) and used it
as the key building block to develop a new vision MLP architecture
referred to as the PosMLP. We also hope this work will inspire further theoretical study of positional encoding
in vision MLPs and could have a mature application as in vision
Transformers.

![](figures/three-method.png)

Our code is based on the [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [attention-cnn](https://github.com/epfml/attention-cnn), [swim-transformer](https://github.com/microsoft/Swin-Transformer)

### Comparison with Recent MLP-like Models

| Model                | Parameters | Throughput | Image resolution | Top 1 Acc. | Download | Logs  |
| :------------------- | :--------- | :--------- | :--------------- | :--------- | :------- | :---- |
| EAMLP-14             | 30M        | 711 img/s  |       224        |  78.9%     |          |       |
| gMLP-S               | 20M        | -          |       224        |  79.6%     |          |       |
| ResMLP-S24           | 30M        | 715 img/s  |       224        |  79.4%     |          |       |
| ViP-Small/7 (ours)   | 25M        | 719 img/s  |       224        |  81.5%     | [link](https://drive.google.com/file/d/1cX6eauDrsGsLSZnqsX7cl0oiKX8Dzv5z/view?usp=sharing) | [log](https://github.com/Andrew-Qibin/VisionPermutator/blob/main/logs/vip_s7.log)    |
| EAMLP-19             | 55M        | 464 img/s  |       224        |  79.4%     |          |       |
| Mixer-B/16           | 59M        | -          |       224        |  78.5%     |          |       |
| ViP-Medium/7 (ours)  | 55M        | 418 img/s  |       224        |  82.7%     | [link](https://drive.google.com/file/d/15y5WMypthpbBFdc01E3mJCZit7q0Yn8m/view?usp=sharing) | [log](https://github.com/Andrew-Qibin/VisionPermutator/blob/main/logs/vip_m7.log)    |
| gMLP-B               | 73M        | -          |       224        |  81.6%     |          |       |
| ResMLP-B24           | 116M       | 231 img/s  |       224        |  81.0%     |          |       |
| ViP-Large/7          | 88M        | 298 img/s  |       224        |  83.2%     | [link](https://drive.google.com/file/d/14F5IXGXmB_3jrwK33Efae-WEb5D_G85c/view?usp=sharing) | [log](https://github.com/Andrew-Qibin/VisionPermutator/blob/main/logs/vip_L7.log)    |

The throughput is measured on a single machine with V100 GPU (32GB) with batch size set to 32.

Training ViP-Small/7 takes less than 30h on ImageNet for 300 epochs on a node with 8 A100 GPUs.

### Requirements

```
torch>=1.4.0
torchvision>=0.5.0
pyyaml
timm==0.4.5
apex if you use 'apex amp'
```

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Validation
Replace DATA_DIR with your imagenet validation set path and MODEL_DIR with the checkpoint path
```
CUDA_VISIBLE_DEVICES=0 bash eval.sh /path/to/imagenet/val /path/to/checkpoint
```

### Training

Command line for training on 8 GPUs (V100)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet --model vip_s7 -b 256 -j 8 --opt adamw --epochs 300 --sched cosine --apex-amp --img-size 224 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```


### Reference
You may want to cite:
```
@misc{hou2021vision,
    title={Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition},
    author={Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng},
    year={2021},
    eprint={2106.12368},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


### License
This repository is released under the MIT License as found in the [LICENSE](LICENSE) file.
