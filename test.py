import torch
from thop import profile
import warnings
import models
import torchvision.transforms as transforms
from torch.utils import data
import os
from torch.utils.data import Dataset
import PIL as pil
# from timm import models 


import argparse
parser = argparse.ArgumentParser(description='ViP Training and Evaluating')
parser.add_argument('-t', '--type', default='', type=str,
                    help='check which type to summary model')
args = parser.parse_args()

warnings.filterwarnings('ignore')
# input=torch.randn([1,3,224,224])
# input=torch.randn([1,3,224,224])
kwargs={
        'stem_name' :'Nest_ConvolutionalEmbed',# 'PatchEmbed' ,#'Nest_ConvolutionalEmbed',#'Nest_ConvolutionalEmbed'
        'quadratic' : True,
        'pos_only': True,
        'gamma': (128,32,32,64),
        "embed_dims" :(96, 192, 384,768),
        "depths" : (2, 2, 6,2),
        "mlp_ratio" : (4,4,4,2),
        "num_levels": 4,
        'generalized': True,
        'img_size':(224,224),
        'chunks':2,
        'num_blocks':(16,16,4,1),
        'depth_conv':True

  }
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((123.675, 116.28, 103.53),(58.395, 57.12, 57.375))])
# path = '/data/zhicai/ckpts/Mgmlp/train/20211021-090705-nest_gmlp_s_b4-224/checkpoint-117.pth.tar'
# pretrained="/data/zhicai/ckpts/Mgmlp/checkpoint-37.pth_nopoolnorm_81.3.tar"
model = models.PosMLP_T14_224()
# model.load_state_dict(torch.load(path)['state_dict_ema'])
img = torch.randn([1,3,224,224])
# model = models.mixer_s16_224()
if args.type == 'ptflops':
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

elif args.type == 'torchsummary':
    from torchsummary import summary
    summary(model.cuda(),(3,224,224))


elif args.type == 'thop' :
    input=torch.randn([1,3,224,224])
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
    
    flop,para = profile(model,inputs=(input,))
    print('with input size {}, model has para numbers {} and flops {} '.format(input.size(),para,flop))

else:
    import torch
    # from torchvision.models import resnet50
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # 创建resnet50网络
    # 创建输入网络的tensor

    # tensor = (torch.rand(1, 3, 224,224),)
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
    # 分析FLOPs
    flops = FlopCountAnalysis(model, img)
    print("FLOPs: ", flops.total())

    # 分析parameters
    print(parameter_count_table(model))
