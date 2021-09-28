import torch
from thop import profile
import warnings
import models
# from timm import models 


import argparse
parser = argparse.ArgumentParser(description='ViP Training and Evaluating')
parser.add_argument('-t', '--type', default='thop', type=str,
                    help='check which type to summary model')
args = parser.parse_args()

models.list_models()
warnings.filterwarnings('ignore')
# input=torch.randn([1,3,224,224])
kwargs={'depth_conv': True,
  'gamma': 8,
  'pos_emb': False,
  'stem_name' : 'PatchEmbed',
  'blockwise': True}
model = models.nest_gmlp_s(**kwargs)


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
else :
    input=torch.randn([1,3,224,224])
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())

    flop,para = profile(model,inputs=(input,))
    print('with input size {}, model has para numbers {} and flops {} '.format(input.size(),para,flop))