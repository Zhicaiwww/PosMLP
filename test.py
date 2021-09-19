import torch
from thop import profile
import warnings
import models
models.list_models()
warnings.filterwarnings('ignore')
input=torch.randn([1,3,224,224])

model = models.gmlp_s16_224()
for name,parameters in model.named_parameters():
    print(name,':',parameters.size())

# flop,para = profile(model,inputs=(input,))
# print('with input size {}, model has para numbers {} and flops {} '.format(input.size(),para,flop))