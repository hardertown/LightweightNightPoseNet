import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile
# from model.nnet import Model
from model.lnnet_mobile_se import Model

# 创建模型实例
# model = Model(17).cuda()
model = Model().cuda()

# 打印模型结构和参数数量
# summary(model, (1, 28, 28))
# 计算模型的 FLOPs
input = torch.randn((1, 1, 224, 224)).cuda()
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops}, Params: {params}")
