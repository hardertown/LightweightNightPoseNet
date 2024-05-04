import torch
import torchvision.models as models


# 获取预训练的 MobileNetV2 模型
mobilenet = models.mobilenet_v3_small()
mobilenet = list(mobilenet.children())

# 获取需要保留的层
features = torch.nn.Sequential(*mobilenet[:])

# 创建新的模型，只包含需要的层
new_mobilenetv = torch.nn.Sequential(*features)
net = torch.nn.Sequential(*mobilenet[:-2])

# 打印新模型的结构
print(net)
x = torch.randn((1, 3, 224, 224))
x = net(x)
print(x.shape)
# y1 = new_mobilenetv(x)
# # l = torch.nn.Conv2d(116, 232, 1)
# # y1 = l(y1)
# y2 = net(y1)
# print(y2.shape)