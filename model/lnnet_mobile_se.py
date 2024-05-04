import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = mobilenet_v3_small(True)
        self.backbone = list(self.base.children())
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.backbone = nn.Sequential(*self.backbone[:-2])
        self.se_layer = SELayer(576)
        self.final_layer = nn.Conv2d(576, 17*30, 1)
        self.head_x = nn.Linear(30*7*7, 224)
        self.head_y = nn.Linear(30*7*7, 224)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.se_layer(x)
        x = self.final_layer(x)

        x = x.view(batchsize, 17, -1)
        pred_x = self.head_x(x)
        pred_y = self.head_y(x)

        return pred_x, pred_y


if __name__ == '__main__':
    img = torch.randn(2, 1, 224, 224)
    net = Model()
    outx, outy= net(img)
    print(outx.shape)