import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = shufflenet_v2_x0_5(True)
        self.backbone = list(self.base.children())
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.backbone = nn.Sequential(*self.backbone[:-2])
        self.final_layer = nn.Conv2d(192, 17*30, 1)
        self.head_x = nn.Linear(30*7*7, 224)
        self.head_y = nn.Linear(30*7*7, 224)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.backbone(x)
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
