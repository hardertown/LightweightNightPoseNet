import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CoordConv, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        b = x.shape[0]
        h = x.shape[2]
        w = x.shape[3]

        gx = torch.arange(w, dtype=torch.float32) / (w - 1.) * 2.0 - 1.
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
        gx.requires_grad = False

        gy = torch.arange(h, dtype=torch.float32) / (h - 1.) * 2.0 - 1.
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
        gy.requires_grad = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gx = gx.to(device)
        gy = gy.to(device)

        y = torch.cat([x, gx, gy], dim=1)
        y = self.conv(y)

        return y


class Deconv_layer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_cat=True):
        super(Deconv_layer, self).__init__()
        self.is_cat = is_cat

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.nolmal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.is_cat:
            x1 = torch.cat((x1, x2), dim=1)
            x1 = self.fusion(x1)
        else:
            x1 = self.nolmal(x1)
        return x1


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.base = shufflenet_v2_x1_0(True)
        self.backbone = list(self.base.children())
        self.layer1 = CoordConv(1, 3, 1, 1, 0)
        self.layer2 = nn.Sequential(*self.backbone[:-4])  # 116, 28, 28
        self.layer3 = nn.Sequential(*self.backbone[-4])  # 232, 14, 14
        self.layer4 = nn.Sequential(*self.backbone[-3])  # 464, 7, 7
        self.up1 = Deconv_layer(464, 232 + 232, 232)
        self.up2 = Deconv_layer(232, 116 + 116, 116)
        self.up3 = Deconv_layer(116, 0, 64, is_cat=False)

        self.conv_last = nn.Conv2d(64, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        s1 = self.layer1(input)  # 3, 224, 224
        s2 = self.layer2(s1)  # 116, 28, 28
        s3 = self.layer3(s2)  # 232, 14, 14
        s4 = self.layer4(s3)  # 464, 7, 7

        d1 = self.up1(s4, s3)  # 232,14,14
        d2 = self.up2(d1, s2)  # 116,28,28
        d3 = self.up3(d2, 0)  # 64,56,56

        out = self.conv_last(d3)  # 17,56,56 一个点一个通道 按顺序第一个通道是第一个点
        return self.sigmoid(out)


if __name__ == '__main__':
    model = Model(17)
    x = torch.randn((1, 1, 224, 224))
    y = model(x)
    print(y.shape)
    print(y.data[0, 1])
