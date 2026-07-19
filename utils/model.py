import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
from torch.utils.checkpoint import checkpoint

class DConv(nn.Module):
    def __init__(self, inc, outc):
        super(DConv, self).__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dconv(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()

        self.conv = DConv(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)

class HoleNet(nn.Module):
    def __init__(self):
        super(HoleNet,self).__init__()
        self.n_classes = 1
        # mobilenet requirement, pretrained weight used as well as architecture for input 3 channels implicitly

        mobilenet = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT
        )

        features = mobilenet.features

        self.encoder1 = features[:2]
        self.encoder2 = features[2:4]
        self.encoder3 = features[4:7]
        self.encoder4 = features[7:13]
        self.encoder5 = features[13:]

        self.up1 = Up(960, 112, 256)
        self.up2 = Up(256, 40, 128)
        self.up3 = Up(128, 24, 64)
        self.up4 = Up(64, 16, 32)

        self.out = nn.Conv2d(32, self.n_classes, kernel_size=1)

        self.checkpointing = False

    def forward(self, x):
        input_size = x.shape[-2:]

        if self.checkpointing:
            x1 = checkpoint(self.encoder1, x, use_reentrant=False)
            x2 = checkpoint(self.encoder2, x1, use_reentrant=False)
            x3 = checkpoint(self.encoder3, x2, use_reentrant=False)
            x4 = checkpoint(self.encoder4, x3, use_reentrant=False)
            x5 = checkpoint(self.encoder5, x4, use_reentrant=False)

            x = checkpoint(self.up1, x5, x4, use_reentrant=False)
            x = checkpoint(self.up2, x, x3, use_reentrant=False)
            x = checkpoint(self.up3, x, x2, use_reentrant=False)
            x = checkpoint(self.up4, x, x1, use_reentrant=False)

        else:
            x1 = self.encoder1(x)
            x2 = self.encoder2(x1)
            x3 = self.encoder3(x2)
            x4 = self.encoder4(x3)
            x5 = self.encoder5(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        x = self.out(x)

        return F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

    def use_checkpointing(self):
        self.checkpointing = True