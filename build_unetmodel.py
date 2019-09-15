import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)        
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = outconv(16, n_classes)
        self.Dropout = nn.Dropout(0.6)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.Dropout(x2)
        x3 = self.down2(x2)
        x3 = self.Dropout(x3)
        x4 = self.down3(x3)
        x4 = self.Dropout(x4)
        x5 = self.down4(x4)
        x5 = self.Dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = F.relu(self.outc(x))
        return x
