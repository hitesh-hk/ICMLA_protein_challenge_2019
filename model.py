import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, dilation=4, padding=4)

def concat(xs):
    return torch.cat(xs, 1)

class Conv3BN(nn.Module):
    def __init__(self, in_, out, bn=False):
        super(Conv3BN,self).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

class UNetModule(nn.Module):
    def __init__(self, in_, out):
        super(UNetModule,self).__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)      
    
class UNet11(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 num_channels=3,
                ):
        super(UNet11, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        encoder = models.vgg11(pretrained=True).features
        self.relu = encoder[1]
        
        #self.mean = (0.485, 0.456, 0.406)
        #self.std = (0.229, 0.224, 0.225)
                
        
        # try to use 8-channels as first input
        if num_channels==3:
            self.conv1 = encoder[0]
        else:
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))            
        
        self.conv2 = encoder[3]
        self.conv3s = encoder[6]
        self.conv3 = encoder[8]
        self.conv4s = encoder[11]
        self.conv4 = encoder[13]
        self.conv5s = encoder[16]
        self.conv5 = encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        self.dec1_ce = nn.Conv2d(num_filters * (2 + 1), num_filters, 3, dilation=4, padding=4)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.Dropout = nn.Dropout(0.)
    def require_encoder_grad(self,requires_grad):
        blocks = [self.conv1,
                  self.conv2,self.final,
                  self.conv3s,
                  self.conv3,
                  self.conv4s,
                  self.conv4,
                  self.conv5s,
                  self.conv5]
        
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad
                
    def forward(self, x):
        conv1 = self.Dropout(self.relu(self.conv1(x)))
        #print(conv1.size())
        conv2 = self.Dropout(self.relu(self.conv2(self.pool(conv1))))
        #print(conv2.size())
        conv3s = self.Dropout(self.relu(self.conv3s(self.pool(conv2))))
        #print(conv3s.size())
        conv3 = self.Dropout(self.relu(self.conv3(conv3s)))
        #print(conv3.size())
        conv4s = self.Dropout(self.relu(self.conv4s(self.pool(conv3))))
        #print(conv4s.size())
        conv4 = self.Dropout(self.relu(self.conv4(conv4s)))
        #print(conv4.size())
        conv5s = self.Dropout(self.relu(self.conv5s(self.pool(conv4))))
        #print(conv5s.size())
        conv5 = self.Dropout(self.relu(self.conv5(conv5s)))
        #print(conv5.size())
        center = self.center(self.pool(conv5))
        #print(center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        dec1_ce = self.dec1_ce(torch.cat([dec2, conv1], 1))
        x1 = F.relu(self.final(dec1))                      # for regression
        x2 = F.sigmoid(self.final(dec1_ce))				   # for cross entropy
        return x1 , x2
        #return self.final(dec1)
