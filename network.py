import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F
from database import *
from helpers import *



#cell content is taken and adapted from https://github.com/milesial/Pytorch-UNet
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(reflection padding => convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(reflection padding => convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ConvReflection(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpUnet(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid=False ):
        super(OutConv, self).__init__()
        self.sigmoid = sigmoid
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if(self.sigmoid):

            return self.conv(x)
        else:
            return nn.Sigmoid(self.conv(x))


#cell content is taken and adapted from https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = UpUNet(1024, 256, bilinear)
        self.up2 = UpUNet(512, 128, bilinear)
        self.up3 = UpUNet(256, 64, bilinear)
        self.up4 = UpUNet(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class EncodeDecodeNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):

      super(UNet, self).__init__()
      self.network = nn.Sequential(
            ConvReflection(3, 8, 11, 5),
            Down(8, 8),
            DoubleConv(8, 8),
            DoubleConv(8, 16),

      )

    def forward(self, x):
        x = self.network(x)
        return x

# create network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.network = nn.Sequential(
            # start kernel size 5 for better processing of original image
            nn.Conv2d(3, 8, 5, stride=1, padding=2), 

            # downsize for feature extraction
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(8), 
    
            # convolution
            nn.Conv2d(8, 8, 3, padding=1),  
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(8), 

            # convolution
            nn.Conv2d(8, 16, 3, padding=1), 
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16), 

            # convolution
            nn.Conv2d(16, 16, 3, stride=1, padding=1), 

            # downsize for feature extraction 
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16), 

            # convolution
            nn.Conv2d(16, 16, 3, stride=1, padding=1),

            # downsize for feature extraction  
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16), 

            # convolution
            nn.Conv2d(16, 32, 3, padding=1),  
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 

            # convolution  
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            
            # downsize for feature extraction  
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 

            # at min size for features find seperation with several convolutions
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 

            # upsample using nearest value to go back to original size progressively
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # convolution
            nn.Conv2d(32, 32, 3, padding=1),  
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 

            # convolution
            nn.Conv2d(32, 32, 3, padding=1),  
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32), 
    
            # upsample using nearest value to go back to original size progressively
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # convolution
            nn.Conv2d(32, 16, 3, padding=1),  
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16),

            # upsample using nearest value to go back to original size progressively
            nn.Upsample(scale_factor=2, mode='nearest'),

            # convolution
            nn.Conv2d(16, 16, 3, padding=1),  
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16), 
    
            # upsample using nearest value to go back to original size progressively
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # convolution with kernel size 1 for weight rescaling
            nn.Conv2d(16, 1, 1, padding=0),

            # end with sigmoid  
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.network(x)
        return x
