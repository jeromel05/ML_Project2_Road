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
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark=True


#cell content is taken and adapted from https://github.com/milesial/Pytorch-UNet
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_padding(kernel_size = 3, dilation = 1):
  """ Get the required padding to be applied in order to extend the image for convolution.

    Args:
      kernel_size: the size of the applied kernel. Should be odd!
      dilation: the dilation factor applied to the kernel. Must be int!
  """
  return (kernel_size-1)//2 * dilation

class DoubleConv(nn.Module):
    """(reflection padding => convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Conv(nn.Module):
    """(reflection padding => convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ConvReflection(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpUNet(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels) # padding is done in the forward

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
    def __init__(self, in_channels, out_channels, sigmoid=True ):
        super(OutConv, self).__init__()
        self.sigmoid = sigmoid
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        if(not self.sigmoid):

            return self.conv(x)
        else:
            return nn.Sigmoid()(self.conv(x))

class UpUNet_crop(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        cropped_x2 = x2.narrow(2, diffY//2, x1.size()[2])
        cropped_x2 = cropped_x2.narrow(3, diffX//2, x1.size()[3])
        x = torch.cat([cropped_x2, x1], dim=1)
        return self.conv(x)

class UpD1D2UNet_crop(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv1 = Conv(in_channels, out_channels*2, kernel_size=3, padding=0, dilation=1) # padding is done in the forward
        self.conv2 = Conv(out_channels*2, out_channels, kernel_size=3, padding=0, dilation=2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        cropped_x2 = x2.narrow(2, diffY//2, x1.size()[2])
        cropped_x2 = cropped_x2.narrow(3, diffX//2, x1.size()[3])
        x = torch.cat([cropped_x2, x1], dim=1)
        return self.conv2(self.conv1(x))

class UpK4K3UNet_crop(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv1 = Conv(in_channels, out_channels*2, kernel_size=4, padding=0) # padding is done in the forward
        self.conv2 = Conv(out_channels*2, out_channels, kernel_size=3, padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        cropped_x2 = x2.narrow(2, diffY//2, x1.size()[2])
        cropped_x2 = cropped_x2.narrow(3, diffX//2, x1.size()[3])
        x = torch.cat([cropped_x2, x1], dim=1)
        return self.conv2(self.conv1(x))

class UpD2D1UNet_crop(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv1 = Conv(in_channels, out_channels*2, kernel_size=3, padding=0, dilation=2) # padding is done in the forward
        self.conv2 = Conv(out_channels*2, out_channels, kernel_size=3, padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        cropped_x2 = x2.narrow(2, diffY//2, x1.size()[2])
        cropped_x2 = cropped_x2.narrow(3, diffX//2, x1.size()[3])
        x = torch.cat([cropped_x2, x1], dim=1)
        return self.conv2(self.conv1(x))

class ActualUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sigmoid=False):
        super(ActualUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.pad_mirror = nn.ReflectionPad2d( (588 - 400) // 2)
        self.inc = DoubleConv(n_channels, 64, kernel_size=3, padding=0)
        self.down1 = Down(64, 128, kernel_size=3, padding=0)
        self.down2 = Down(128, 256, kernel_size=3, padding=0)
        self.down3 = Down(256, 512, kernel_size=3, padding=0)
        self.down4 = Down(512, 512, kernel_size=3, padding=0)
        self.up1 = UpUNet_crop(1024, 256, kernel_size=3, padding=0, bilinear=bilinear)
        self.up2 = UpK4K3UNet_crop(512, 128, bilinear=bilinear)
        self.up3 = UpUNet_crop(256, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.up4 = UpUNet_crop(128, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.outc = OutConv(64, n_classes, sigmoid)

    def forward(self, x):
        x0 = self.pad_mirror(x)
        # print(x0.size())
        x1 = self.inc(x0)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        # print(x.size())
        x = self.up2(x, x3)
        # print(x.size())
        x = self.up3(x, x2)
        # print(x.size())
        x = self.up4(x, x1)
        # print(x.size())
        logits = self.outc(x)
        # print(logits.size())
        return logits

class ActualNoPairKernelUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sigmoid=False):
        super(ActualNoPairKernelUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.pad_mirror = nn.ReflectionPad2d( (588 - 400) // 2)
        self.inc = DoubleConv(n_channels, 64, kernel_size=3, padding=0)
        self.down1 = Down(64, 128, kernel_size=3, padding=0)
        self.down2 = Down(128, 256, kernel_size=3, padding=0)
        self.down3 = Down(256, 512, kernel_size=3, padding=0)
        self.down4 = Down(512, 512, kernel_size=3, padding=0)
        self.up1 = UpUNet_crop(1024, 256, kernel_size=3, padding=0, bilinear=bilinear)
        self.up2 = UpUNet_crop(512, 128, kernel_size=3, padding=0, bilinear=bilinear)
        self.up3 = UpD2D1UNet_crop(256, 64, bilinear=bilinear)
        self.up4 = UpUNet_crop(128, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.outc = OutConv(64, n_classes, sigmoid)

    def forward(self, x):
        x0 = self.pad_mirror(x)
        # print(x0.size())
        x1 = self.inc(x0)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        # print(x.size())
        x = self.up2(x, x3)
        # print(x.size())
        x = self.up3(x, x2)
        # print(x.size())
        x = self.up4(x, x1)
        # print(x.size())
        logits = self.outc(x)
        # print(logits.size())
        return logits

class ResizeUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sigmoid=False):
        super(ResizeUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.pad_mirror = nn.ReflectionPad2d( (588 - 400) // 2) # WRONG CHANGE FOR PAPER VALUES
        self.inc = DoubleConv(n_channels, 64, kernel_size=3, padding=0)
        self.down1 = Down(64, 128, kernel_size=3, padding=0)
        self.down2 = Down(128, 256, kernel_size=3, padding=0)
        self.down3 = Down(256, 512, kernel_size=3, padding=0)
        self.down4 = Down(512, 512, kernel_size=3, padding=0)
        self.up1 = UpUNet_crop(1024, 256, kernel_size=3, padding=0, bilinear=bilinear)
        self.up2 = UpUNet_crop(512, 128, kernel_size=3, padding=0, bilinear=bilinear)
        self.up3 = UpUNet_crop(256, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.up4 = UpUNet_crop(128, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.outc = OutConv(64, n_classes, sigmoid)

    def forward(self, x):
        x0 = self.pad_mirror(x)
        # print(x0.size())
        x1 = self.inc(x0)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        # print(x.size())
        x = self.up2(x, x3)
        # print(x.size())
        x = self.up3(x, x2)
        # print(x.size())
        x = self.up4(x, x1)
        # print(x.size())
        logits = self.outc(x)
        # print(logits.size())
        return logits

class ActualUNetDilated(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sigmoid=False):
        super(ActualUNetDilated, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.pad_mirror = nn.ReflectionPad2d( (760 - 400) // 2)
        self.inc = DoubleConv(n_channels, 64, kernel_size=3, padding=0, dilation=2)
        self.down1 = Down(64, 128, kernel_size=3, padding=0, dilation=2)
        self.down2 = Down(128, 256, kernel_size=3, padding=0, dilation=2)
        self.down3 = Down(256, 512, kernel_size=3, padding=0, dilation=2)
        self.down4 = Down(512, 512, kernel_size=3, padding=0, dilation=2)
        self.up1 = UpUNet_crop(1024, 256, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.up2 = UpD1D2UNet_crop(512, 128, bilinear=bilinear)
        self.up3 = UpUNet_crop(256, 64, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.up4 = UpUNet_crop(128, 64, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.outc = OutConv(64, n_classes, sigmoid)

    def forward(self, x):
        x0 = self.pad_mirror(x)
        # print(x0.size())
        x1 = self.inc(x0)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        # print(x.size())
        x = self.up2(x, x3)
        # print(x.size())
        x = self.up3(x, x2)
        # print(x.size())
        x = self.up4(x, x1)
        # print(x.size())
        logits = self.outc(x)
        # print(logits.size())
        return logits



class DeeperUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sigmoid=False):
        super(DeeperUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.pad_mirror = nn.ReflectionPad2d( (764 - 388) // 2)
        self.inc = DoubleConv(n_channels, 64, kernel_size=3, padding=0)
        self.down1 = Down(64, 128, kernel_size=3, padding=0)
        self.down2 = Down(128, 256, kernel_size=3, padding=0)
        self.down3 = Down(256, 512, kernel_size=3, padding=0)
        self.down4 = Down(512, 1024, kernel_size=3, padding=0)
        self.down5 = Down(1024, 1024, kernel_size=3, padding=0)
        self.up1 = UpUNet_crop(2048, 512, kernel_size=3, padding=0, bilinear=bilinear)
        self.up2 = UpUNet_crop(1024, 256, kernel_size=3, padding=0, bilinear=bilinear)
        self.up3 = UpUNet_crop(512, 128, kernel_size=3, padding=0, bilinear=bilinear)
        self.up4 = UpUNet_crop(256, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.up5 = UpUNet_crop(128, 64, kernel_size=3, padding=0, bilinear=bilinear)
        self.outc = OutConv(64, n_classes, sigmoid)

    def forward(self, x):
        x0 = self.pad_mirror(x)
        # print(x0.size())
        x1 = self.inc(x0)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x6 = self.down5(x5)
        # print(x5.size())
        x = self.up1(x6, x5)
        # print(x.size())
        x = self.up2(x, x4)
        # print(x.size())
        x = self.up3(x, x3)
        # print(x.size())
        x = self.up4(x, x2)
        # print(x.size())
        x = self.up5(x, x1)
        # print(x.size())
        logits = self.outc(x)
        # print(logits.size())
        return logits

class DeeperUNetDilated(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, sigmoid=False):
        super(DeeperUNetDilated, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.pad_mirror = nn.ReflectionPad2d( (1144 - 392) // 2)
        self.inc = DoubleConv(n_channels, 64, kernel_size=3, padding=0, dilation=2)
        self.down1 = Down(64, 128, kernel_size=3, padding=0, dilation=2)
        self.down2 = Down(128, 256, kernel_size=3, padding=0, dilation=2)
        self.down3 = Down(256, 512, kernel_size=3, padding=0, dilation=2)
        self.down4 = Down(512, 1024, kernel_size=3, padding=0, dilation=2)
        self.down5 = Down(1024, 1024, kernel_size=3, padding=0, dilation=2)
        self.up1 = UpUNet_crop(2048, 512, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.up2 = UpUNet_crop(1024, 256, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.up3 = UpUNet_crop(512, 128, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.up4 = UpUNet_crop(256, 64, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)
        self.up5 = UpUNet_crop(128, 64, kernel_size=3, padding=0, dilation=2, bilinear=bilinear)

        self.outc = OutConv(64, n_classes, sigmoid)

    def forward(self, x):
        x0 = self.pad_mirror(x)
        # print(x0.size())
        x1 = self.inc(x0)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x6 = self.down5(x5)
        # print(x5.size())
        x = self.up1(x6, x5)
        # print(x.size())
        x = self.up2(x, x4)
        # print(x.size())
        x = self.up3(x, x3)
        # print(x.size())
        x = self.up4(x, x2)
        # print(x.size())
        x = self.up5(x, x1)
        # print(x.size())
        logits = self.outc(x)
        # print(logits.size())
        return logits


#cell content is taken and adapted from https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, sigmoid=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, kernel_size=3, padding=0)
        self.down2 = Down(128, 256, kernel_size=3, padding=0)
        self.down3 = Down(256, 512, kernel_size=3, padding=0)
        self.down4 = Down(512, 512, kernel_size=3, padding=0)
        self.up1 = UpUNet(1024, 256, bilinear)
        self.up2 = UpUNet(512, 128, bilinear)
        self.up3 = UpUNet(256, 64, bilinear)
        self.up4 = UpUNet(128, 64, bilinear)
        self.outc = OutConv(64, n_classes, sigmoid)

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

# create network
class Net(nn.Module):
    def __init__(self, sigmoid=True, n_channels=3):
        super(Net, self).__init__()
        self.sigmoid = sigmoid


        self.network = nn.Sequential(
            # start kernel size 5 for better processing of original image
            nn.Conv2d(n_channels, 8, 5, stride=1, padding=2), 

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


        )

    def forward(self, x):
        # end with sigmoid if needed
        x = self.network(x)
        return nn.Sigmoid()( x ) if self.sigmoid else x 
