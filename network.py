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

