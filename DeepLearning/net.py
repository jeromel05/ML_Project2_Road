import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
import glob
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import random

from database import *
from helpers import *
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, dilation = 2, padding = 2, stride = 1)
        self.conv2 = nn.Conv2d(64, 128, 3, dilation = 2, padding = 2, stride = 1)
        self.conv3 = nn.Conv2d(128, 256, 3, dilation = 2, padding = 2, stride = 1)
        self.conv4 = nn.Conv2d(256, 512, 3, dilation = 2, padding = 2, stride = 1)
        self.conv5 = nn.Conv2d(512, 1024, 3, dilation = 2, padding = 2, stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LeakyReLu()
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)
        self.norm4 = nn.BatchNorm2d(512)
        self.norm4 = nn.BatchNorm2d(1024)
        
        
        self.outconv = nn.Conv2d(64, 2, kernel_size=2)
        
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        print(x.size())
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv4(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv5(x)))
        print(x.size())
        x = self.up1(x)
        print(x.size())
        x = self.up1(x)
        print(x.size())
        x = self.up1(x)
        print(x.size())
        x = self.up1(x)
        print(x.size())
        
        x = self.outconv(x)
        print(x.size())
        x = self.out(x)
        print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

	

if __name__ == '__main__':
	net = Net()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters())
	epochs = 200;

	net.train()
	for i in range(epochs):
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()   # zero the gradient buffers
			output = net(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()    # Does the update

	print(loss)
	print(net)
