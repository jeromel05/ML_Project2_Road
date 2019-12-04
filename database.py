import torchvision.transforms
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
from network import *
from helpers import *

# adapted from pytorch doc
class RotationTransform:
    """Rotate by a given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):

        return TF.rotate(x, self.angle)

class CropResizeTransform:
    """Rotate by a given angles if do is set to true."""
    def __init__(self, top, left, height, width, do):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.do = do
    def __call__(self, x):
        if self.do:
          x = TF.resized_crop(x, self.top, self.left, self.height, self.width, (400,400))

        return x

class Road_Segmentation_Database(torch.utils.data.Dataset):
    """Road Segmentation database. Reads a h5 for performance. Caches the whole h5 and performs transformations on the images."""
    def __init__(self, thing, training, frei_chen=False):
        super(Road_Segmentation_Database, self).__init__()
        self.hf_path = thing
        self.hf = h5py.File(self.hf_path, 'r')    
        self.training = training
        self.frei_chen = frei_chen

        if self.training:
            self.sizeTrain = len(self.hf['train'])

        else:
            self.sizeTrain = len(self.hf['test'])


    def __getitem__(self, index):
    
        hfFile = self.hf   
        
        # get input image and label image
        if self.training:
            imgX = hfFile['train'][index, ...]
            imgY = hfFile['train_groundtruth'][index, ...]
            imgFC = hfFile['train_frei_chen'][index, ...]
        else:
            imgX = hfFile['test'][index, ...]
            imgY = hfFile['test_groundtruth'][index, ...]
            imgFC = hfFile['train_frei_chen'][index, ...]

        # transform from numpy to PIL image
        imgX = Image.fromarray(imgX)
        imgY = Image.fromarray(imgY)
        imgFC = Image.fromarray(imgFC)

        # init transformations

        # create random rotation in one cardinal directions
        rotation = np.random.randint(4) * 90

        # create random crop with min 0.5 to 1.0 scale and random ratio
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            imgY, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.))
        
        # decide if we actually do the crop and resize
        do_resize_crop = np.random.randint(2)

        # create the same transformation to apply to both input and label
        transform = transforms.Compose([
            RotationTransform(rotation),
            CropResizeTransform(top, left, height, width, do_resize_crop),
            transforms.ToTensor(),
          ])
        
        # apply transformation
        tensorX = transform(imgX)
        tensorY = transform(imgY)
        tensorFC = transform(imgFC)

        if(self.frei_chen):
            tensorX = append_channel(tensorX, tensorFC)


        # send to GPU if it exists
        if torch.cuda.is_available():
            tensorX = tensorX.cuda()
            tensorY = tensorY.cuda()
        return (tensorX, tensorY)
 
    def __len__(self):
        
        return self.sizeTrain 

def load_dataset(path, training, frei_chen=False):

    dataset = Road_Segmentation_Database(path, training, frei_chen)

    # create pytorch dataloader with batchsize of 8
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    return loader

