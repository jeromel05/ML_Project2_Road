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
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark=True


# adapted from pytorch doc
class RotationTransform:
    """Rotate by a given angles."""

    def __init__(self, angle_base=90):
        # create random rotation in one cardinal directions
        self.angle_base = angle_base
        self.angle_factor = 360/self.angle_base
        angle = np.random.randint(self.angle_factor) * self.angle_base
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

    def set_random(self):
        angle = np.random.randint(self.angle_factor) * self.angle_base
        self.angle = angle

class CropResizeTransform:
    """Rotate by a given angles if do is set to true."""
    def __init__(self,  chance, size=(400,400,3)):
        # create random crop with min 0.5 to 1.0 scale and random ratio
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            Image.fromarray(np.zeros(size).astype(np.uint8)), scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.))

        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.chance = chance
        self.size = size
        self.do = 0
    def __call__(self, x):
        if self.do < 1:
          x = TF.resized_crop(x, self.top, self.left, self.height, self.width, (400,400))
        return x

    def set_random(self):
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            Image.fromarray(np.zeros(self.size).astype(np.uint8)), scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.))
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.do = np.random.randint(self.chance)

def set_transform_random(list_of_transforms):
    for transform in list_of_transforms:
        transform.set_random()
    return list_of_transforms

class Road_Segmentation_Database(torch.utils.data.Dataset):
    """Road Segmentation database. Reads a h5 for performance. Caches the whole h5 and performs transformations on the images."""
    def __init__(self, thing, training, list_of_transforms, forced_transform=None):
        super(Road_Segmentation_Database, self).__init__()
        self.hf_path = thing
        self.hf = h5py.File(self.hf_path, 'r')    
        self.training = training
        self.list_of_transforms = list_of_transforms
        self.forced_transform = forced_transform
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
        else:
            imgX = hfFile['test'][index, ...]
            imgY = hfFile['test_groundtruth'][index, ...]

        # transform from numpy to PIL image
        imgX = Image.fromarray(imgX)
        imgY = Image.fromarray((imgY > 150).astype(np.float64)*255)

        # do transform do be done on both train and test
        if self.forced_transform is not None:
            imgX = self.forced_transform(imgX)
            imgY = self.forced_transform(imgY)

        list_of_transforms = set_transform_random(self.list_of_transforms).copy()
        list_of_transforms.append(transforms.ToTensor())
        if self.training:
            transform = transforms.Compose(list_of_transforms)
        else:
            transform = transforms.ToTensor()

        tensorX = transform(imgX)
        tensorY = transform(imgY)

        # send to GPU if it exists
        if torch.cuda.is_available():
            tensorX = tensorX.cuda()
            tensorY = tensorY.cuda()

        return (tensorX, tensorY)
 
    def __len__(self):
        
        return self.sizeTrain 

def load_dataset(path, training, list_of_transforms, batch_size=8, forced_transform=None):

    dataset = Road_Segmentation_Database(path, training, list_of_transforms, forced_transform=forced_transform)

    # create pytorch dataloader with batchsize of 8
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return loader

