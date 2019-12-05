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

    def __init__(self):
        # create random rotation in one cardinal directions
        angle = np.random.randint(4) * 90
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

    def set_random(self):
        angle = np.random.randint(4) * 90
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
    def __init__(self, thing, training, list_of_transforms, frei_chen=False):
        super(Road_Segmentation_Database, self).__init__()
        self.hf_path = thing
        self.hf = h5py.File(self.hf_path, 'r')    
        self.training = training
        self.frei_chen = frei_chen
        self.list_of_transforms = list_of_transforms

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

        list_of_transforms = set_transform_random(self.list_of_transforms).copy()
        list_of_transforms.append(transforms.ToTensor())
        transform = transforms.Compose(list_of_transforms)

        tensorX = transform(imgX)
        tensorY = transform(imgY)
        

        if(self.frei_chen):
            imgFC = Image.fromarray(imgFC)
            tensorFC = transform(imgFC)
            tensorX = append_channel(tensorX, tensorFC)

        # send to GPU if it exists
        if torch.cuda.is_available():
            tensorX = tensorX.cuda()
            tensorY = tensorY.cuda()
        return (tensorX, tensorY)
 
    def __len__(self):
        
        return self.sizeTrain 

def load_dataset(path, training, list_of_transforms, batch_size=8, frei_chen=False):

    dataset = Road_Segmentation_Database(path, training, list_of_transforms, frei_chen)

    # create pytorch dataloader with batchsize of 8
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return loader

