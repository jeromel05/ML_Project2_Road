import numpy as np
import h5py
import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from network import *
from helpers import *




class Road_Segmentation_Database(keras.Sequence):
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
		
		
        return np.array((ImgX), (ImgY))
 
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

def load_dataset_patches(path, training, list_of_transforms, batch_size=8, forced_transform=None):

    dataset = Road_Segmentation_Database(path, training, list_of_transforms, forced_transform=forced_transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return loader

