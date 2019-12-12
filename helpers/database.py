import numpy as np
import random
import h5py
import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras.utils as utl
from networks import *
from helpers import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

class Road_Segmentation_Database(utl.Sequence):
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
		
        imgY = imgY.reshape((*imgY.shape, 1))
        imgX = np.expand_dims(imgX, axis=0)
        imgY = np.expand_dims(imgY, axis=0)


        return (imgX/255, imgY > 50)
 
    def __len__(self):
        
        return self.sizeTrain 