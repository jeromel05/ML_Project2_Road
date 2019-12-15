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
from skimage.transform import resize

class Road_Segmentation_Database(utl.Sequence):
    """Road Segmentation database. Reads a h5 for performance. Caches the whole h5 and performs transformations on the images."""
    def __init__(self, thing, training, batchsize=None, input_size=(400,400), output_size=(400,400)):
        super(Road_Segmentation_Database, self).__init__()
        self.hf_path = thing
        self.hf = h5py.File(self.hf_path, 'r')    
        self.training = training
        self.input_size = input_size
        self.output_size = output_size
        if self.training:
            self.sizeTrain = len(self.hf['train'])

        else:
            self.sizeTrain = len(self.hf['test'])

        if batchsize is None:
            batchsize = self.sizeTrain

    def __getitem__(self, index):
    
        hfFile = self.hf   
        
        # get input image and label image
        if self.training:
            imgX = hfFile['train'][index, ...]
            imgY = hfFile['train_groundtruth'][index, ...]
        else:
            imgX = hfFile['test'][index, ...]
            imgY = hfFile['test_groundtruth'][index, ...]
		

        imgX = resize(imgX, (*self.input_size,3)) # divides per 255
        imgY = resize(imgY, (self.output_size)) # divides per 255



        imgY = imgY.reshape((*imgY.shape, 1))
        imgX = to_single_batch(imgX)
        imgY = to_single_batch(imgY)


        return (imgX, imgY > 0.5)
 
    def __len__(self):
        
        return self.sizeTrain 

# class Shuffle_Batches(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
