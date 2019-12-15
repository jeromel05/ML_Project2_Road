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
from database import *
from helpers import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def get_all_network_names():
    """Returns all network names"""
    names = ["unet_ugly"]

    return names



def unet_ugly(pretrained_weights = None,input_size = (400,400,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['acc',f1_m,precision_m, recall_m])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def double_conv(output_channels, kernel_size=3, padding=False, dilation_rate=(1, 1), dropout=False, activation='relu', kernel_initializer = 'he_normal'):
    padding='valid' if not padding else 'same'

    layer = Sequential([
        Conv2D(output_channels, kernel_size, activation = activation, padding = padding, dilation_rate=dilation_rate, kernel_initializer = kernel_initializer),
        Conv2D(output_channels, kernel_size, activation = activation, padding = padding, dilation_rate=dilation_rate, kernel_initializer = kernel_initializer),
    ])
    if dropout:
        layer.add(Dropout(0.2)) # 0.2 from paper of Sungheon Park and Nojun Kwak
    return layer

def down(maxpool=True, kernel_size=3,  activation='relu', kernel_initializer = 'he_normal'):
    if maxpool:
        return MaxPooling2D(pool_size=(2, 2))
    else:
        return Conv2D(output_channels, kernel_size, strides=(2, 2), activation = activation, padding = 'valid', kernel_initializer = kernel_initializer)

def up(output_channels, kernel_size=3, activation='relu', kernel_initializer = 'he_normal'):

    return Conv2DTranspose(output_channels, kernel_size, strides=(2, 2), padding='valid', kernel_initializer=kernel_initializer) # glorot_uniform

def merge(down, up):
    if padding: # add padding to cropped
        return concatenate([down, up], axis = 3)
    else:       # add padding to cropped
        diff = np.abs(down.shape[1] - up.shape[1])//2 # has to be a square shape
        cropped_down = down[diff:-diff,diff:-diff,:]
        return concatenate([cropped_down, up], axis = 3)


def unet(pretrained_weights = None,input_size = (400,400,3), kernel_size=3, start_output_size=64, dropout=False,dilation_rate=(1, 1), activation='relu' , kernel_initializer='he_normal'):
    
    inputs = Input(input_size)
    
    double_conv1 = double_conv(output_channels=start_output_size, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(inputs)
    down1 = down()(double_conv1)
    
    double_conv2 = double_conv(output_channels=start_output_size*2, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(down1)
    down2 = down()(double_conv2)
    
    double_conv3 = double_conv(output_channels=start_output_size*4, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(down2)
    down3 = down()(double_conv3)
    
    double_conv4 = double_conv(output_channels=start_output_size*8, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(down3)
    down4 = down()(double_conv4)

    double_conv5 = double_conv(output_channels=start_output_size*16, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(down4)
    up4 = up(start_output_size*8, kernel_size=kernel_size, activation=activation, kernel_initializer = kernel_initializer)(double_conv5)
    merge4 = merge(down4, up4, padding)

    double_conv6 = double_conv(output_channels=start_output_size*8, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(merge4)
    up3 = up(start_output_size*4, kernel_size=kernel_size, activation=activation, kernel_initializer = kernel_initializer)(double_conv6)
    merge3 = merge(down3, up3, padding)

    double_conv7 = double_conv(output_channels=start_output_size*4, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(merge3)
    up2 = up(start_output_size*2, kernel_size=kernel_size, activation=activation, kernel_initializer = kernel_initializer)(double_conv7)
    merge2 = merge(down2, up2, padding)

    double_conv8 = double_conv(output_channels=start_output_size*2, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(merge2)
    up1 = up(start_output_size, kernel_size=kernel_size, activation=activation, kernel_initializer = kernel_initializer)(double_conv8)
    merge1 = merge(down1, up1, padding)

    double_conv9 = double_conv(output_channels=start_output_size*2, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, dropout=dropout, activation=activation, kernel_initializer = kernel_initializer)(merge1)
    rescale = Conv2D(1, 1, activation = 'sigmoid')(double_conv9)

    model = Model(inputs = inputs, outputs = rescale)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['acc',f1_m,precision_m, recall_m])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model