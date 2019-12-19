import os, sys
import re
import glob
import scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mask_to_submission as msk

from PIL import Image
from keras.optimizers import Adam
from keras.layers import Input, Dropout, Conv2D, UpSampling2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN

from keras import backend as K

import helpers as hlp


def double_conv2d_dropout(inputs, nb_out_channels, kernel_size = 3, activation_f = 'relu', padding = 'same', dilation_rate = (1,1), \
							kernel_initializer = 'glorot_uniform', dropout_rate = 0.2):
	"""
	Performs 2 convolutions and a droput in between
	"""
	conv1 = Conv2D(nb_out_channels, kernel_size, activation = activation_f, padding = padding, dilation_rate = dilation_rate, \
					kernel_initializer = kernel_initializer)(inputs)
	drop1 = Dropout(dropout_rate)(conv1)
	conv1 = Conv2D(nb_out_channels, kernel_size, activation = activation_f, padding = padding, dilation_rate=dilation_rate, \
					kernel_initializer = kernel_initializer)(drop1)
	return conv1
		
def up(inputs, conv_to_merge, nb_out_channels, activation_f = 'relu', padding = 'same', dilation_rate = (1,1), \
							kernel_initializer = 'glorot_uniform', dropout_rate = 0.2):
		"""
		Performs a convolution to reduce the number of channels by half, followed by a 2d upsampling to double the image size
		Merges the descending layers with the corresponding output layer
		Finally performs two convolutions with a driopout in between
		"""
		up1 = Conv2D(nb_out_channels, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_initializer)(UpSampling2D(size = (2,2))(inputs))
		merge1 = concatenate([conv_to_merge,up1], axis = 3)					
		conv1 = double_conv2d_dropout(merge1, nb_out_channels, activation_f = activation_f, padding = padding, \
							dilation_rate = dilation_rate, kernel_initializer = kernel_initializer, dropout_rate = dropout_rate)
		return conv1

def unet_5_layers(pretrained_weights = None, input_size = (128,128,3), nb_out_channels = 16,  activation_f = 'relu', dilation_rate = (1,1), padding = 'same', \
			kernel_init = 'glorot_uniform', verbose = 0):
    """Creates our implementation of UNet compiled with the following parameters:
        optimiser: Adam with default parameters
        loss: binary crossentropy
        metrics monitored: f1, accuracy, precision, recall
    Args:
        input_size: format of the input image
        nb_out_channels: number of output channels. Modifying this parameter increases the number of channel on each layer. 
					It corresponds to the initial (after the first convolution) and output channel (before the last convolution) size 
		activation_f: activation function to use
		dilation_rate: dilation parameter for 2d convolution
		padding: padding method
		kernel_init: kernel initialization method
		verbose: prints model summary if > 0
    Returns:
        UNet model
    """
    inputs = Input(input_size)

    conv1 = double_conv2d_dropout(inputs, nb_out_channels, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv2d_dropout(pool1, nb_out_channels*2, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv2d_dropout(pool2, nb_out_channels*4, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv2d_dropout(pool3, nb_out_channels*8, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = double_conv2d_dropout(pool3, nb_out_channels*16, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(nb_out_channels*32, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_init)(pool5)
    conv6 = Conv2D(nb_out_channels*32, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_init)(conv6)
    
    up6 = up(conv6, conv5, nb_out_channels*16, kernel_size = kernel_init, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)				
	
    up7 = up(up6, conv4, nb_out_channels*8, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)				
		
    up8 = up(up7, conv3, nb_out_channels*4, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
							
    up9 = up(up8, conv2, nb_out_channels*2, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)	
							
    up10 = up(up9, conv1, nb_out_channels, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)	
											
    conv10 = Conv2D(2, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_init)(up10)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = [hlp.f1_m, 'accuracy', hlp.precision_m, hlp.recall_m])
    
    if(verbose > 0):
	    model.summary()
		
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
    

def unet_4_layers(pretrained_weights = None, input_size = (128,128,3), nb_out_channels = 32,  activation_f = 'relu', dilation_rate = (1,1), padding = 'same', \
					kernel_init = 'glorot_uniform', verbose = 0):
    """Creates our implementation of UNet compiled with the following parameters:
        optimiser: Adam with default parameters
        loss: binary crossentropy
        metrics monitored: f1, accuracy, precision, recall

    Args:
        input_size: format of the input image
        nb_out_channels: number of output channels. Modifying this parameter increases the number of channel on each layer. 
					It corresponds to the initial (after the first convolution) and output channel (before the last convolution) size 
		activation_f: activation function to use
		dilation_rate: dilation parameter for 2d convolution
		padding: padding method
		kernel_init: kernel initialization method
		verbose: prints model summary if > 0
    Returns:
        UNet model
    """
    inputs = Input(input_size)

    conv1 = double_conv2d_dropout(inputs, nb_out_channels, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv2d_dropout(pool1, nb_out_channels*2, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv2d_dropout(pool2, nb_out_channels*4, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv2d_dropout(pool3, nb_out_channels*8, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(nb_out_channels*16, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_init)(pool4)
    conv5 = Conv2D(nb_out_channels*16, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_init)(conv5)
	
    up6 = up(conv5, conv4, nb_out_channels*8, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)				
	
    up7 = up(up6, conv3, nb_out_channels*4, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)
					
    up8 = up(up7, conv2, nb_out_channels*2, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)	

    up9 = up(up8, conv1, nb_out_channels, activation_f = activation_f, padding = padding, \
						dilation_rate = dilation_rate, kernel_initializer = kernel_init)	
											
    conv10 = Conv2D(2, 3, activation = activation_f, padding = padding, kernel_initializer = kernel_init)(up9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = [hlp.f1_m, 'accuracy', hlp.precision_m, hlp.recall_m])
    
    if(verbose > 0):
	    model.summary()
		
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
