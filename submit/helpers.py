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
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.preprocessing.image as kerasimg
from keras import backend as K
from skimage.util.shape import view_as_windows
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

def get_patches_from_img(img, patch_size = 80, stride = 80, binary = True):
    """ If binary: takes input of shape (size_x, size_y)
        If not binary: takes input of shape (size_x, size_y,3) """
    if(binary):
      patch = view_as_windows(img, (patch_size,patch_size), step=stride)
    else:
      patch = view_as_windows(img, (patch_size,patch_size, 3), step=stride)
    return patch

def reconsrtuct_img_from_patches(patch, output_image_size=(400,400), patch_size = 80, stride = 80, mode = 'max', binary= False):
  """ If binary: takes input of shape (nbpatches_x, nbpatches_y, patch_size, patch_size)
        If not binary: takes input of shape (nbpatches_x, nbpatches_y, patch_size, patch_size, 3) """
  reconstructed = np.zeros(output_image_size)
  normalize_count = np.zeros(output_image_size)
  ones_patch = np.ones((patch_size,patch_size))

  if(not binary):
      ones_patch = np.stack((ones_patch,)*3, axis=-1)

  for i in range(patch.shape[0]):
    for j in range(patch.shape[1]):
      normalize_count[i * stride:i * stride + patch_size, j* stride:j* stride + patch_size] = \
                normalize_count[i * stride:i * stride + patch_size, j* stride:j* stride + patch_size] + ones_patch
      if(mode == 'max'):
          reconstructed[i * stride:i * stride + patch_size, j* stride:j* stride + patch_size] = \
                  np.maximum(reconstructed[i * stride:i * stride + patch_size, j* stride:j* stride + patch_size], patch[i,j])
      else:
          reconstructed[i * stride:i * stride + patch_size, j* stride:j* stride + patch_size] = \
                reconstructed[i * stride:i * stride + patch_size, j* stride:j* stride + patch_size] + patch[i,j]

  reconstructed = np.divide(reconstructed, normalize_count)

  if(binary):
    reconstructed[reconstructed >= 0.3] = 1
    reconstructed[reconstructed < 0.3] = 0
  return reconstructed


def recall_m(y_true, y_pred):
    """Computes the recall of the model. Source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    Args:
        y_true: true expected result
        y_pred: result from the network

    Returns:
        the recall of the model
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """Computes the precision of the model. Source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    Args:
        y_true: true expected result
        y_pred: result from the network

    Returns:
        the precision of the model
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """Computes the f1-score of the model. Source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    Args:
        y_true: true expected result
        y_pred: result from the network

    Returns:
        the f1-score of the model
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def nnztr_m(y_true, y_pred):
    """Computes the sum of all probabilities predicted by the network

    Args:
        y_true: true expected result
        y_pred: result from the network

    Returns:
        the sum of all probabilities predicted by the network
    """
    return (K.sum(y_pred))

def nnzte_m(y_true, y_pred):
    """Computes the sum of road pixels

    Args:
        y_true: true expected result
        y_pred: result from the network

    Returns:
        the sum of road pixels
    """
    return (K.sum(y_true))

def plot_analyze(data_for_graph):
    """Plots three graphs comparing train and loss: accuracy, loss, validation

        Args:
            data_for_graph: historic of training
            
    """
    # Plot training & validation accuracy values
    plt.plot(data_for_graph.history['acc'])
    plt.plot(data_for_graph.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(data_for_graph.history['loss'])
    plt.plot(data_for_graph.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation f1 values
    plt.plot(data_for_graph.history['f1_m'])
    plt.plot(data_for_graph.history['val_f1_m'])
    plt.title('Model F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


