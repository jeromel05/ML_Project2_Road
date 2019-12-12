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
from networks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage import rotate


def to_single_batch(numpy_array):
    """Transform numpy array to corresponding single batch

    Args:
        numpy_array: numpy array 

    Returns:
        a numpy array in batch format

    """
    return np.expand_dims(numpy_array, axis=0)

def from_single_batch(batch_numpy_array):
    """Transform single batch to corresponding numpy array  

    Args:
        batch_numpy_array: numpy array of a single batch

    Returns:
        corresponding numpy array

    """
    return batch_numpy_array[0]

def recall_m(y_true, y_pred):
    """Returns the recall considering a prediction and a groundtruth
  
    Args:
        y_true: the true image. Should be either 1s or 0s
        y_pred: the image predicted by the network. Should be either 1s or 0s

    Returns:
        the recall of the network

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """Returns the precision considering a prediction and a groundtruth
  
    Args:
        y_true: the true image. Should be either 1s or 0s
        y_pred: the image predicted by the network. Should be either 1s or 0s

    Returns:
        the precision of the network

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """Returns the f1-score considering a prediction and a groundtruth
  
    Args:
        y_true: the true image. Should be either 1s or 0s
        y_pred: the image predicted by the network. Should be either 1s or 0s

    Returns:
        the f1-score of the network

    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def multi_decision(net, np_image, threshold=0.5):
  """Computes a decision of the network of one image with its four rotations and then outputs the four decisions in the correct orientation.
  
    Args:
        net: the network you use for the prediction
        np_image: image you want to test
        threshold: if you BCEWithLogits loss use 0 otherwise use 0.5

    Returns:
        four decisions computed from the four rotations of the image by the network in the correct orientation
  """
  rotations = []
  for i in range(4):
    rotated_image = rotate(np_image, i*90)
    rotations.append(rotated_image)

  for i in range(4):
    inputs = to_single_batch(rotations[i])
    outputs = net.predict(inputs)
    rotations[i] = outputs > threshold
    
  for i in range(4):
    rotations[i] = rotate(rotations[i], -i*90)

  return rotations

def decide_simple(list_of_decisions):
  """Return the decision of the reference image (without rotation)

    Args:
        list_of_decisions: all the decisions the network has provided

    Return:
        the final decision aka the prediction of the network

  """
  return list_of_decisions[0]

def decide_or_logic(list_of_decisions):
  """Decides with a list of decisions for each pixel vote what the pixel should be by deciding that any pixel found as road is one

    Args:
        list_of_decisions: all the decisions the network has provided

    Return:
        the final decision aka the prediction of the network
  """
  decision = np.zeros(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision + vote

  return decision >= 1

def decide_and_logic(list_of_decisions):
  """Decides with a list of decisions for each pixel vote what the pixel should be by deciding that  a pixel should be considered as a road by all networks

    Args:
        list_of_decisions: all the decisions the network has provided

    Return:
        the final decision aka the prediction of the network
  """
  decision = np.ones(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision * vote

  return decision == 1

def decide_majority_logic(list_of_decisions):
  """Decides with a list of decisions for each pixel vote what the pixel should be by deciding that a pixel is a road if a majority of the decisions think it is

    Args:
        list_of_decisions: all the decisions the network has provided

    Return:
        the final decision aka the prediction of the network
  """
  decision = np.zeros(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision + vote

  return decision >= len(list_of_decisions)//2 + 1

def decide(net, np_image, decider=decide_simple):
  """Computes a decision of the network of one image with its four rotations and then outputs the four decisions in the correct orientation.
  
    Args:
        net: the network you use for the prediction
        np_image: image you want to test
        decider: decide logic function needs to return an image and input a list of decisions
        threshold: if you BCEWithLogits loss use 0 otherwise use 0.5

    Returns:
        decision for this image
  """
  list_of_decisions = multi_decision(net, np_image)
  return decider(list_of_decisions)

def convert_1_to_3_channels(image):
    """Converts a 1 channel image to a 3 channel image by duplicating the channel three times

    Args:
        image: image with only one channel

    Returns:
        The image with the 3 duplicated channels

    """
    stacked_img = np.stack((image,)*3, axis=-1)
    return stacked_img

def save_all_results(net, prefix, path_to_results, compare=False, patch=True, net_size=(400,400) , decider=decide_simple):
    """ Saves all results of the net on the test set in the drive

        Args:
            net: net you want to create the images with
            prefix: the prefix to the google colab result locations
            path_to_results: where you want to save the results in google drive
            compare: if you want to save both the original image and the result next to each other or only the result if False
            patch: if you want to see patches or a grayscale representation of probabilities 
            net_size: the input size of the network
            decider: the decision strategy you choose

    """

    satelite_images_path = prefix + 'test_set_images'
    image_names = glob.glob(satelite_images_path + '/*/*.png')
    test_images = list(map(Image.open, image_names))

    for i, image_test in enumerate(test_images):

        image = transforms.Resize(net_size)(image_test)
        # make decision
        net_result = decide(net, np.array(image), decider)
        net_result = transform_to_patch_format(net_result) if patch else net_result # do we want to see patches or a grayscale representation of probabilities
        net_result = (net_result*255).astype("uint8")
        net_result = net_result.reshape(net_size)
        net_result = convert_1_to_3_channels(net_result)
        net_result = Image.fromarray(net_result).resize((608,608))
        net_result = np.array(net_result)
        if compare:
            net_result = Image.fromarray(np.hstack([image_test, net_result]))
        else:    
            net_result = Image.fromarray(net_result)

        net_result.save(path_to_results+"test_image_" + str(int(re.search(r"\d+", image_names[i]).group(0))) + ".png", "PNG")

def patch_to_label(patch, foreground_threshold = 0.25 ):
    """Decides if one patch should be considered a road or background

    Args:
        patch: one patch to predict. Can be probabilities or strictly 1 and 0 values
        foreground_threshold: threshold of which a patch's mean is decided to be a road or not

    Returns:
        The result patched with the threshold provided

    """  
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def transform_to_patch_format(mask, foreground_threshold = 0.25):
    """Reads a single mask and converts image to patch image with all patches to their corresponding value

    Args:
        mask: result of the network. Can be probabilities or strictly 1 and 0 values
        foreground_threshold: threshold of which a patch's mean is decided to be a road or not

    Returns:
        The result patched with the threshold provided

    """    
    im = mask
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            # is a road of not?
            label = patch_to_label(patch, foreground_threshold)
            # convert whole patch to be the same as label
            im[i:i + patch_size, j:j + patch_size] = np.ones_like(patch) if label else np.zeros_like(patch)
    return im

def see_result_on_test_set(net, prefix, compare=False, net_size=(400,400) ):
    """ Calculates one random test images result and compares it to the actual image if required

    Args:
        net: net you want to create the images with
        prefix: the prefix to the google colab 
        path_to_results: where you want to save the results in google drive
        threshold: if you BCEWithLogits loss use 0 otherwise use 0.5
        compare: if you want to save both the original image and the result next to each other or only the result if False

    Returns:
        the PIL image of either the result or the result with its base input

    """
    
    satelite_images_path = prefix + 'test_set_images'

    rdm_index = np.random.randint(len(test_images))
    image = Image.open(glob.glob(satelite_images_path + '/*/*.png')[rdm_index])

    image = image.resize(net_size)
    image_batch = to_single_batch(image)
    output = net.predict(image_batch)
    net_result = from_single_batch(output) > 0.5
    net_result = transform_to_patch_format(net_result)
    net_result = (net_result*255).astype("uint8")   
    net_result = net_result.reshape((400,400))
    net_result = convert_1_to_3_channels(net_result)

    if compare:
        net_result = Image.fromarray(np.hstack([image, net_result]))
    else:    
        net_result = Image.fromarray(net_result)

    return net_result

def see_result(loader, net, proba=False, net_size=(400,400)):
    """Computes the result of the network on one random input image and compares it to the actual required result

        Args:
            loader: pytorch loader to test the image of
            net: network you want to test
            proba: false if you want to see patches otherwise true to see the network probabilities per pixel 

        Returns:
            The image comparing all
    """
    rdm_index = np.random.randint(len(loader))
    images, groundtruth = loader[rdm_index] 
    outputs = net.predict(images)

     # gets the first image from the batch
    image = images[0]
    groundtruth = groundtruth[0]
    net_result = outputs[0]

    image = image*255

    if not proba:
        net_result = net_result > 0.5
        net_result = transform_to_patch_format(net_result)  

    image = image.astype("uint8")
    groundtruth = (groundtruth*255).astype("uint8")
    net_result = (net_result*255).astype("uint8")

    groundtruth = groundtruth.reshape(net_size)
    net_result = net_result.reshape(net_size)

    groundtruth = convert_1_to_3_channels(groundtruth)
    net_result = convert_1_to_3_channels(net_result)

    compare = np.hstack([image, groundtruth, net_result])

    return Image.fromarray(compare)

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
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(data_for_graph.history['loss'])
    plt.plot(data_for_graph.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation f1 values
    plt.plot(data_for_graph.history['f1_m'])
    plt.plot(data_for_graph.history['val_f1_m'])
    plt.title('F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

