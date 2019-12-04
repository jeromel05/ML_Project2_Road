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
from database import *
from network import *
import re
import glob

torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark=True

def find_mean_std(test_images):
    """Finds mean and std of images (UNUSED and probably false too don't use) """
      # find mean
    cnt = 0
    fst_moment = np.empty(3)
    snd_moment = np.empty(3)
    
    for i in range(len(test_images)):
        
        img = test_images[i]
        
        img = img.convert('RGB')
        
        img = np.array(img)
        img = np.moveaxis(img, 2, 0).astype(np.float64)
        c, h, w = img.shape
        nb_pixels = h * w       
        sum_ = np.sum(img, axis=(1,2))
        sum_of_square = np.sum(img ** 2, axis=(1,2))
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
    mean = fst_moment
    std = np.sqrt(snd_moment - fst_moment ** 2)

    return mean/255 , std/255


def get_padding(kernel_size = 3, dilation = 1):
  """ Get the required padding to be applied in order to extend the image for convolution.

    Args:
      kernel_size: the size of the applied kernel. Should be odd!
      dilation: the dilation factor applied to the kernel. Must be int!
  """
  return (kernel_size-1)//2 * dilation


def save_all_results(net, prefix, path_to_results, threshold=0.5,compare=False, patch=True ):
  """ Saves all results of the net on the test set in the drive

    Args:
      net: net you want to create the images with
      prefix: the prefix to the google colab 
      path_to_results: where you want to save the results in google drive
      threshold: if you BCEWithLogits loss use 0 otherwise use 0.5
      compare: if you want to save both the original image and the result next to each other or only the result if False
      patch: if you want to see patches or a grayscale representation of probabilities 

  """
  net.eval()
  with torch.no_grad():
    satelite_images_path = prefix + 'test_set_images'
    image_names = glob.glob(satelite_images_path + '/*/*.png')
    test_images = list(map(Image.open, image_names))
    transformX = transforms.Compose([
    transforms.ToTensor(), # transform to range 0 1
    ])

    for i, image_test in enumerate(test_images):

      image = transforms.Resize((400,400))(image_test)
      image_batch = transformX(image)
      image_batch = torch.from_numpy(np.array(image_batch)).unsqueeze(0).cuda()
      output = net(image_batch)
      net_result = nn.Sigmoid()(output) if threshold == 0 else output
      net_result = net_result[0].clone().detach().squeeze().cpu().numpy()
      net_result = transform_to_patch_format(net_result) if patch else net_result # do we want to see patches or a grayscale representation of probabilities
      net_result = (net_result*255).astype("uint8")
      net_result = net_result.reshape((400,400))
      net_result = convert_1_to_3_channels(net_result)
      net_result = Image.fromarray(net_result).resize((608,608))
      net_result = np.array(net_result)
      if compare:
        net_result = Image.fromarray(np.hstack([image_test, net_result]))
      else:    
        net_result = Image.fromarray(net_result)

      net_result.save(path_to_results+"test_image_" + str(int(re.search(r"\d+", image_names[i]).group(0))) + ".png", "PNG")


def mask_to_submission_strings(image, img_number):
    """Reads a single image and outputs the strings that should go into the submission file

    Args:
        image: one image to convert to string format
        img_number: the corresponding number in the test set

    """
    patch_size = 16
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[0], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(prefix, submission_filename, images, image_names):
	"""Converts images into a submission file

    Args:
        prefix: the prefix to the google colab 
        submission_filename: the name of the submission file
        images: all images you want to convert
        image_names: all images name in the same order than their corresponding images you want to convert

  """
	with open(prefix + 'results/' +submission_filename, 'w') as f:
		f.write('id,prediction\n')
		# order images
		image_in_order = np.zeros(np.array(images).shape)
		for i,name in enumerate(image_names):  
			image_nb = int(re.search(r"\d+", name).group(0))
			image_in_order[image_nb - 1][:][:] = images[i]

		for i in range(image_in_order.shape[0]):  
			image = image_in_order[i][:][:]
			f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image, i+1))

def get_submission(net, prefix, submission_filename, threshold=0.5):
  """Converts test set into a submission file in the results google drive folder

    Args:
        net: net you want to create the submission with
        prefix: the prefix to the google colab 
        submission_filename: the name of the submission file
        threshold: if you BCEWithLogits loss use 0 otherwise use 0.5

  """
  results = []
  net.eval()
  with torch.no_grad():

    # find all file names
    satelite_images_path = prefix + 'test_set_images'
    image_names = glob.glob(satelite_images_path + '/*/*.png')

    # get all images
    test_images = list(map(Image.open, image_names))
    transformX = transforms.Compose([
      transforms.ToTensor(), # transform to range 0 1
    ])

    for i, image in enumerate(test_images):

      # images are 608*608 so we need to resize to fit network
      image = transforms.Resize((400,400))(image)
      image_batch = transformX(image)
      image_batch = torch.from_numpy(np.array(image_batch)).unsqueeze(0).cuda()
      output = net(image_batch)
      net_result = output[0].clone().detach().squeeze().cpu().numpy() > threshold 
      net_result = Image.fromarray(net_result).resize((608,608))   
      results.append(np.array(net_result))
    
    masks_to_submission(prefix, submission_filename, results, image_names)
      

def see_result_on_test_set(net, prefix, compare=False, threshold=0.5 ):
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
    
    net.eval()
    with torch.no_grad():
      satelite_images_path = prefix + 'test_set_images'
      test_images = list(map(Image.open, glob.glob(satelite_images_path + '/*/*.png')))
      transformX = transforms.Compose([
        transforms.ToTensor(), # transform to range 0 1
      ])

      image = test_images[np.random.randint(len(test_images))]
      
      image = transforms.Resize((400,400))(image)
      image_batch = transformX(image)
      image_batch = torch.from_numpy(np.array(image_batch)).unsqueeze(0).cuda()
      output = net(image_batch)
      net_result = output[0].clone().detach().squeeze().cpu().numpy() >threshold
      net_result = transform_to_patch_format(net_result)
      net_result = net_result.astype("uint8")   
      net_result = net_result.reshape((400,400))*255
      net_result = convert_1_to_3_channels(net_result)
      

      if compare:
          net_result = Image.fromarray(np.hstack([image, net_result]))
      else:    
          net_result = Image.fromarray(net_result)
      return net_result

def calculate_f1_score(actual, predictions):
    """Calculates the f1_score between the predictions and the actual data

    Args:
        actual: Actual results of the road segmentation 400*400 boolean matrix 
        predictions: The raw output of our neural network 400*400 float matrix 

    Returns:
        The calculated f1_score: 2 * ( precision * recall )/( precision + recall )

    """
    
    total_predicted_positives = predictions.sum()
    total_actual_positives = actual.sum()
    true_positives = (actual*predictions).sum()
    
    precision = true_positives/total_predicted_positives
    recall = true_positives/total_actual_positives
    
    f1_score = 2 * ( precision * recall )/( precision + recall )
    
    return f1_score
    

def convert_1_to_3_channels(image):
    """Converts a 1 channel image to a 3 channel image by duplicating the channel three times

    Args:
        image: image with only one channel

    Returns:
        The image with the 3 duplicated channels

    """
    stacked_img = np.stack((image,)*3, axis=-1)
    return stacked_img    
    


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

def tensor_to_PIL(tensor):
	"""Transforms tensor to PIL image. This multiplies the tensor by 255 and converts to RGB"""
	return transforms.ToPILImage()(tensor).convert("RGB")

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

# eps to remove most nan values because of divide by 0

def f1(actual, prediction):
    """Gives the f1 score of the network results.

    Args:
        actual: the groundtruth of the input 
        prediction: the result the network obtained. Should be a binary image

    Returns:
        The f1 score for this prediction

    """    
    eps=1e-10
    tp = torch.sum(actual*prediction)
    total_predicted_positives = torch.sum(prediction)
    total_real_positives = torch.sum(actual)

    p = tp.float() / (total_predicted_positives.float())
    r = tp.float() / (total_real_positives.float())

    f1 = 2*p*r / (p+r+eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

    return f1

def f1_loss(actual, prediction):
    """Implements f1_loss for pytorch. As long as only pytorch functions are used then the pytorch implements backpropagation by itself.

    Args:
        actual: the groundtruth of the input 
        prediction: the result the network obtained. Should be an image with probabilities for the roads

    Returns:
        The f1 loss for this prediction

    """
    eps=1e-10
    tp = torch.sum(actual*prediction)
    total_predicted_positives = torch.sum(prediction)
    total_real_positives = torch.sum(actual)

    p = tp.float() / (total_predicted_positives.float())
    r = tp.float() / (total_real_positives.float())

    f1 = 2*p*r / (p+r+eps)
    # removes nan values due to division to close to 0
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    
    return 1 - f1

def save_if_best_model(net, last_best_f1_test, contender_f1_test, contender_f1_train, path_to_models, min_train_f1 = 0.80, min_test_f1 = 0.80):
  """Saves model only if specific conditions are obtained:
      train f1 and test f1 needs to be atleast a minimum value and also beat the previous f1 test.

    Args:
        net: iteration of the network you want to save
        last_best_f1_test: the best f1 score you got for this model
        contender_f1_test: the f1 score of the test set you obtained on this epoch
        contender_f1_train: the f1 score of the train set you obtained on this epoch
        path_to_models: path in the drive where you 
        min_train_f1: constraint on the minimum value you want the network to score on the training set
        min_test_f1: constraint on the minimum value you want the network to score on the training set

    """
  if contender_f1_train > min_train_f1 and contender_f1_test > last_best_f1_test and contender_f1_test > min_test_f1:
    # save net 
    torch.save(net.state_dict(), path_to_models+'/best_model.pt')
    # if model beats the last f1 test then return the new best test f1
    return contender_f1_test
  # if model does not beat the last one return the last f1  
  return last_best_f1_test

def see_result(loader, net, threshold=0.5):
	"""Computes the result of the network on one random input image and compares it to the actual required result
	
	Args:
        loader: pytorch loader to test the image of
        net: network you want to test
		    threshold: if you BCEWithLogits loss use 0 otherwise use 0.5

    Returns:
        The image comparing all
	"""
	images, groundtruth = next(iter(loader)) 
	outputs = net(images)

	image = images[0].cpu().numpy()
	groundtruth = groundtruth[0].cpu().numpy()
	net_result = outputs[0].detach().cpu().numpy()
	net_result = net_result > 0.5
	image = np.moveaxis(image, 0, 2)

	image = image*255
	# image = image*std + mean
	image = image
	groundtruth = np.moveaxis(groundtruth, 0, 2)
	net_result = np.moveaxis(net_result, 0, 2)
	net_result = transform_to_patch_format(net_result)
	image = image.astype("uint8")
	groundtruth = groundtruth.astype("uint8")
	net_result = net_result.astype("uint8")

	groundtruth = groundtruth.reshape((400,400))*255
	net_result = net_result.reshape((400,400))*255
	# print(net_result)

	groundtruth = convert_1_to_3_channels(groundtruth)
	net_result = convert_1_to_3_channels(net_result)

	compare = np.hstack([image, groundtruth, net_result])
	return Image.fromarray(compare)

from scipy.ndimage import rotate

def multi_decision(net, np_image, threshold=0.5):
  """Computes a decision of the network of one image with its four rotations and then outputs the four decisions in the correct orientation.
  
    Args:
        net: the network you use for the prediction
        np_image: image you want to test
        threshold: if you BCEWithLogits loss use 0 otherwise use 0.5

    Returns:
        four decisions computed from the four rotations of the image by the network in the correct orientation
  """
  np_image = np.moveaxis(np_image, 0, 2)
  rotations = []
  for i in range(4):
    rotated_image = rotate(np_image, i*90)#np.rot90(np_image, k=i+1)
    rotated_image = np.moveaxis(rotated_image, 2, 0)
    rotations.append(rotated_image)

  for i in range(4):
    rotations[i] = torch.from_numpy(rotations[i].copy())

  for i in range(4):
    rotations[i] = (net(rotations[i].unsqueeze(0).cuda()).detach().squeeze().cpu() > threshold).float()
    
  for i in range(4):
    rotations[i] = np.array(TF.rotate(transforms.ToPILImage()(rotations[i]).convert("L"), -i*90))>128

  return rotations

def append_channel(image_tensor, channel_tensor):
  """Appends a channel to an image
    Args:
        image_tensor: tensor of the image 3 400 400
        channel: the tensor of the channel you want to append 1 400 400

    Returns:
        the image with the appended channel 4 400 400
  """
  # now we can stack
  return torch.cat([image_tensor, channel_tensor], dim=0)  # 4 400 400

def decide(net, np_image, decider,threshold=0.5):
  """Computes a decision of the network of one image with its four rotations and then outputs the four decisions in the correct orientation.
  
    Args:
        net: the network you use for the prediction
        np_image: image you want to test
        decider: decide logic function needs to return an image and input a list of decisions
        threshold: if you BCEWithLogits loss use 0 otherwise use 0.5

    Returns:
        decision for this image
  """
  list_of_decisions = multi_decision(net, np_image, threshold)
  return decider(list_of_decisions)


def decide_simple(list_of_decisions):
  """Return the decision of the reference image (without rotation)
    Args:
      list_of_decisions: all the decisions the network has provided
  """

  return list_of_decisions[0]

def decide_or_logic(list_of_decisions):
  """Decides with a list of decisions for each pixel vote what the pixel should be
    Args:
      list_of_decisions: all the decisions the network has provided
  """
  decision = np.zeros(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision + vote

  return decision >= 1

def decide_and_logic(list_of_decisions):
  """Decides with a list of decisions for each pixel vote what the pixel should be
    Args:
      list_of_decisions: all the decisions the network has provided
  """
  decision = np.ones(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision * vote

  return decision == 1

def decide_majority_logic(list_of_decisions):
  """Decides with a list of decisions for each pixel vote what the pixel should be
    Args:
      list_of_decisions: all the decisions the network has provided
  """
  decision = np.zeros(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision + vote

  return decision >= len(list_of_decisions)//2 + 1
  

  


            