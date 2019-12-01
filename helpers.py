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

torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark=True

import glob


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

def calculate_all_results(net, compare=False):
    """ Calculates one random test images result and compares it to the actual image if required  """
    
    net.eval()
    with torch.no_grad():
      satelite_images_path = prefix + 'test_set_images'
      test_images = list(map(Image.open, glob.glob(satelite_images_path + '/*/*.png')))
      # mean, std = find_mean_std(test_images)
      # print(mean, std)
      transformX = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
      ])

      image = test_images[np.random.randint(len(test_images))]
      
      image = transforms.Resize((400,400))(image)
      image_batch = transformX(image)
      image_batch = torch.from_numpy(np.array(image_batch)).unsqueeze(0).cuda()
      output = net(image_batch)
      net_result = output[0].clone().detach().squeeze().cpu().numpy() > 0.5
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
   """Decides if one patch should be considered a road or background"""
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def transform_to_patch_format(mask):
    """Reads a single mask and converts image to patch image with all patches to their corresponding value"""
    im = mask
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            # is a road of not?
            label = patch_to_label(patch)
            # convert whole patch to be the same as label
            im[i:i + patch_size, j:j + patch_size] = np.ones_like(patch) if label else np.zeros_like(patch)
    return im

# eps to remove most nan values because of divide by 0

def f1(actual, prediction):
    """Gives the f1 score of the network results. """
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
    """Implements f1_loss for pytorch. All long as only pytorch functions are used then the pytorch implements backpropagation by itself."""
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

def save_if_best_model(net, last_best_f1_test, contender_f1_test, contender_f1_train, min_train_f1 = 0.80, last_best_f1_test = 0.80):
  """Saves model only if specific conditions are obtained:
      train f1 and test f1 needs to be atleast a minimum value and also beat the previous f1 test.
  """
  if contender_f1_train > min_train_f1 and contender_f1_test > last_best_f1_test:
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

def multi_decision(net, np_image, threshold):
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

def decide_or_logic(list_of_decisions):
  decision = np.zeros(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision + vote

  return decision >= 1

def decide_and_logic(list_of_decisions):
  decision = np.ones(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision * vote

  return decision == 1

def decide_majority_logic(list_of_decisions):
  decision = np.zeros(list_of_decisions[0].shape)
  for vote in list_of_decisions:
    decision = decision + vote

  return decision >= len(list_of_decisions)//2 + 1
  

  


            