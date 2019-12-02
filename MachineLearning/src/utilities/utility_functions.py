# -*- coding: utf-8 -*-
"""functions for feature extraction"""
import csv
import mahotas
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog

from PIL import Image
from skimage.filters import prewitt_h,prewitt_v
from skimage.filters import sobel_h,sobel_v
from scipy import ndimage, misc
from skimage.color import rgb2gray
from helper_functions import *

def calculate_f1_score(actual, predictions):
    
    total_predicted_positives = predictions.sum()
    total_actual_positives = actual.sum()
    true_positives = (actual*predictions).sum()
 
    precision = true_positives/total_predicted_positives
    recall = true_positives/total_actual_positives
    
    f1_score = 2 * ( precision * recall )/( precision + recall )
    return f1_score
   

#DEFINITION OF FEATURES
#Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    #shape 1x3 + 1x3 = 1x6
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image, used at the end for visual comparison
def extract_img_features(filename, patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X1 = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))]) # dim 6
    X2 = np.asarray([ laplace_gaussian_edge_detection(img_patches[i]) for i in range(len(img_patches))]) # dim 32
    X3 = np.asarray([ horizontal_and_vertical_edge_detection(img_patches[i]) for i in range(len(img_patches))]) # dim 32
    X4 = np.asarray([ get_grey_features(get_gray_mask(img_patches[i])) for i in range(len(img_patches))]) # dim 32
    X5 = np.asarray([ threshold_eroded_img(img_patches[i]) for i in range(len(img_patches))]) # dim 32
    X6 = np.asarray([ fd_hu_moments(img_patches[i]) for i in range(len(img_patches))]) # dim 7
    X7 = np.asarray([ fd_haralick(img_patches[i]) for i in range(len(img_patches))]) # dim 13
    X8 = np.asarray([ hog_features(img_patches[i]).ravel() for i in range(len(img_patches))]) # dim 16 * 16
    print("stop")
    X = np.concatenate((X1,X2,X3,X4,X6,X7,X8),axis=1) # dim 402
    X = feature_interaction(X) # dim = dim**2
    X = add_offset(X) # dim = dim + 1
    print(X.shape)
    return X

def fd_hu_moments(image):
    image = img_float_to_uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):    # convert the image to grayscale
    image = img_float_to_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def hog_features(image):
    #histogram of oriented gradients
    grey_bombus = rgb2gray(image)
    hog_features, hog_image = hog(grey_bombus,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(5,5))
        
    #plt.imshow(hog_image, cmap=cm.gray)
    return hog_image
    
def threshold_eroded_img(img):
    img = rgb2gray(img)
    #plt.imshow(img)
    #plt.show()
    img = img_float_to_uint8(img)
    ret,thresh1 = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
    #plt.imshow(thresh1)
    #plt.show()
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 1)
    dilated = cv2.dilate(erosion,kernel,iterations = 1)
    #plt.imshow(dilated)
    #plt.show()
    
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1300  

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    #plt.imshow(img2)
    #plt.show()
        
    feat_vert = np.sum(erosion, axis=1) / 255 / erosion.shape[1]
    feat_hori = np.sum(erosion, axis=0) / 255 / erosion.shape[0]
    
    feat_vert[feat_vert < 0.35] = 0
    feat_hori[feat_hori < 0.35] = 0
    #plt.bar(range(erosion.shape[0]), feat_hori)
    #plt.bar(range(erosion.shape[0]), feat_vert)
    #plt.show()
    return np.hstack((feat_vert,feat_hori))

def min_and_max_features(img):
    """
    returns the min and max rgb value for each channel
    """
    feat_max_rgb = np.max(img, axis=(0,1))
    feat_min_rgb = np.min(img, axis=(0,1))
    feat = np.append(feat_max_rgb, feat_min_rgb)
    return feat


def horizontal_and_vertical_edge_detection(image1):
    """
    applies a the sobel vertical and horizontal edge detection filter and then takes the mean for each channel
    dim = 32
    """
    image1 = rgb2gray(image1)
    
    edges_horizontal = sobel_h(image1)
    #calculating vertical edges using sobel kernel
    edges_vertical = sobel_v(image1)
    
    #ajouter np.sum(abs(sobel_x) + abs(sobel_y))
    feat_road_vert = np.sum(edges_horizontal,axis=0)/edges_horizontal.shape[0]/255 # dim 16
    feat_road_hori = np.sum(edges_vertical,axis=1)/edges_vertical.shape[1]/255 # dim 16
    
    #peut etre ajouter moyenne
    #plt.imshow(edges_vertical)
    #plt.show()
    
    return np.hstack((feat_road_vert, feat_road_hori))

def laplace_gaussian_edge_detection(image1):
    """
    applies a basic vertical and horizontal edge detection filter
    return: sum of pixel intensity in edge image over each column and each row in each channel,
            and mean and variance for each channel
            dim = (16 + 16 ) = 32
    """
    image1 = rgb2gray(image1)

    edges_image = ndimage.gaussian_laplace(image1, sigma=1.5)
    feat_road_vert = np.sum(edges_image,axis=0)/edges_image.shape[0]/255 # dim 16
    feat_road_hori = np.sum(edges_image,axis=1)/edges_image.shape[1]/255 # dim 16
    #feat_road_vert = feat_road_vert.reshape(1,feat_road_vert.shape[0])

    #plt.imshow(edges_image)
    #plt.show()

    return np.hstack((feat_road_vert,feat_road_hori))

#rem important to condider to 2 formats : 0-1 and 0-255
def get_gray_mask(img):
    img = img_float_to_uint8(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([15,20,65])
    upper = np.array([179, 50, 255])
    mask_gray = cv2.inRange(hsv, lower, upper)
    img_res = cv2.bitwise_and(img, img, mask = mask_gray)
    #transform into hue (hsv)
    
    #plt.imshow(mask_gray)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    #plt.imshow(gt_imgs[0]) #est faux, mais juste pour visualisation
    #plt.show()
    
    return mask_gray

def get_grey_features(mask):
    """
    returns the proportion of gey/black pixels in each line (16) and in each column (16)
    # dim = 16 + 16 = 32
    """
    feat_road_vert = np.sum(mask,axis=0)/mask.shape[0]/255
    feat_road_hori = np.sum(mask,axis=1)/mask.shape[1]/255
    feat_road_vert = feat_road_vert.reshape(1,feat_road_vert.shape[0])
    feat_road_hori = feat_road_hori.reshape(1,feat_road_hori.shape[0])
    # a voir si pas hozintal/verical
    # trouver continuité de pixels gris
    # faire somme selon differents angles
    #print(feat_road_vert.shape,feat_road_vert.shape)
    
    return np.append(feat_road_vert,feat_road_hori)