# -*- coding: utf-8 -*-
"""some helper functions for project 2."""
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from skimage.filters import prewitt_h,prewitt_v
from skimage.filters import sobel_h,sobel_v
from scipy import ndimage, misc


# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def calculate_f1_score(actual, predictions):
  
    total_predicted_positives = predictions.sum()
    total_actual_positives = predictions.sum()
    true_positives = (actual*predictions).sum()
    
    precision = true_positives/total_predicted_positives
    recall = true_positives/total_actual_positives
    
    f1_score = 2 * ( precision * recall )/( precision + recall )
    
    return f1_score
    

#DEFINITION OF FEATURES
# Extract 6-dimensional features consisting of average RGB color as well as variance
#inutile pck on a normalisé avant
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
def extract_img_features(filename,patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    #X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    X1 = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))]) # dim 6
    X2 = np.asarray([ laplace_gaussian_edge_detection(img_patches[i]) for i in range(len(img_patches))]) # dim 3
    X3 = np.asarray([ horizontal_and_vertical_edge_detection(img_patches[i]) for i in range(len(img_patches))]) # dim 6
    X4 = np.asarray([ get_grey_features(get_gray_mask(img_patches[i])) for i in range(len(img_patches))]) # dim 2
    X = np.concatenate((X1,X2,X3,X4),axis=1) # dim 13
    X = feature_expansion(X,5) # dim = dim * degree
    X = add_offset(X) # dim = dim + 1
    return X

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
    applies a basic vertical and horizontal edge detection filter and then takes the mean for each channel
    """
    nb_channels = image1.shape[2]
    feat_edge_vert_avg = np.zeros((1,nb_channels))
    feat_edge_hori_avg = np.zeros((1,nb_channels))
    
    for i in range(nb_channels):
        edges_prewitt_horizontal = sobel_h(image1[:,:,i])
        #calculating vertical edges using sobel kernel
        edges_prewitt_vertical = sobel_v(image1[:,:,i])
        
        feat_edge_vert_avg[0,i] = np.mean(edges_prewitt_vertical)
        feat_edge_hori_avg[0,i] = np.mean(edges_prewitt_horizontal)
        
    return np.append(feat_edge_vert_avg, feat_edge_hori_avg)

def laplace_gaussian_edge_detection(image1):
    """
    applies a basic vertical and horizontal edge detection filter and then takes the mean for each channel
    """
    nb_channels = image1.shape[2]
    feat_edges_laplace_gaussian = np.zeros((2*nb_channels))
    sum_edges_col = np.array([])
    sum_edges_row = np.array([])
    
    for i in range(nb_channels):
        edges_image = ndimage.gaussian_laplace(image1[:,:,i], sigma=1.5)
        feat_road_vert = np.sum(edges_image,axis=0)/edges_image.shape[0]/255
        feat_road_hori = np.sum(edges_image,axis=1)/edges_image.shape[1]/255
        feat_road_vert = feat_road_vert.reshape(1,feat_road_vert.shape[0])
        feat_road_hori = feat_road_hori.reshape(1,feat_road_hori.shape[0])
        sum_edges_col = np.append(sum_edges_col, feat_road_vert)
        sum_edges_row = np.append(sum_edges_row, feat_road_hori)
        feat_edges_laplace_gaussian[i] = np.mean(edges_image)
        feat_edges_laplace_gaussian[i+nb_channels] = np.var(edges_image)
        #print(edges_image)
        #plt.imshow(edges_image)
        #plt.show()
        #print(np.max(edges_image),np.min(edges_image),np.mean(edges_image))
        #print(sum_edges_col.shape)
    feat_edges_laplace_gaussian = feat_edges_laplace_gaussian.reshape(1,feat_edges_laplace_gaussian.shape[0])
    #return np.append((sum_edges_col.ravel(),sum_edges_row.ravel(),feat_edges_laplace_gaussian))
    return np.concatenate((sum_edges_col.ravel(),sum_edges_row.ravel(),np.squeeze(feat_edges_laplace_gaussian)),axis=0)

def feature_expansion(features,degree):
    """
    feature_expansion in the format X,X^2,..,X^degree (element-wise)
    """
    res = features
    iter1 = np.arange(2,degree+1,1)
    for i in iter1:
        res = np.concatenate((res,features**(i)),axis=1)
        
    return res

def add_offset(features):
    """
    adds a first column of ones to the features X
    """
    offset = np.ones((features.shape[0],1))
    return np.concatenate((offset,features),axis=1)

#rem important to condider to 2 formats : 0-1 and 0-255
import cv2

def get_gray_mask(img):
    img = img_float_to_uint8(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([15,20,65])
    upper = np.array([179, 50, 255])
    mask_gray = cv2.inRange(hsv, lower, upper)
    img_res = cv2.bitwise_and(img, img, mask = mask_gray)
    #plt.imshow(mask_gray)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    #plt.imshow(gt_imgs[0]) #est faux, mais juste pour visualisation
    #plt.show()
    
    return mask_gray

def get_grey_features(mask):
    """
    returns the proportion of gey/black pixels in each line (400) and in each column (400)
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