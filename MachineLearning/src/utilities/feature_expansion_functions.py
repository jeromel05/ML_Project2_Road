# -*- coding: utf-8 -*-
"""functions for feature expansion"""
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def feature_expansion(features,degree):
    """
    feature_expansion in the format X,X^2,..,X^degree (element-wise)
    """
    res = features
    iter1 = np.arange(2,degree+1,1)
    for i in iter1:
        res = np.concatenate((res,features**(i)),axis=1)
        
    return res

def feature_interaction(features):
    """
    calculates all the possible non-linear combinations of degree 1 of the features
    uses element wise product between two feature vectors
    """
    res = np.zeros((features.shape[0], features.shape[1]**2))
    print(res.shape)
    for i in range(np.around(features.shape[1]/2)):
        for j in arange(np.around(features.shape[1]/2), features.shape[1], 1):
            res[:,i+j] = features[:,i] * features[:,j]
       
    return res

def add_offset(features):
    """
    adds a first column of ones to the features X
    """
    offset = np.ones((features.shape[0],1))
    return np.concatenate((offset,features),axis=1)


