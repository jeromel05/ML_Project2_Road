# -*- coding: utf-8 -*-
"""functions for plotting"""
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def hist_of_features(X):
    """
    performs a histogram of the distribution of all the parameters in the dataset 
    parameters:
        X: the dataset to plot
    """
    n = X.shape[1]
    fig, ax = plt.subplots(4,5)

    for i in range(4):
        for j in range(5):
            ax[i,j].hist(X[:,i+j], bins = 30)
            ax[i,j].set_title(i+j,fontsize=40)
        
    fig.set_figheight(150)
    fig.set_figwidth(150)
    plt.suptitle("Distribution of all the features",fontsize=150)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()
    
def print_feature_stats(X,Y):
    # Print feature statistics
    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)))  #TODO: fix, length(unique(Y)) 

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')
    
    