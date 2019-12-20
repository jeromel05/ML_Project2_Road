# Project 2: Road Segmentation

## Description of the project
In this project we use our implementation of the UNet to perform road segmentation on aerial satellite images.

## How to run the code

Notice that you should first download the data to be able to run the code, they are available at AICrowd at the EPFL ML Road Segmentation 2019 challenge, 
in the tab 'resources': [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation-2019/dataset_files), and save the downloaded files in a folder 'data' that should be put in the 'ML_Project2_Road' folder.  
Then you should run the file 'run.py' from 'ML_Project2_Road/scripts/src/'    
Our code adapts automatically to GPU or CPU, however the training of our neural network is a computationally intensive process. So it is strongly recommended to run it on a GPU with at least 18GB of RAM.   
The training process takes approximately 2 hours to complete on a GPU, so we stored the weights of the pre-trained network in a hdf5 file on Google Drive. So you can just skip the cell and load the model with the given weights and directly make a prediction.  
To toggle wether you want to use a pre-defined weights file dowloaded from Google Drive, you can add an argument to the main.
So if you run 
```python
python run.py True
```
The weights file will be used.  
If you do not want to use a weights file, just run it without any arguments.
```python
python run.py
```
The weights file can be accessed via two different ways:   
    - Log in the following Google account:   
    - Follow the shared  that gives you access to the google drive folder named Road Segmentation  
    The file is located at: Colab Notebooks/road_segamentation/models/saved_net/keras_patch128_new1.hdf5  

### Prerequisites
In order to be able to run the code, you should have Python3 installed, the installation of Anaconda is also recommended,
and the following libraries should be installed:

- numpy: to manipulate arrays
- h5py: to save files in this format
- glob
- Pillow: for image manipulation
- matplotlib: for plotting
- skimage: for image transformations
- keras: for simple high level implementations of neural networks
- sklearn: for train / validation splitting
- tensorflow version=1: the framework Keras is based on


## Folder plan 
You should put the downoaded files in a new folder 'data' in the folder 'ML_Project2_Road'. 
The folders are organized this way : 


- ML_Project2_Road
    - project1_description.pdf
    - data
       - models
       - results
       - training
       - test set images
    - scripts
       - src
       - utilities
       
         
The src folder contains the main python file "run.py" that yields a .csv file with our best predictions
You can find all the neural networks in the path : /ML_Project2_Road/scripts/utilities/network.py.  
The other files in the folder utilities are all the functions we used to modularize the code in the project.
The 'models' and 'results' are just empty folders you have to create in order to save files later.  
If a weights file is to be used, it has to be placed in the models folder.


## Guide to use the google colab
If you do not have acces to a GPU, we also have a repository in Google Colab where a GPU is accessible, all the data files are already uploaded
and the libraries are already pre-installed.  
To have the exact environment we used to train the network you can use our google colab code. You can either you the account provided above or follow these instructions:    
 
    - 1: follow the share   
    - 2: locate the previously shared folder "road segmentation project" in "Shared with me" on your google drive.  
    - 3 (optional): if you never have created a google colabotary file you need to:  
        - 3.1: head to https://colab.research.google.com/ and create a "new notebook in python 3"  
    - 4: head to your drive and locate the folder "road segmentation project" you just added.  
    - 5: add the "road segmentation project" folder to the folder "Colab Notebooks" of your drive (added in step 3.1 or already there if            you previously created a google colab)  
    - 6: open the "road segmentation project" folder and right click the "UNet.ipynb". Select "open with >" and choose "Google                     Colaboratory"  
    - 7: You can now follow the instructions in the google colab and run the code cell per cell or just run all cells at once to directly get the results in a .csv format  
 

