import numpy as np
import h5py
import glob
from PIL import Image

path = 

images = map(Image.open, glob.glob('your/path/*.gif'))



train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

train_shape = (len(train_addrs), 224, 224, 3)
test_shape = (len(test_addrs), 224, 224, 3)