import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from skimage.util.shape import view_as_windows
import os
import matplotlib.image as mpimg
import scipy
import masks_to_submission as msk

def make_patches(imgs, gt_imgs, patch_size1 = 128, stride1= 16):
	n = len(imgs)
	img_patches = [get_patches_from_img(imgs[i], patch_size = patch_size1, stride = stride1, binary = False) for i in range(n)]
	gt_patches = [get_patches_from_img(gt_imgs[i], patch_size = patch_size1, stride = stride1, binary = True) for i in range(n)]
	# Linearize list of patches
	img_patches = np.asarray([img_patches[i][j][k] for i in range(len(img_patches)) for j in range(len(img_patches[i])) for k in range(len(img_patches[i][j]))])
	gt_patches =  np.asarray([gt_patches[i][j][k] for i in range(len(gt_patches)) for j in range(len(gt_patches[i])) for k in range(len(gt_patches[i][j]))])

	#remove/add dimension to have format: (nb of patches, patch size, patch size, nb channels)
	img_patches = img_patches.squeeze(1)
	gt_patches = gt_patches.reshape((*gt_patches.shape, 1))
	
	return img_patches, gt_patches

def load_data(path_prefix):
	root_dir = prefix + "data/training/"

	image_dir = root_dir + "images/"
	files = os.listdir(image_dir)
	n = len(files) # Load all 100 images
	print("Loading " + str(n) + " images")
	imgs = [mpimg.imread(image_dir + files[i]) for i in range(n)]

	gt_dir = root_dir + "groundtruth/"
	print("Loading " + str(n) + " groundtruth images")
	for i in range(n): print(files[i]) 
	gt_imgs = [mpimg.imread(gt_dir + files[i]) for i in range(n)]
	
	return imgs, gt_imgs

def get_patches_from_img(img, patch_size = 80, stride = 80, binary = True):
    """ Computes the patches for a single image
		Args:
			img: input image. If binary: input of shape (size_x, size_y), else: input of shape (size_x, size_y,3)
			patch_size: size of patches to use
			stride: stride to be used in patches formation
			binary: wether the image is binary or not
		Return:
			array of patches for image
    """
    if(binary):
      patch = view_as_windows(img, (patch_size,patch_size), step=stride)
    else:
      patch = view_as_windows(img, (patch_size,patch_size, 3), step=stride)
    return patch

def reconsrtuct_img_from_patches(patch, output_image_size=(400,400), patch_size = 80, stride = 80, mode = 'max', binary= False):
	""" Reconstructs a single image from patches
		Args:
			patch: array of patches, if binary: input shape (nbpatches_x, nbpatches_y, patch_size, patch_size) else
					input shape (nbpatches_x, nbpatches_y, patch_size, patch_size, 3)
			output_image_size: size of output image
			patch_size: size of patches to use
			stride: stride to be used in patches formation
			mode: aggragtion method to use for overlapping patches. if values is'max', else takes average
			binary: wether the image is binary or not
		Return:
			array of patches for image
    """
    
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
  
def create_train_test_generators(X_train, y_train, X_test, y_test, batch_size = 32, rotations_range = 0, padding = 'reflect', \
					hori_flip = False, vert_flip = False, brightness_range = [1.0,1.0], seed = 1):
	"""Create a test and a train generator to yield images in patches of 32 to the network
	Args:
		X_train: train patches
		y_train: train groundtruth patches 
		X_test: validation patches
		y_test: validation groundtruth patches 
		batch_size: size of the batches
		rotations_range: angle where random rotations can be performed
		padding: padding method to be used. possible values: 'reflect' or 'nearest'
		hori_flip: wether to perform horizontal flip
		vert_flip: wether to perform horizontal flip
		brightness_range: range of random brightness tranformations to perform
		seed: random seed
	Return:
		train and test generator
	"""

	# we define two arguments dictionaries with slightly altered parameters for images and groundtruths
	data_gen_args = dict(featurewise_center = False,
                    featurewise_std_normalization = False,
                    rotation_range = rotations_range,
                    width_shift_range = 0.0,
                    height_shift_range = 0.0,
                    preprocessing_function = lambda x: x/255.0,
                    zoom_range = 0.0,
                    fill_mode = padding,
                    horizontal_flip = True,
                    vertical_flip = True,
                    brightness_range = brightness_range,
                    data_format = "channels_last")

	data_gen_args_mask = dict(samplewise_center = False,
                    featurewise_std_normalization = False,
                    rotation_range = rotations_range,
                    width_shift_range = 0,
                    height_shift_range = 0,
                    zoom_range = 0,
                    fill_mode = padding,
                    preprocessing_function = lambda x: x/255.0,
                    horizontal_flip = True,
                    vertical_flip = True,
                    brightness_range = [1.0,1.0],
                    data_format = "channels_last")

	#create the generators with the given arguments
	image_datagen = kerasimg.ImageDataGenerator(**data_gen_args)
	mask_datagen = kerasimg.ImageDataGenerator(**data_gen_args_mask)
	#no transform on validation dataset
	image_datagen_val = kerasimg.ImageDataGenerator(data_format="channels_last")
	mask_datagen_val = kerasimg.ImageDataGenerator(data_format="channels_last")

	# Provide the same seed and keyword arguments to the fit and flow methods
	image_generator = image_datagen.flow(X_train, batch_size = batch_size, seed = seed)
	mask_generator = mask_datagen.flow(y_train, batch_size = batch_size, seed = seed)
	# combine generators into one which yields image and masks
	train_generator = zip(image_generator, mask_generator)

	image_generator_val = image_datagen_val.flow(X_test, shuffle = False, batch_size = batch_size, seed = seed)
	mask_generator_val = mask_datagen_val.flow(y_test, shuffle = False, batch_size = batch_size, seed = seed)
	# combine generators into one which yields image and masks
	test_generator = zip(image_generator_val, mask_generator_val)
	
	return train_generator, test_generator


def recall_m(y_true, y_pred):
    """Computes the recall of the model. Source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    Args:
        y_true: true expected result
        y_pred: result from the network
    Returns:
        recall
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
        precision
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

def remove_small_globs(prediction, threshold=16*16):
	"""
	Removes objects smaller than threshold in the prediction image
	Args:
		prediction: predicted results image
		threshold: minimum size of object to remove
	Return:
		prediction image without small objects
	"""
	labeled, nr_objects = scipy.ndimage.label(prediction)
	count_per_glob = np.zeros(nr_objects+1)

	for i in range(labeled.shape[0]):
		for u in range(labeled.shape[1]):
			count_per_glob[labeled[i][u]] = count_per_glob[labeled[i][u]] + 1
	for i in range(prediction.shape[0]):
		for u in range(prediction.shape[1]):
			if count_per_glob[labeled[i][u]] <= threshold:
				prediction[i][u] = 0
			# else keep prediction as is and let the thresholding decide if it's a road
	return prediction

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

def predict_with_patches(net, image, patch_size, stride, output_img_size = (608,608), aggregation_method = 'mean'):
	"""
	Predicts results on a single image
	Args:
		net: network to be used for prediction
		image: image to predict upon
		patch_size: size of patches to use
		stride: stride to be used in patches formation
		output_img_size: size of output image
		aggregation_method: how to aggrgate patches for prediction on entire image. possible values are 'max' or any other string
	Return
		returns the prediction on a single image
	"""
	patches = get_patches_from_img(image, patch_size = patch_size, stride = stride, binary = False)
	patches = patches.squeeze(2)
	p1 = patches.shape[0]
	p2 = patches.shape[1]
	flatten_size = p1*p2
	flatten_patches = patches.reshape((flatten_size,patch_size,patch_size,3))

	predicted_patches = net.predict(flatten_patches)
	predicted_patches = predicted_patches.reshape((p1,p2,patch_size,patch_size))
	net_result = reconsrtuct_img_from_patches(predicted_patches, output_img_size, patch_size=patch_size, \
                                              stride = stride, mode = aggregation_method, binary=True)
    
	return net_result

def save_all_results_patches(path_to_results, net, patch_size, stride, glob_remove = False, threshold = 100):
	"""
	Computes all the predcitions on the test set and saves them to a results folder
	Args:
		path_to_results: path where the results are to be saved
		net: network to be used for prediction
		patch_size: size of patches to use
		stride: stride to be used in patches formation
		glob_remove: wether to remove small objects
		threshold: threshold to be used for small objects removal
	"""
	satelite_images_path = prefix + 'test_set_images'
	test_paths = glob.glob(satelite_images_path + '/*/*.png')

	test_images = list(map(mpimg.imread, test_paths.copy()))

	for i in range(len(test_images)):
		net_result = predict_with_patches(net, test_images[i], patch_size, stride)
		name = test_paths[i]
		if glob_remove:
			net_result = remove_small_globs(net_result, threshold = threshold)
        
		Image.fromarray((net_result*255).astype(np.uint8)).save(path_to_results+"test_image_" + \
                                                str(int(re.search(r"\d+", name).group(0))) + ".png", "PNG")

def create_sumbmission():
	"""
	creates a submission file named 'submission.csv' which contains all the predictions in the AI crowd format
	"""
	submission_filename = 'submission.csv'
	image_filenames = []

	for i in range(1, 51):
		image_filename = 'results/test_image_' + '%.1d' % i + '.png'
		image_filenames.append(image_filename)
    
	msk.masks_to_submission(submission_filename, *image_filenames)
