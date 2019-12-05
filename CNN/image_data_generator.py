# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# Fix the seed for random operations
# to make experiments reproducible.
seed = 1234
tf.random.set_seed(seed)

# Get current working directory
cwd = os.getcwd()


# Uc Merced Land Use Dataset

def setup_data_generator():
	print("ImageDataGenerator ... ")

	# 1. Organize dataset folders:
	# --------------------------

	# - UCMerced_LandUse/
	#     - training/
	#         - agricultural/
	#             - img1, img2, …, imgN
	#		  ...
	#     - validation/
	#         - agricultural/
	#             - img1, img2, …, imgN
	#         - …
	#         ...
	#     - test/
	#         ...


	# 2 - Define data augmentation configuration
	# -----------------------------------------------------------------------

	apply_data_augmentation = False

	if apply_data_augmentation:
		train_data_gen = ImageDataGenerator(rotation_range=10,
											width_shift_range=10,
											height_shift_range=10,
											zoom_range=0.3,
											horizontal_flip=True,
											vertical_flip=True,
											fill_mode='constant',
											cval=0,
											# normalization
											# rescale 1./255 is to transform every pixel value from range [0,255] -> [0,1].
											rescale=1. / 255)
	else:
		train_data_gen = ImageDataGenerator(rescale=1. / 255)

	valid_data_gen = ImageDataGenerator(rescale=1. / 255)
	test_data_gen = ImageDataGenerator(rescale=1. / 255)



	# 3 - Create generators to read images from dataset directory
	# -----------------------------------------------------------

	# UC Merced Land Use Dataset
	# x: 256x256 pixels
	# y: 21 classes, 100 images each

	dataset_dir = os.path.join(cwd, 'datasets/UCMerced_LandUse')

	# Batch size
	bs = 8

	# img shape
	img_h = 256
	img_w = 256
	channels = 3

	num_classes = 21

	# img shape
	# print("imag shape ... ")
	# training_dir = os.path.join(dataset_dir, 'training')
	#
	# images, labels = next(train_data_gen.flow_from_directory(training_dir))
	# print(images.dtype, images.shape)
	# print(labels.dtype, labels.shape)
	#
	# img_h = images.shape[1]
	# img_w = images.shape[2]
	# channels = images.shape[3]
	# num_classes = labels.shape[1]
	#
	# print("img_h", img_h)
	# print("img_w", img_w)
	# print("channels", )
	# print("num_classes", labels.shape[1])

	decide_class_indices = False
	if decide_class_indices:
		classes = ['agricultural',      # 0
				   'airplane',          # 1
				   'baseballdiamond',   # 2
				   'beach',             # 3
				   'buildings',         # 4
				   'chaparral',         # 5
				   'denseresidensial',  # 6
				   'forest',            # 7
				   'freeway',           # 8
				   'golfcourse',        # 9
				   'harbor',            # 10
				   'intersection',      # 11
				   'mediumresidential', # 12
				   'mobilehomepark',    # 13
				   'overpass',          # 14
				   'parkinglot',        # 15
				   'river',             # 16
				   'runway',            # 17
				   'sparseresidential', # 18
				   'storagetanks',      # 19
				   'tenniscourt']       # 20
	else:
		classes = None

	# Training
	training_dir = os.path.join(dataset_dir, 'training')
	train_gen = train_data_gen.flow_from_directory(training_dir,
												   batch_size=bs,
												   classes=classes,
												   # targets are directly converted into one-hot vectors
												   class_mode='categorical',
												   shuffle=True,
												   seed=seed)

	# Validation
	validation_dir = os.path.join(dataset_dir, 'validation')
	valid_gen = valid_data_gen.flow_from_directory(validation_dir,
												   batch_size=bs,
												   classes=classes,
												   class_mode='categorical',
												   shuffle=False,
												   seed = seed)

	# Test
	test_dir = os.path.join(dataset_dir, 'test')
	test_gen = test_data_gen.flow_from_directory(test_dir,
												 batch_size=bs,
												 classes=classes,
												 class_mode='categorical',
												 # to yield the images in “order”, to predict the outputs
												 # and match them with their unique ids or filenames
												 shuffle=False,
												 seed=seed)


	# 4 - Create Dataset objects
	# ----------------------

	# Training
	train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,  # generator
												   output_types=(tf.float32, tf.float32),
												   output_shapes=([None, img_h, img_w, channels], [None, num_classes]))

	# Shuffle (Already done in generator..)

	# Normalize images (Already done in generator..)

	# 1-hot encoding <- for categorical cross entropy (Already done in generator..)

	# Divide in batches (Already done in generator..)

	train_dataset = train_dataset.repeat() # repeat

	# Validation
	# ----------
	valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
												   output_types=(tf.float32, tf.float32),
												   output_shapes=([None, img_h, img_w, channels], [None, num_classes]))
	valid_dataset = valid_dataset.repeat()  # repeat

	# Test
	# ----
	test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
												  output_types=(tf.float32, tf.float32),
												  output_shapes=([None, img_h, img_w, channels], [None, num_classes]))
	test_dataset = test_dataset.repeat()  # repeat


	print("class labels ...", train_gen.class_indices)  # check the class labels

	return train_dataset, valid_dataset, test_dataset, train_gen, valid_gen, test_gen


# Let's test data augmentation
# ----------------------------
# import time
# import matplotlib.pyplot as plt
#
#
# fig = plt.figure()
# ax = fig.gca()
# fig.show()
#
# iterator = iter(train_dataset)
#
# for _ in range(100):
#     augmented_img, target = next(iterator)
#     augmented_img = augmented_img[0]  # First element
#     augmented_img = augmented_img * 255  # denormalize
#
#     plt.imshow(np.uint8(augmented_img))
#     fig.canvas.draw()
#     time.sleep(1)7