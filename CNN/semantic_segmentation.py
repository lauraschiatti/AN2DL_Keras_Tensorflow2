
## FCNN in keras


# It is necessary to get and set the weights by of networks by means of methods
# get_weights and set_weights

# get the weights of the trained CNN
# w7, b7 = model.layers[7].get_weights()

# reshape these weights to become a convolution
# w7.reshape(20, 20, 10, 256)

# assign these weights to the FCNN architecture
# model2.layers[i].set_weights(w7, b7)


# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# Fix the seed for random operations
# to make experiments reproducible.
SEED = 1234
tf.random.set_seed(SEED)


# Get current working directory
import os
cwd = os.getcwd()

# Set GPU memory growth
# Allows to only as much GPU memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



# Oxford Pet Dataset
# Directory structure
# - Pet_Dataset/
#     - training/
#         - images/
#             - img1, img2, …, imgN
#         - masks/
#             - mask1, mask2, ... , maskN
#     - validation/
#         - images/
#             - img1, img2, …, imgN
#         - masks/
#             - mask1, mask2, ... , maskN
#     - test/
#         - images/
#             - img1, img2, …, imgN
#         - masks/
#             - mask1, mask2, ... , maskN



# ImageDataGenerator
# ------------------

from keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = False

# Create training ImageDataGenerator object
# We need two different generators for images and corresponding masks

if apply_data_augmentation:
    train_img_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            rescale=1./255)
    train_mask_data_gen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=10,
                                             height_shift_range=10,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='constant',
                                             cval=0)
else:
    train_img_data_gen = ImageDataGenerator(rescale=1./255)
    train_mask_data_gen = ImageDataGenerator()

# Create validation and test ImageDataGenerator objects
valid_img_data_gen = ImageDataGenerator(rescale=1./255)
valid_mask_data_gen = ImageDataGenerator()
test_img_data_gen = ImageDataGenerator(rescale=1./255)
test_mask_data_gen = ImageDataGenerator()