# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np

# Fix the seed for random operations to make experiments reproducible.
SEED = 1234
tf.random.set_seed(SEED)

# Get current working directory
cwd = os.getcwd()


# UC Merced Land Use Dataset
# --------------------------

# 1. Organize dataset folders:

# - UCMerced_LandUse/
#     - training/
#         - agricultural/
#             - img1, img2, …, imgN
#         - …
#         - parkinglot/
#             - img1, img2, ... , imgN
#     - validation/
#         - agricultural/
#             - img1, img2, …, imgN
#         - …
#         - parkinglot/
#             - img1, img2, ... , imgN
#     - test/
#         - agricultural/
#             - img1, img2, …, imgN
#         - …
#         - parkinglot/
#             - img1, img2, ... , imgN



# 2 - Initialize training, validation and test ImageDataGenerator objects
# ------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = True

# Create training ImageDataGenerator object
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
                                        rescale=1./255)
else:
    train_data_gen = ImageDataGenerator(rescale=1./255)

# Create validation and test ImageDataGenerator objects
valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)



# 3 - Create generators to read images from dataset directory
# -------------------------------------------------------
# x: 256x256 pixels
# y: 21 classes, 100 images each

dataset_dir = os.path.join(cwd, 'datasets/UCMerced_LandUse')

# Batch size
bs = 8

# img shape
img_h = 256
img_w = 256

num_classes = 21

# Training
training_dir = os.path.join(dataset_dir, 'training')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors

# Validation
validation_dir = os.path.join(dataset_dir, 'validation')
valid_gen = valid_data_gen.flow_from_directory(validation_dir,
                                               batch_size=bs,
                                               class_mode='categorical',
                                               shuffle=False,
                                               seed=SEED)

# Test
test_dir = os.path.join(dataset_dir, 'test')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             batch_size=bs,
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=SEED)





# 4 - Create Dataset objects
# ----------------------

# Training
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, # generator
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

# Shuffle (Already done in generator..)
# Normalize images (Already done in generator..)
# 1-hot encoding <- for categorical cross entropy (Already done in generator..)
# Divide in batches (Already done in generator..)

# Repeat
# Without calling the repeat function the dataset
# will be empty after consuming all the images
train_dataset = train_dataset.repeat()

# Validation
# ----------
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
valid_dataset = valid_dataset.repeat() # repeat

# Test
# ----
test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
test_dataset = valid_dataset.repeat() # repeat


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
#     time.sleep(1)