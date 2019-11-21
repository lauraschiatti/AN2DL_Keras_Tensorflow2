# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
import tensorflow as tf

from image_data_generator import setup_data_generator
from CNNClassifier import CNNClassifier


# CNN_classification
# ---------------------------------------------

# Fix the seed for random operations
# to make experiments reproducible.
seed = 1234
tf.random.set_seed(seed)

# Get current working directory
cwd = os.getcwd()

#@todo: solve GPU
# Set GPU memory growth
# Prevent tensorflow from allocating the totality of a GPU memory
# Allows to only as much GPU memory as needed
# gpus = tf.config.experimental.list_physical_devices('GPU')
#
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)



# ImageDataGenerator
train_dataset, valid_dataset, test_dataset, img_w, img_h, channels, num_classes = setup_data_generator()


# Convolutional Neural Network (CNN)
depth = 5
num_filters = 8 #num_filters

# Create Model instance
model = CNNClassifier(depth=depth, num_filters=num_filters, num_classes=num_classes)

# Build Model (Required)
model.build(input_shape=(None, img_h, img_w, channels))


# Visualize created model as a table
model.feature_extractor.summary()

# Visualize initialized weights
print("initial model weights", model.weights)


# Prepare the model for training
# ------------------------------

# Optimization params
# -------------------
loss = tf.keras.losses.CategoricalCrossentropy()

lr = 1e-3 # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Validation metrics
# ------------------
metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)