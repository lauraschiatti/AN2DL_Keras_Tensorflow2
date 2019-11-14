# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

# ------------------------------------------------------------------ #
    ##### Example: Fashion MNIST - Multi-class classification #####
# ------------------------------------------------------------------ #

import tensorflow as tf

from utils import data_preparation as dp

# Set the seed for random operations.
# This let our experiments to be reproducible.
tf.random.set_seed(1234)


### Dataset ###

# Load built-in dataset
# ---------------------
# (train_images, train_labels), (test_images, test_labels)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("inputs", x_train) # inputs shape
print("outputs", y_train) # outputs shape

# Split in training and validation sets
# e.g., 50000 samples for training and 10000 samples for validation

x_valid = x_train[50000:, ...]
y_valid = y_train[50000:, ...]

x_train = x_train[:50000, ...]
y_train = y_train[:50000, ...]


# Create Training Dataset object
# ------------------------------
train_dataset = dp.multiclass_dataset(x_train, y_train, bs=32, shuffle=True)

# Create Validation Dataset
# -----------------------
valid_dataset = dp.multiclass_dataset(x_valid, y_valid, bs=1)

# Create Test Dataset
# -------------------
test_dataset = dp.multiclass_dataset(x_test, y_test, bs=1)


### Check that is everything is ok.. ###