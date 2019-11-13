# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# Deep Learning Framework
# ---------------

import tensorflow as tf

from utils import data_preparation as dp

# Set the seed for random operations.
# This let our experiments to be reproducible.
tf.random.set_seed(1234)

# ------------------------------------------------------------------ #
                ##### Data Preparation #####
# ------------------------------------------------------------------ #

## Dataset and Data Loader ##

# different ways of creating a dataset
dp.sequential_dataset()
print("\n")

dp.dataset_from_tensors()
print("\n")

dp.dataset_from_tensor_slices()
print("\n")

zipped = dp.combine_datasets()
print("\n")

dp.iterate_range_dataset(zipped)

## Model creation ##
# tf.keras.layers
# tf.keras.Model
# tf.keras.Sequential



# ------------------------------------------------------------------ #
                ##### Training Loop #####
# ------------------------------------------------------------------ #

## Model training and validation ##
# model.fit
# tf.keras.optimizers
# tf.keras.losses


## Model test ##
#  model.evaluate
#  model.metrics
#  model.predict


## Save and Restore models ##
# callbacks.ModelCheckpoint
# model.save_weights
# model.save

## Visualize Learning
# callbacks.Tensorboard
