# !/usr/bin/env python3.6
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


## Dataset and Data Loader ##

# different ways of creating a dataset
dp.sequential_dataset()
print("\n")

tensor = [1, 2, 3, 4, 5, 6, 7, 8, 9]
dp.dataset_from_tensors(tensor)
print("\n")

x_train = np.random.uniform(size=[10, 2, 2])
y_train = np.random.randint(10, size=[10])
dp.dataset_from_tensor_slices(x_train, y_train)
print("\n")


x = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=10))
y = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])
zipped = dp.combine_datasets(x, y)
print("\n")

dp.iterate_range_dataset(zipped)
