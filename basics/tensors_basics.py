# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

# A tensor is an N-dimensional array containing the same type of data (int32, bool, etc.)
#   => It's a collection of feature vectors (i.e., array) of n-dimensions.

import numpy as np

# Create a tensor
# ---------------

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# np.array(object, dtype)
array2d = np.array(matrix, dtype=np.int32)
print("array2d", array2d)

# other alternatives:
# np.zeros
zeros = np.zeros(shape=[4, 4], dtype=np.int32)
print("zeros", zeros)

# np.ones
ones = np.zeros(shape=[4, 4], dtype=np.int32)
print("ones", ones)

# np.zeros_like / np.ones_like
zeros_like = np.zeros_like(array2d)
print("zeros_like", zeros_like)

# np.arange
range_array = np.arange(start=0, stop=10, step=2)
print("range_array", range_array)

# random integers over [low, high)
rnd_int = np.random.randint(low=0, high=10, size=3)
print("rnd_int", rnd_int)

# sampling from distribution: np.random
# uniform [0, 1)
rnd_uniform = np.random.rand(3, 3)  # rand(d0, d1, ..., dN)
print("rnd_uniform", rnd_uniform)

# normal with mean 0 and variance 1
rnd_normal = np.random.randn(3, 3)  # randn(d0, d1, ..., dN)
print("rnd_normal", rnd_normal)

# many others (see documentation)
#?np.random


# Attributes
# ----------

# dtype: type of the elements
print('array.dtype: %s' % array2d.dtype)

# shape: dimensions of the array
print('array.shape: ', array2d.shape)

# ndim: number of dimensions of the array (axis)
print('array.ndim: {}'.format(array2d.ndim))


# ----------- ----------- ----------- ----------- ----------- ----------- ----------- #

import tensorflow as tf

# Create a tensor
# ---------------

# tf.constant(object, dtype)
tensor2d = tf.constant(matrix, dtype=tf.int32)

# tf.convert_to_tensor(object, dtype)
tensor2d = tf.constant(array2d, dtype=tf.float32)

# other alternatives to create a tensor:
# tf.zeros
zeros = tf.zeros(shape=[4, 4], dtype=tf.int32)

# tf.ones
ones = tf.zeros(shape=[4, 4], dtype=tf.int32)

# tf.zeros_like / tf.ones_like
zeros_like = tf.zeros_like(array2d)

# np.arange
range_array = tf.range(start=0, limit=10, delta=2)

# sample from some distribution (different)
rnd_uniform = tf.random.uniform(shape=[5, 5], minval=10, maxval=20)

# ?tf.random

# remember you can use convert_to_tensor()
# e.g.
rnd_rayleigh = tf.convert_to_tensor(
    np.random.rayleigh(size=[2, 2]), dtype=tf.float32)
rnd_rayleigh


# Attributes
# ----------

# dtype: type of the elements
print('tensor2d.dtype: %s' % tensor2d.dtype)

# shape: dimensions of the array
print('tensor2d.shape: ', tensor2d.shape)

# ndim: number of dimensions of the array (axis)
print('tensor2d.ndim: {}'.format(tensor2d.ndim))

# device: device placement for tensor
print('tensor2d.device: {}'.format(tensor2d.device))

