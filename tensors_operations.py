# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# Operations on tensors
# ---------------

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array2d = np.array(matrix, dtype=np.int32)
# tf.convert_to_tensor(object, dtype)
tensor2d = tf.constant(array2d, dtype=tf.float32)


# Cast: tf.cast(x, dtype)
# -----------------------

print("Cast ... ")
print(tensor2d.dtype)
tensor2d = tf.cast(tensor2d, dtype=tf.int32)
print(tensor2d.dtype)

# ?tf.DType
# -----------------------


# Reshape:
# --------

print("Reshape ... ")
# tf.reshape(tensor, shape)
tensor1d = tf.range(9)
print("tensor1d", tensor1d)

# Reshape organization of values in the tensor (e.g 1D to 2D-tensor)
tensor2d = tf.reshape(tensor1d, shape=[3, 3])
# tensor2d = tf.reshape(tensor1d, shape=[3, -1]) # negative dimension is accepted
print("tensor2d", tensor2d)

# flattening: move from n-D vector to 1-D one
flattened = tf.reshape(tensor2d, shape=[-1])
print("flattened", flattened)


# # Expand dimensions: obtain 3-D tensor from 2-D one.
# # Mainly add additional dimensions to the original tensor
# # e.g. to add channels to an image
# tf.expand_dims(input, axis)
tensor2d = tf.reshape(tf.range(1, 5), shape=[2, 2])
print("tensor2d", tensor2d)

tensor3d = tf.expand_dims(tensor2d, axis=-1)
# tensor3d = tf.reshape(tensor2d, shape=[2, 2, 1])
print("tensor3d", tensor3d)

# Also possible the other way around
# tf.squeeze(input, axis)
tensor2d = tf.squeeze(tensor3d, axis=-1)
# tensor2d = tf.reshape(tensor3d, shape=[2, 2])
print("tensor2d squeeze", tensor2d)

# --------


# Math: operations between tensors are performed element-wise
# -----

# +, -, *, / operators (element-wise)
t1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32) # casting required if different types
t2 = tf.eye(num_rows=3, num_columns=3) # diagonal matrix
print("add", t1+t2)
# tf.add(t1, t2)
print("subs", t1-t2)
# tf.subtract(t1, t2)
print("multi", t1*t2)
# tf.multiply(t1, t2)
print("division", t1/(t2+1))
# tf.divide(t1, tf.add(t2, 1))

# tf.tensordot(a, b, axes)
t = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
tf.tensordot(t, t, axes=[[1], [0]]) # matrix multiplication (dot product)


# Slicing: extract some elements from the original tensor
# --------

tensor2d = tf.convert_to_tensor(matrix, dtype=tf.float32)
print("tensor2d", tensor2d)

# Pythonic '[]'
# -------------

# Get a single element: tensor[idx1, idx2, ..., idxN]
# e.g. get element '3'
print("tensor2d[0, 2]", tensor2d[0, 2])
print("tensor2d[0, -1]", tensor2d[0, -1])
print("tensor2d[-3, -1]", tensor2d[-3, -1])

# Get a slice: tensor[startidx1:endidx1:step1,..., startidxN:endidxN:stepN] (endidx excluded)
#          |5 6|
# e.g. get |8 9| sub-tensor
print("tensor2d[1:3:1, 1:3:1]", tensor2d[1:3:1, 1:3:1])
print("tensor2d[1:3, 1:3]", tensor2d[1:3, 1:3])
print("tensor2d[1:, 1:]", tensor2d[1:, 1:])

# Missing indices are considered complete slices
# e.g. get first row
print("tensor2d[0, :]", tensor2d[0, :])

# Negative indices are accepted
print("tensor2d[0:-1, 0]", tensor2d[0:-1, 0])

# e.g. get first row reversed
print("tensor2d[0, ::-1]", tensor2d[0, ::-1])
# -------------

# API
# ---

# tf.slice(input_, begin, size)

#          |5 6|
# e.g. get |8 9| sub-tensor
tf.slice(tensor2d, [1, 1], [2, 2])

# indexing with condition
array2d = np.array(matrix, dtype=np.float32)
array2d[np.where(array2d % 2 == 0)]

# tensor2d[tf.where(array2d % 2 == 0)] does not work!
# use tf.gather_nd(params, indices) and tf.where(condition) to find indices
indices = tf.where(tensor2d % 2 == 0)
tf.gather_nd(tensor2d, indices)

# ---