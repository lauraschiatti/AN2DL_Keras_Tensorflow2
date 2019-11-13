# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# Load and show an image with Pillow
from PIL import Image
# Load the image
img = Image.open('airplane.jpg')

# Get basic details about the image
print("img", img)
print(img.format)
print(img.mode)
print(img.size)

# Show the image
img.show()

# Convert image to tensor: img -> array -> tensor
rgb_tensor = tf.convert_to_tensor(np.array(img))
print("rgb_tensor", rgb_tensor)

# obtain gray-scaled image (only one channel)
gs_tensor = tf.convert_to_tensor(np.array(img.convert('L')))
print("gs_tensor", gs_tensor)

# Change the color of the airplane!
# tf.where again, but tf.where(condition, x, y)
new_t = tf.where(tf.expand_dims(gs_tensor, -1) > 235, [255, 178, 102], rgb_tensor)
new_img = Image.fromarray(new_t.numpy().astype(np.uint8))
#show the image
new_img.show()


# Splitting:
# ----------
print("rgb_tensor.shape", rgb_tensor.shape) # image shape (tensor shape)

# tf.split(value, num_or_size_splits, axis) # num_or_size_splits: parts in which it will be split around axis
# vertical
output = tf.split(rgb_tensor, 3, axis=0)
print("output[0].shape", output[0].shape) # output[0] - first 300 rows of the tensor
print("output[1].shape", output[1].shape)
print("output[2].shape", output[2].shape)

# or
top, bottom = tf.split(rgb_tensor, [500, -1], axis=0)
print("top.shape", top.shape)
print("bottom.shape", bottom.shape)

# it can be done horizontally too
top_left, top_right = tf.split(top, [900, -1], axis=1)
bottom_left, bottom_right = tf.split(bottom, [900, -1], axis=1)

Image.fromarray(top_left.numpy())
Image.fromarray(top_right.numpy())
Image.fromarray(bottom_left.numpy())
Image.fromarray(bottom_right.numpy())

# ----------
