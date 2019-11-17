# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


# Create a Dataset of sequential numbers
#  --------------------------------------
def sequential_dataset():

    print("Dataset.range examples:")
    print("-----------------------")

    # Create dataset
    range_dataset = tf.data.Dataset.range(0, 20, 1)

    # Iterate over the dataset
    print("\n1. Dataset ... ")
    for el in range_dataset:
        print(el)

    # Divide into batches
    bs = 3 # each batch contains 3 samples

    # Remark: if N samples not divisible by # batches,
        # the last one is smaller and Keras gives the chance to discard the sample in excess by using drop_remainder
    range_dataset = tf.data.Dataset.range(0, 20, 1).batch(bs, drop_remainder=False)

    print("\n2. Dataset + batch ... ")
    for el in range_dataset:
        print(el)

    # Apply a transformation to each element
    def map_fn(x):
        return x**2

    range_dataset = tf.data.Dataset.range(0, 20, 1).batch(bs, drop_remainder=False).map(map_fn) # compute mapping for every new sample

    print("\n3. Dataset + batch + map ... ")
    for el in range_dataset:
        print(el)

    # Filter dataset based on a condition
    def filter_fn(x):
        return tf.equal(tf.math.mod(x, 2), 0)

    range_dataset = tf.data.Dataset.range(0, 20, 1).filter(filter_fn)

    print("\n4. Dataset + filter ... ")
    for el in range_dataset:
        print(el)

    # Random shuffling
    range_dataset = tf.data.Dataset.range(0, 20, 1).shuffle(
        buffer_size=20, reshuffle_each_iteration=False, seed=1234).batch(bs) # obtain random permutation

    print("\n5. Dataset + shuffle + batch ... ")
    for el in range_dataset:
        print(el)


# Create Dataset as unique element
# --------------------------------
def dataset_from_tensors(tensor):

    from_tensors_dataset = tf.data.Dataset.from_tensors(tensor)

    print("Dataset.from_tensors example:")
    print("-----------------------------")
    for el in from_tensors_dataset:
        print(el)


# Create a Dataset of slices
# --------------------------
def dataset_from_tensor_slices(x, y):
    # All the elements must have the same size in first dimension (axis 0)
    from_tensor_slices_dataset = tf.data.Dataset.from_tensor_slices((x, y))

    print("Dataset.from_tensor_slices example:")
    print("-----------------------------")
    for el in from_tensor_slices_dataset:
        print(el)

    return from_tensor_slices_dataset


# Combine multiple datasets
# -------------------------
def combine_datasets(x, y):
    zipped = tf.data.Dataset.zip((x, y))

    print("Dataset.from_tensors example:")
    print("-----------------------------")
    for el in zipped:
        print(el)

    return zipped


# Iterate over range dataset
#  --------------------------
def iterate_range_dataset(zipped):

    # for a in b
    for el in zipped:
        print(el)

    print('\n')

    # for a in enumerate(b)
    for el_idx, el in enumerate(zipped):
        print(el)

    print('\n')

    # get iterator
    iterator = iter(zipped)
    print(next(iterator))

# Transform targets into one-hot encoding form
# for categorical cross entropy
#  --------------------------
def to_categorical(x_, y_):

    return x_, tf.one_hot(y_, depth=10) # 10 classes


# Normalize images
#  --------------------------
# convert our data type to float32 and normalize pixel values to the range [0, 1]
def normalize_img(x_, y_):

    return tf.cast(x_, tf.float32) / 255., y_


# Divide in batches
#  --------------------------
def divide_batches(dataset, bs):
    train_dataset = dataset.batch(bs)

    # Repeat
    # Without calling the repeat function the dataset
    # will be empty after consuming all the images (after the batching)
    train_dataset = train_dataset.repeat()

    return train_dataset


# Create Multiclass Dataset
#  --------------------------------------
# Preprocess input data

def images_dataset(x, y, bs, shuffle=False, ): # x: images, y: labels

    # Create dataset
    dataset = dataset_from_tensor_slices(x, y)

    # Shuffle
    if shuffle == True: # No need to shuffle validation and test sets
        print("1. Dataset + shuffle ... ")
        dataset = dataset.shuffle(buffer_size=x.shape[0])


    # Normalization
    print("2. Dataset + shuffle (train) + normalization  ... ")
    dataset = dataset.map(normalize_img) # compute mapping for every sample


    # One-hot Encoding
    print("3. Dataset + shuffle (train) + normalization + one-hot ... ")
    dataset = dataset.map(to_categorical) # compute mapping for every sample


    # Divide dataset into batches
    print("4. Dataset + shuffle (train) + normalization + one-hot + batch ... ")
    dataset = divide_batches(dataset, bs)

    return dataset


def iterate_dataset(dataset):
    for el in dataset:
        print(el)