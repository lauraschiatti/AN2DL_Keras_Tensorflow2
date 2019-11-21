# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import data_preparation as dp
from training_callbacks import callbacks, set_which_model


# Fix the seed for random operations
# to make experiments reproducible.
SEED = 1234
tf.random.set_seed(SEED)


# Multi-class Fashion MNIST classification
# ----------------------------------------

# x: 28x28 (grayscale images)
# y: 10 classes
    # T-shirt/top       # 0
    # Trouser/pants     # 1
    # Pullover shirt    # 2
    # Dress             # 3
    # Coat              # 4
    # Sandal            # 5
    # Shirt             # 6
    # Sneaker           # 7
    # Bag               # 8
    # Ankle boot        # 9


def create_model():

    which_model = 'base_weight_decay'
    set_which_model(which_model) # set model for training_callbacks

    # Create base model using functional API Model (e.g., Input -> Hidden -> Out)
    if which_model == 'model':
        # x = tf.keras.Input(shape=[28, 28])  # input tensor
        # flatten = tf.keras.layers.Flatten()(x)
        # h = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid)(flatten)  # hidden layers
        # output layer:probabccc of belonging to each class
        # out = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(h)
        # model = tf.keras.Model(inputs=x, outputs=out)

        # equivalent formulation:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
        model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

    # Create base model using sequential model (e.g., Input -> Hidden -> Out)
    elif which_model == 'base':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
        model.add(tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.sigmoid))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

    # Create model with Dropout layer
    elif which_model == 'base_dropout':

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
        model.add(tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.sigmoid))
        model.add(tf.keras.layers.Dropout(0.3))  # rate (probab): 0.3
        model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

    # Create model with weights penalty (L2 regularization)
    elif which_model == 'base_weight_decay':

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
        model.add(tf.keras.layers.Dense(units=1000,
                                        activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        model.add(tf.keras.layers.Dense(units=10,
                                        activation=tf.keras.activations.softmax,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

    return model


# 1 - Dataset
# -----------

# Load built-in dataset
# (train_images, train_labels), (test_images, test_labels)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Look at the shape of the dataset
print('x_train.shape', x_train.shape)
# (60000, 28, 28) 60,000 samples in our training set, and the images are 28 pixels x 28 pixels each


# Data splitting
TRAIN_SAMPLES = 50000 # e.g., Reserve 10,000 sampels for validation, then 50000 samples for training

x_valid = x_train[TRAIN_SAMPLES:, ...]
y_valid = y_train[TRAIN_SAMPLES:, ...]

x_train = x_train[:TRAIN_SAMPLES, ...]
y_train = y_train[:TRAIN_SAMPLES, ...]

# or the other way around, reserve 10,000 samples for validation
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]


# Create Datasets
bs = 32
train_dataset = dp.images_dataset(x_train, y_train, bs=bs, shuffle=True)
valid_dataset = dp.images_dataset(x_valid, y_valid, bs=1)
test_dataset = dp.images_dataset(x_test, y_test, bs=1)

print('---------- ---------- ---------- ---------- ---------- ')


# Check that is everything is ok..
# -------------------------------

print ('Checking training_set was correctly built')
iterator = iter(train_dataset)
sample, target = next(iterator)

# Just for visualization purposes
sample = sample[0, ...]  # select first image in the batch
sample = sample * 255  # denormalize

from PIL import Image
img = Image.fromarray(np.uint8(sample))
img = img.resize([128, 128])
print('img', img)
img.show()

print('target', target[0], '\n')  # select corresponding target
print('---------- ---------- ---------- ---------- ---------- ')


# 2 - Model creation => define model architecture
# -----------------------------
# x: 28x28 (grescale images)
# y: 10 classes

# then, in: 28x28 -> h: 10 units -> out: 10 units (number of classes)

model = create_model()

# Visualize created model as a table
model.summary()

# Visualize initialized weights
print('model initial weights', model.weights)

# Representation of the model as a JSON string
# json_string = model.to_json()
# print(json_string)
# loaded_model = tf.keras.models.model_from_json(json_string)

print('---------- ---------- ---------- ---------- ---------- ')


# 3 - Specify the training configuration (optimizer, loss, metrics)
# -----------------------------------------------------------------
loss = tf.keras.losses.CategoricalCrossentropy() # loss function to minimize

lr = 1e-4 # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # stochastic gradient descent optimizer

metrics = ['accuracy'] # validation metrics to monitor

# Compile Model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)


# 4 - Train the model
# -------------------

# Train the model by slicing the data into 'batches'
# of size 'batch_size', and repeatedly iterating over
# the entire dataset for a given number of 'epochs'

print('Fit model on training data ...')

# history = model.fit(x=train_dataset,  # you can give directly numpy arrays x_train
#           y=None,   # if x is a Dataset y has to be None, y_train otherwise
#           epochs=10,
#           steps_per_epoch=int(np.ceil(x_train.shape[0] / bs)),  # how many batches per epoch
#           validation_data=valid_dataset,  # give a validation Dataset if you created it manually,
#                                           # otherwise you can use 'validation_split' for automatic split
#           validation_steps=10000)  # number of valid samples


# Training with callbacks
history = model.fit(x=train_dataset,
          epochs=10,  #### set repeat in training dataset
          steps_per_epoch=int(np.ceil(x_train.shape[0] / bs)),
          # we pass some validation for
          # monitoring validation loss and metrics
          # at the end of each epoch
          validation_data=valid_dataset,
          validation_steps=10000,
          callbacks=callbacks)

# The returned 'history' object holds a record of the loss values and metric values during training
# print('\nhistory dict:', history.history)


# 5 - Evaluate Model
# --------------

# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
eval_out = model.evaluate(x=x_test / 255.,
                          y=tf.keras.utils.to_categorical(y_test),
                          verbose=0)
print('test loss, test acc:', eval_out)

print('---------- ---------- ---------- ---------- ---------- ')


# 5 - Compute prediction
# ----------------------

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for picture shoe.png')

shoe_img = Image.open('shoe.png').convert('L') # open into greyscale, or L mode

shoe_arr = np.expand_dims(np.array(shoe_img), 0)

out_softmax = model.predict(x=shoe_arr / 255.)

print('out_softmax', out_softmax) # is already a probability distribution (softmax)

# Get predicted class as the index corresponding to the maximum value in the vector probability
predicted_class = tf.argmax(out_softmax, 1)
print('predicted_class', predicted_class)  # predictions as tensors
