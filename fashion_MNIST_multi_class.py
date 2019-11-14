# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from utils import data_preparation as dp
from utils.training_callbacks import ckpt_callback, tb_callback


# Set the seed for random operations.
# This let our experiments to be reproducible.
tf.random.set_seed(1234)


# Fashion MNIST classification
# ----------------------------

# x: 28x28 (grayscale images)
# y: 10 classes
    # 1. T-shirt/top
    # 2. Trouser/pants
    # 3. Pullover shirt
    # 4. Dress
    # 5. Coat
    # 6. Sandal
    # 7. Shirt
    # 8. Sneaker
    # 9. Bag
    # 10. Ankle boot


# 1 - Dataset
# ---------------------

# Load built-in dataset
# (train_images, train_labels), (test_images, test_labels)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# print("inputs", x_train) # inputs shape
# print("outputs", y_train) # outputs shape

# Split in training and validation sets
TRAIN_SAMPLES = 50000 # e.g., 50000 samples for training and 10000 samples for validation
x_valid = x_train[TRAIN_SAMPLES:, ...]
y_valid = y_train[TRAIN_SAMPLES:, ...]

x_train = x_train[:TRAIN_SAMPLES, ...]
y_train = y_train[:TRAIN_SAMPLES, ...]

# Create Datasets
bs = 32
train_dataset = dp.multiclass_dataset(x_train, y_train, bs=bs, shuffle=True)
valid_dataset = dp.multiclass_dataset(x_valid, y_valid, bs=1)
test_dataset = dp.multiclass_dataset(x_test, y_test, bs=1)

print("---------- ---------- ---------- ---------- ---------- ")

# Check that is everything is ok..
# ---------------------

print ("Checking training_set was correctly built")
iterator = iter(train_dataset)
sample, target = next(iterator)

# Just for visualization purposes
sample = sample[0, ...]  # select first image in the batch
sample = sample * 255  # denormalize

from PIL import Image
img = Image.fromarray(np.uint8(sample))
img = img.resize([128, 128])
print("img", img)
img.show()

print("target", target[0], "\n")  # select corresponding target
print("---------- ---------- ---------- ---------- ---------- ")


# 2 - Create Model
# ------------
# e.g. in: 28x28 -> h: 10 units -> out: 10 units (number of classes)

# Define Input keras tensor
x = tf.keras.Input(shape=[28, 28])

# Define and chain intermediate hidden layers
flatten = tf.keras.layers.Flatten()(x)
h = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid)(flatten)

# Define and chain output layer => interpreted as probab of belonging to each class
out = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(h)


# Create Model instance defining inputs and outputs
model = tf.keras.Model(inputs=x, outputs=out)

# Visualize created model as a table
model.summary()

# Visualize initialized weights
print("model initial weights", model.weights)

print("---------- ---------- ---------- ---------- ---------- ")


# Equivalent formulation: Create model with sequential
# (uncomment to run)
# seq_model = tf.keras.Sequential()
# seq_model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # or as a list
# seq_model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid))
# seq_model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

# seq_model.summary()
# seq_model.weights


# 3 - Training Loop
# ----------------------

# Prepare the model for training: optimization params and compilation
loss = tf.keras.losses.CategoricalCrossentropy()

lr = 1e-4 # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

metrics = ['accuracy'] # validation metrics

# compile Model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

# Training the model
# model.fit(x=train_dataset,  # you can give directly numpy arrays x_train
#           y=None,   # if x is a Dataset y has to be None, y_train otherwise
#           epochs=10,
#           steps_per_epoch=int(np.ceil(x_train.shape[0] / bs)),  # how many batches per epoch
#           validation_data=valid_dataset,  # give a validation Dataset if you created it manually,
#                                           # otherwise you can use 'validation_split' for automatic split
#           validation_steps=10000)  # number of batches in validation set


# Training with callbacks
model.fit(x=train_dataset,
          epochs=10,  #### set repeat in training dataset
          steps_per_epoch=int(np.ceil(x_train.shape[0] / bs)),
          validation_data=valid_dataset,
          validation_steps=10000,
          callbacks=[ckpt_callback, tb_callback])


# 4 - Test Model
# ----------------------
# Let's try a different way to give data to model
# using directly the NumPy arrays

# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

eval_out = model.evaluate(x=x_test / 255.,
                          y=tf.keras.utils.to_categorical(y_test),
                          verbose=0)

print("eval_out", eval_out)
print("---------- ---------- ---------- ---------- ---------- ")


# 5 - Compute prediction
# ----------------------

# Compute output given x
print("Compute prediction, picture shoe.png")

shoe_img = Image.open('img/shoe.png').convert('L') # open into greyscale, or L mode

shoe_arr = np.expand_dims(np.array(shoe_img), 0)

out_softmax = model.predict(x=shoe_arr / 255.)
print("out_softmax", out_softmax) # is already a probability distribution (softmax)

out_softmax = tf.keras.activations.softmax(tf.convert_to_tensor(out_softmax))

# Get predicted class as the index corresponding to the maximum value in the vector probability
predicted_class = tf.argmax(out_softmax, 1)
print("predicted_class", predicted_class)  # predictions as tensors