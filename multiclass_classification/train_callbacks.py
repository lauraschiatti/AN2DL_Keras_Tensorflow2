# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
from datetime import datetime
import tensorflow as tf

exp_name = ""
def set_which_model(which_model):
    global exp_name
    exp_name = which_model

# Callbacks array
callbacks = []

# Create experiments folder
# ------------------------

# get current working directory
cwd = os.getcwd()

exps_dir = os.path.join(cwd, 'overfitting_experiments')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

exp_dir = os.path.join(exps_dir, exp_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)


# Model checkpoint
# ----------------

ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Save the model after every epoch.
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'ckpt_{epoch:02d}'), # path to save the model file
                                                   save_weights_only=True)  # only the model's weights will be saved

callbacks.append(ckpt_callback)


# Visualize Learning on Tensorboard
# ---------------------------------

tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0, # disable profiling
                                             # frequency (in epochs) at which to compute activation and
                                             # weight histograms for the layers of the model
                                             histogram_freq=1)  # if 1 shows weights histograms

callbacks.append(tb_callback)

# How to visualize Tensorboard
# 1. tensorboard --logdir <EXPERIMENTS_DIR> --port <PORT>     <- from terminal <PORT> 6060
# 2. localhost:<PORT>   <- in your browser


# Early Stopping
# --------------
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks.append(es_callback)
