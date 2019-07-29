# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:02:31 2019

@author: tkapetano

Collection of helper functions:
    - int_shape
    - split_along_channels
    - act_init
    - preprocess
    - postprocess
    - sampleplot*
    - invertability*
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


def int_shape(x):
    """Transforms tensor shape to int array of length 4"""
    shape = x.get_shape()
    if shape[0]:
        return [int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])]
    else: 
        return [None, int(shape[1]), int(shape[2]), int(shape[3])]


def split_along_channels(inputs):
    """Splits a tensor with even channel number into to tensor with
        half the channel dim."""
    dims = int_shape(inputs)
    channels = dims[-1]
    assert channels % 2 == 0
    # split along channels
    c_half = channels // 2
    return inputs[:, :, :, :c_half], inputs[:, :, :, c_half:]

    
class Scale_init(tf.keras.initializers.Initializer):
  """Initializer that generates scale and bias tensors from a single data
  batch, such that this batch has all ones and zeros in the activation layer."""
  def __init__(self, scale):
    self.scale = scale
    
  def __call__(self, shape=None, dtype=None, partition_info=None):
    return self.scale


class Bias_init(tf.keras.initializers.Initializer):
  """Initializer that generates scale and bias tensors from a single data
  batch, such that this batch has all ones and zeros in the activation layer."""
  def __init__(self, bias):
      self.bias = bias

  def __call__(self, shape=None, dtype=None, partition_info=None):
    return self.bias


def data_init_acn(model, batch):
    initializer = []
#    for var in model.trainable_variables:
#        if 'actnorm' in var.name:
#            shape = var.shape
#            val = var.numpy()
#            Bias
    layers = model.layers
    squeeze = layers[0]
    squeeze_out = squeeze(batch)
    shape = int_shape(squeeze_out)
    print(shape)
    
    scale = tf.math.reduce_std(squeeze_out, axis=(0,1,2))
    scale = tf.reshape(scale, [1, 1, shape[3]])
    scale = tf.math.reciprocal(scale)
    bias = tf.math.reduce_mean(squeeze_out, axis=(0,1,2))
    bias = tf.reshape(bias, [1, 1, shape[3]])
    bias = -bias * scale 
    
    initializer.append((Scale_init(scale), Bias_init(bias)))
    t_1 = Scale_init(scale)
    t_2 = Bias_init(bias)
    return [(t_1, t_2)]


def preprocess(train_data, discrete_vals=256):
    """Maps discrete data to floats in interval [0,1]"""
    x = tf.cast(train_data, 'float32')
    x = x / discrete_vals 
    x += tf.random.uniform(x.shape, 0, 1. / discrete_vals)
    return x
    
def postprocess(z, discrete_vals=256):
    """Maps floats to discrete vals, inverts preprocessing"""
    return tf.cast(tf.clip_by_value(tf.floor(z*discrete_vals), 0, discrete_vals-1), 'uint8')
    

def sampleplot(img, dims):
    img = tf.reshape(img, dims).numpy()
    img = postprocess(img)
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def invertability(model, img):
    z = model(img)
    img_reconstruct = model.invert(z)
    dist = np.linalg.norm(img - img_reconstruct)
    print(dist)
    
    
#==============================================================================
#     class Scale_init(tf.keras.initializers.Initializer):
#   """Initializer that generates scale and bias tensors from a single data
#   batch, such that this batch has all ones and zeros in the activation layer."""
#   def __init__(self, scale):
#     self.scale = scale
#     
#   def __call__(self, shape=None, dtype=None, partition_info=None):
#     scale = tf.math.reduce_std(self.batch, axis=(0,1,2))
#     s = tf.reshape(scale, [1, 1, self.channels])
#     return tf.math.reciprocal(s)
# 
# 
# class Bias_init(tf.keras.initializers.Initializer):
#   """Initializer that generates scale and bias tensors from a single data
#   batch, such that this batch has all ones and zeros in the activation layer."""
#   def __init__(self, batch):
#     self.batch = batch
#     self.channels = int_shape(batch)[-1]
#     
# 
#   def __call__(self, shape=None, dtype=None, partition_info=None):
#     bias = tf.math.reduce_mean(self.batch, axis=(0,1,2))
#     b = tf.reshape(bias, [1, 1, self.channels])
#     return -b
#==============================================================================
