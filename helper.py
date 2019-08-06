# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:02:31 2019

@author: tkapetano

Collection of helper functions:
    - int_shape
    - split_along_channels
    - reinstantiate_with_data_init
    - preprocess
    - postprocess
    - sampleplot
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
  def __init__(self, scale):
    self.scale = scale
    
  def __call__(self, shape=None, dtype=None, partition_info=None):
    return self.scale


class Bias_init(tf.keras.initializers.Initializer):
  def __init__(self, bias):
      self.bias = bias

  def __call__(self, shape=None, dtype=None, partition_info=None):
    return self.bias


def _data_init_acn(model, batch):
    """Generates a list of Initializer pairs (scale, bias) from a single data
    batch, such that on this batch the effective activations of each ACN layer
    after being initialized with this list have mean 0 and stddev 1."""
    initializers = []
    current_activations = batch
    layers = model.layers
    current_layer = 0
    
    while not('flatten' in layers[current_layer].name):
      if 'acn' in layers[current_layer].name:
          out_vals = current_activations
      elif 'squeeze' in layers[current_layer].name:
          squeeze = layers[current_layer]
          out_vals = squeeze(current_activations)
          current_layer += 1

      # normalize
      shape = int_shape(out_vals)
      scale = tf.math.reduce_std(out_vals, axis=(0,1,2))
      scale = tf.reshape(scale, [1, 1, shape[3]])
      scale = tf.math.reciprocal(scale)
      bias = tf.math.reduce_mean(out_vals, axis=(0,1,2))
      bias = tf.reshape(bias, [1, 1, shape[3]])
      bias = -bias * scale 
      # add to list  
      initializers.append((Scale_init(scale), Bias_init(bias)))

      current_activations = layers[current_layer](out_vals)
      current_layer += 1
 
    return initializers 

 
def reinstantiate_with_data_init(ModelClass, label_num, batch, ml=False):
    model = ModelClass(label_num, ml)
    list_of_inits = _data_init_acn(model, batch)
    weight_list = []
    acn_count = 0

    for layer in model.layers:
      if layer.trainable_variables:
        weights = layer.get_weights()
        if 'acn' in layer.name:
          weights[0] = list_of_inits[acn_count][0]()
          weights[1] = list_of_inits[acn_count][1]()
          acn_count += 1

        weight_list.append(weights)
        #print(weight_list)
      else:
        weight_list.append(0)
    model_reinst = ModelClass(label_num, ml)
    
    for i in range(len(weight_list)):
       if model_reinst.layers[i].trainable_variables:
          model_reinst.layers[i].set_weights(weight_list[i])
    
    return model_reinst

    
def preprocess(train_data, discrete_vals=256):
    """Maps discrete data to floats in interval [0,1]"""
    x = tf.cast(train_data, 'float32')
    x = x / discrete_vals 
    x += tf.random.uniform(x.shape, 0, 1. / discrete_vals)
    return x
    
def postprocess(z, discrete_vals=256):
    """Maps floats to discrete vals, inverts preprocessing"""
    return tf.cast(tf.clip_by_value(tf.floor(z*discrete_vals), 0, discrete_vals-1), 'uint8')
    

def sampleplot(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()
   
