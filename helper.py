# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:02:31 2019

@author: tempo
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


def int_shape(x):
    shape = x.get_shape()
    if shape[0]:
        return [int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])]
    else: 
        return [None, int(shape[1]), int(shape[2]), int(shape[3])]


def split_along_channels(inputs):
    dims = int_shape(inputs)
    channels = dims[-1]
    assert channels % 2 == 0
    # split along channels
    c_half = channels // 2
    return inputs[:, :, :, :c_half], inputs[:, :, :, c_half:]
    
def actn_init(batch):
    dims = int_shape(batch)
    def scale_init(shape, dtype=None):
        return tf.keras.backend.random_normal(shape, dtype=dtype)
    
    def bias_init(shape, dtype=None):
        return tf.keras.backend.random_normal(shape, dtype=dtype)
        
    #s = tf.reshape(self.scale, [dims[0], 1, 1, dims[3]])
    #b = tf.reshape(self.bias, [dims[0], 1, 1, dims[3]])
    return scale_init, bias_init

def preprocess(train_data, discrete_vals=256):
    x = tf.cast(train_data, 'float32')
    x = x / discrete_vals 
    x += tf.random.uniform(x.shape, 0, 1. / discrete_vals)
    return x
    
def postprocess(z, discrete_vals=256):
    return tf.cast(tf.clip_by_value(tf.floor(z*discrete_vals), 0, discrete_vals-1), 'uint8')
    

def sampleplot(img):
    img = tf.reshape(img, (28,28)).numpy()
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