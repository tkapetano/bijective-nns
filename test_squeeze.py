# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:56:32 2019

@author: tempo
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor,
                       width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width //
                       factor, n_channels*factor*factor])
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor),
                       int(width*factor), int(n_channels/factor**2)))
    return x

# Reverse features across channel dimension


def reverse_features(name, h, reverse=False):
    return h[:, :, :, ::-1]

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

img = train_images[0]
img.shape

#y = squeeze2d(img)
x = tf.ones([1,2,2,1])
x.shape
y = squeeze2d(x)
y.shape
print(y)

x = tf.ones([1,2,2,4])
x.shape
y = squeeze2d(x)
y.shape
print(y)

x = tf.ones([1,4,4,1])
x.shape
y = squeeze2d(x)
y.shape
print(y)

z = x.get_shape()
z[1]
z.numpy()[1:]

a = tf.ones(shape=(3,64))
print(a)
b = tf.ones([3,64])
print(b)
inputs = tf.ones([1,2,2,4])
print(inputs)