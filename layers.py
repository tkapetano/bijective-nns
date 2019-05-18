# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:57:47 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt



class Squeeze(layers.Layer):
    """Squeeze layer (Shi 2014).
    Simple transformation that trades spatial dimensions for a greater number of channels. 
    No trainable parameters.
    # Arguments
        
    # Input shape
        Needs to have even spatial dimensions. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        If input is a b x h x w x c tensor, output is b x h/2 x w/2 x 4*c.
    # References
        -
    """
    def __init__(self, name='squeeze', **kwargs):
        super(Squeeze, self).__init__(name=name, **kwargs)
                                               
    def call(self, inputs):
        shape = inputs.get_shape()
        h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
        assert h % 2 == 0 and w % 2 == 0
        y = tf.reshape(inputs, [-1, h//2, 2, w//2, 2, c])
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, [-1, h//2, w //2, 4*c])
        return y
      
    def invert(self, outputs):
        shape = outputs.get_shape()
        h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
        assert c >= 4 and c % 4 == 0
        x = tf.reshape(outputs, [-1, h, w, c//4, 2, 2])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, [-1, 2*h, 2*w, c//4])
        return x
        

        
class Actnorm(layers.Layer):
    """Activation normalization layer (Kingma and Dharwal, 2014).
    Use a affine transformation to standardize mean and variance
    # Arguments
        scale: If True, multiply by scale vector s
        bias: If True, add bias vector b
    # Input shape
        Arbitrary. 
    # Output shape
        Same shape as input.
    # References
        - 
    """
    def __init__(self, ml=True, name='actnorm', **kwargs):
        super(Actnorm, self).__init__(name=name, **kwargs)
        self.ml = ml
        
    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        self.scale = self.add_weight(shape=(self.channels,),
                               initializer='random_normal',
                               trainable=True)
        self.bias = self.add_weight(shape=(self.channels,),
                                initializer='random_normal',
                                trainable=True)
                                               
    def call(self, inputs):
        if self.ml: 
            # add loss for max likelihood term
            log_det = - tf.reduce_sum(tf.math.log(tf.math.abs(self.scale)))
            #print('this is actnorm:' + str(log_det))
            self.add_loss(log_det)
        # forward pass
        s = tf.reshape(self.scale, (1, 1, self.channels))
        b = tf.reshape(self.bias, (1, 1, self.channels))
        return inputs * s + b
        
        
    def invert(self, outputs):
        s = tf.reshape(self.scale, (1, 1, self.channels))
        b = tf.reshape(self.bias, (1, 1, self.channels))
        return (outputs - b) / s    
        
    def get_config(self):
        config = {
            'act_scale': self.scale,
            'act_bias': self.bias,
            'act_channels': self.channels
        }
        base_config = super(Actnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
 
class Conv1x1(layers.Layer):
    """Invertible 1x1 convolution layer (Kingma and Dharwal, 2014).
    A generalized permutation of channels as preparation for a coupling layer
    # Arguments
    # Input shape
        Arbitrary. 
    # Output shape
        Same shape as input.
    # References
        - 
    """
    def __init__(self, ml=True, name='conv1x1', **kwargs):
        super(Conv1x1, self).__init__(name=name, **kwargs)
        self.ml = ml
        
    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        self.w = self.add_weight(shape=(self.channels, self.channels),
                               initializer='orthogonal',
                               trainable=True)
                                               
    def call(self, inputs):
        if self.ml:
            # add loss for max likelihood term
            w, h = float(inputs.shape[1]), float(inputs.shape[2])
            log_det = -w * h * tf.math.log(tf.math.abs(tf.linalg.det(self.w)))
            #print('this is 1x1conv ' + str(log_det))
            self.add_loss(log_det)
        # forward pass
        w_filter = tf.reshape(self.w, [1,1, self.channels, self.channels])
        return tf.nn.conv2d(inputs, w_filter, [1,1,1,1], 'SAME')
        
        
    def invert(self, outputs):
        w_inv = tf.linalg.inv(self.w)
        w_filter = tf.reshape(w_inv, [1,1, self.channels, self.channels])
        return tf.nn.conv2d(outputs, w_filter, [1,1,1,1], 'SAME')
        
    def get_config(self):
        config = {
            'conv1x1_w': self.w,
            'conv1x1_channels': self.channels
        }
        base_config = super(Conv1x1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

# coupling layer with nn integrated
class CouplingLayer2(layers.Layer):
    """Affine Coupling layer (Dinh.., 2014).
    A generalized permutation of channels as preparation for a coupling layer
    # Arguments
        nn: A shallow neural network. In- and output shape need to be equal.
    # Input shape
        Needs to have an even number of channels. Recommended to have 
        some permuation of the channels preceeding this layer.
    # Output shape
        Same shape as input.
    # References
        - 
    """
    def __init__(self, ml=True, name='conv1x1', **kwargs):
        super(CouplingLayer2, self).__init__(name=name, **kwargs)
        self.ml = ml
        
        
    def build(self, input_shape):        
        width = 2
        channels = int(input_shape[-1])
        self.conv1 = layers.Conv2D(channels, 
                                   width, 
                                   padding='same', 
                                   activation='relu')
        self.conv2 = layers.Conv2D(channels, 
                                   width, 
                                   padding='same', 
                                   activation='relu')
        self.conv3 = layers.Conv2D(channels, 
                                   width, 
                                   padding='same', 
                                   kernel_initializer='zeros')
                                               
    def call(self, inputs):
        # split along channels
        c_half = int(inputs.shape[-1]) // 2
        x_a = inputs[:, :, :, :c_half]
        x_b = inputs[:, :, :, c_half:]
        # apply the neural net to first partition component to get scaling 
        # and translation parameters
        intermediate = self.conv3(self.conv2(self.conv1(x_a)))
        t = intermediate[:, :, :, 0::2]
        s = intermediate[:, :, :, 1::2]
        if self.ml:
            # add loss for max likelihood term
            #log_det = tf.reduce_sum(tf.reduce_sum(s, axis=[1,2,3])).numpy()
            log_det = -sum(tf.reduce_sum(s, axis=[1,2,3]))
            #print('this is coupling ' + str(log_det))
            self.add_loss(log_det)
        # forward pass
        scale = tf.math.exp(s)
        y_b = x_b * scale + t
        return tf.concat([x_a,y_b], 3)
        
        
    def invert(self, outputs):
        # split along channels
        c_half = int(outputs.shape[-1]) // 2
        y_a = outputs[:, :, :, :c_half]
        y_b = outputs[:, :, :, c_half:]
        # apply nn
        intermediate = self.conv3(self.conv2(self.conv1(y_b)))
        t = intermediate[:, :, :, 0::2]
        s = intermediate[:, :, :, 1::2]
        # backward pass
        scale = tf.math.exp(s)
        x_a = (y_a - t) / scale
        return tf.concat([x_a,y_b], 3)
        
    def get_config(self):
        config = {
        }
        base_config = super(CouplingLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Glowblock(layers.Layer):
    def __init__(self, input_shape, ml=True, **kwargs):
        super(Glowblock, self).__init__(**kwargs)
        self.squeeze = Squeeze()
        self.actn = Actnorm()
        self.conv1x1 = Conv1x1(ml)
        self.coupling = CouplingLayer2(ml)
        
        
    def call(self, inputs):
        y = self.squeeze(inputs)
        y = self.actn(y)
        y = self.conv1x1(y)
        y = self.coupling(y)
        return  self.squeeze.invert(y)
        
    def invert(self, outputs):
        x = self.squeeze(outputs)
        x = self.coupling.invert(x)
        x = self.conv1x1.invert(x)
        x = self.actn.invert(x)
        return self.squeeze.invert(x)
