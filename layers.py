# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:30:49 2019

@author: tempo
"""

# the layers / components of Glow

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from helper import int_shape, split_along_channels

class Squeeze(tf.keras.layers.Layer):
    """Squeeze layer (Shi et. al. 2016).
    Simple transformation that trades spatial dimensions for a greater number of channels. 
    No trainable parameters.
    # Input shape: Needs to have even spatial dimensions. Use the keyword 
        argument `input_shape` (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape: If input is a b x h x w x c tensor, output is b x h/2 x w/2 x 4*c.
    """
    def __init__(self, factor=2, name='squeeze', **kwargs):
        super(Squeeze, self).__init__(name=name, **kwargs)
        self.factor = factor
                                               
    def call(self, inputs):
        shape = inputs.get_shape()
        h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
        assert h % self.factor == 0 and w % self.factor == 0
        y = tf.reshape(inputs, [-1, h//self.factor, self.factor, w//self.factor, self.factor, c])
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, [-1, h//self.factor, w //self.factor, self.factor*self.factor*c])
        return y
      
    def invert(self, outputs):
        shape = outputs.get_shape()
        h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
        square = self.factor*self.factor
        assert c >= square and c % square == 0
        x = tf.reshape(outputs, [-1, h, w, c//square, self.factor, self.factor])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, [-1, self.factor*h, self.factor*w, c//square])
        return x


class Actnorm(layers.Layer):
    """Activation normalization layer (Kingma and Dhariwal, 2018).
    Use a affine transformation to standardize mean and variance
    # Arguments: - scale: If True, multiply by scale vector s
                 - bias: If True, add bias vector b
    # Input shape: Arbitrary. 
    # Output shape: Same shape as input.
    """
    def __init__(self, ml=True, data_depent_init=None, name='actnorm', **kwargs):
        super(Actnorm, self).__init__(name=name, **kwargs) #dynamic=True, **kwargs)
        self.ml = ml
        self.data_depent_init = data_depent_init
 
        
    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        if self.data_depent_init:
             scale_init, bias_init = self.data_depent_init()
        else:
            scale_init, bias_init = 'random_normal', 'random_normal'
            
        self.scale = self.add_weight(shape=(self.channels,),
                               initializer=scale_init,
                               trainable=True)
        self.bias = self.add_weight(shape=(self.channels,),
                                initializer=bias_init,
                                trainable=True)
                                               
    def call(self, inputs):
        dims = int_shape(inputs)
        if self.ml: 
            # add loss for max likelihood term
            log_det = - dims[1] * dims[2] * tf.reduce_sum(tf.math.log(tf.math.abs(self.scale)))
            #print('this is actnorm:' + str(log_det))
            self.add_loss(log_det)
        # forward pass - channelwise ops
        s = tf.reshape(self.scale, [1, 1, self.channels])
        b = tf.reshape(self.bias, [1, 1, self.channels])
        #print(s)
        #print(b)
        return inputs * s + b
        
        
    def invert(self, outputs):
        s = tf.reshape(self.scale, [1, 1, self.channels])
        b = tf.reshape(self.bias, [1, 1, self.channels])
        return (outputs - b) / s  
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = {
            'act_scale': self.scale,
            'act_bias': self.bias,
            'act_channels': self.channels
        }
        base_config = super(Actnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
#ml = True # the most general setting
#squeeze = Squeeze()
#actn = Actnorm(ml)      
#inputs = tf.ones([4, 2, 2, 2])
#actnormed = actn(inputs)
#print(actn.losses)      
      
        
class Conv1x1(layers.Layer):
    """Invertible 1x1 convolution layer (Kingma and Dhariwal, 2018).
    A generalized permutation of channels as preparation for a coupling layer.
    # Input shape:  Arbitrary. 
    # Output shape:  Same shape as input.
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
        #print('Inverting ..')
        w_inv = tf.linalg.inv(self.w)
        #print('Entry {}, and {}'.format(w_inv[0][0], w_inv[1][0]))
        w_filter = tf.reshape(w_inv, [1,1, self.channels, self.channels])
        return tf.nn.conv2d(outputs, w_filter, [1,1,1,1], 'SAME')
        
    def get_config(self):
        config = {
            'conv1x1_w': self.w,
            'conv1x1_channels': self.channels
        }
        base_config = super(Conv1x1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class CouplingLayer2(layers.Layer):
    """Affine Coupling layer (Dinh, Krueger, Bengio 2015).
    A generalized permutation of channels as preparation for a coupling layer
    # Arguments
        nn: A shallow neural network. In- and output shape need to be equal.
    # Input shape:  Needs to have an even number of channels. Recommended to have 
            some permuation of the channels preceeding this layer, e.g. 1x1 Convolution
    # Output shape: Same shape as input.
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
        #print(inputs.get_shape())
        x_a, x_b = split_along_channels(inputs)
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
        #scale = tf.math.exp(s)
        scale = tf.nn.sigmoid(s + 2.)
        y_b = x_b * scale + t
        return tf.concat([x_a,y_b], 3)
        
        
    def invert(self, outputs):
        # split along channels
        y_a, y_b = split_along_channels(outputs)
        # apply nn
        intermediate = self.conv3(self.conv2(self.conv1(y_a)))
        t = intermediate[:, :, :, 0::2]
        s = intermediate[:, :, :, 1::2]
        # backward pass
        #scale = tf.math.exp(s)
        scale = tf.nn.sigmoid(s + 2.)
        x_b = (y_b - t) / scale
        return tf.concat([y_a,x_b], 3)
        
    def get_config(self):
        config = {
        }
        base_config = super(CouplingLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

#ml = True # the most general setting
#coup = CouplingLayer2(ml)    
#inputs = tf.ones([4, 2, 2, 2])
#actnormed = coup(inputs)
#print(coup.losses)      
