# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:50:50 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from layers import Glowblock

class Sampling(layers.Layer):
    """ Uses (z_mean, z_log_var) to sample z, the vector encoding digit."""
    
    def call(self, inputs, temperature=0.7):
        z_mean = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        z_var = tf.ones((batch,dim))
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean +   temperature * tf.math.sqrt(z_var) * epsilon
        
class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
    
    def __init__(self, input_shape, ml=True, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.block_1 = Glowblock(self.ml, input_shape)
        self.block_2 = Glowblock(self.ml, input_shape)
        self.flat = layers.Flatten()
        self.sampling = Sampling()
        
    def call(self, inputs):
        x = self.block_1(inputs)
        z = self.block_2(x)
        #z = self.flat(z)
        z = tf.reshape(z, (inputs.shape[0],784))        
        if self.ml:
            logpz = 0.5 * tf.reduce_sum(z ** 2 + tf.math.log(2 * np.pi), [0,1])
            #print('this is prior ' + str(logpz))
            self.add_loss(logpz)
        return z
        
    def sample(self):
        # std normal
        z_params = tf.zeros((64,784))
        z = self.sampling(z_params)
        z = tf.reshape(z, (64,28,28,1))
        x = self.block_2.invert(z)
        return self.block_1.invert(x)
        
    def sampleplot(self):
        img = self.sample()[0]
        img = tf.reshape(img, (28,28)).numpy()
        img = tf.cast(tf.clip_by_value(tf.floor(img*255), 0, 255), 'uint8')

        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.grid(False)
        plt.show()

