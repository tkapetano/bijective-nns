# -*- coding: utf-8 -*-
"""
@author: tkapetano

Collection of helper functions:
    - int_shape
    - split_along_channels
    - GaussianIsotrop
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np


def int_shape(x):
    """Transforms tensor shape to int array of length 4, 
    having a batch/sample dimension is recommended
    """
    return list(x.get_shape())


def split_along_channels(inputs):
    """Splits a tensor with even number of channels into two tensors with
        half the channel dimension."""
    dims = int_shape(inputs)
    channels = dims[-1]
    assert channels % 2 == 0
    c_half = channels // 2
    return inputs[:, :, :, :c_half], inputs[:, :, :, c_half:]

  
class GaussianIsotrop(object):
    """An Gaussian isotropic distribution that can be used for sampling and 
        calculating the log density.
    """
    def __init__(self, mean, log_var):
        self.mean = mean
        self.log_var = log_var
    
    def logp(self, x):
        log2pi = tf.math.log(2 * np.pi)
        logp_val = -0.5 * ((x - self.mean) ** 2. * tf.exp(-self.log_var) + self.log_var + log2pi)
        return tf.reduce_sum(logp_val, axis=[1,2,3])
    
    def eps_recon(self, x):
        return (x - self.mean) / tf.exp(self.log_var)
    
    def sample(self, eps=None):
        if eps is not None:
            eps = tf.keras.backend.random_normal(shape=self.mean.get_shape())
        return self.mean + tf.exp(0.5 * self.log_var) * eps            
        
   
