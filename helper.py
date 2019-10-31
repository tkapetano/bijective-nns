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
    """A Gaussian isotropic distribution that can be used for sampling and 
        calculating the log density.
    """
    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std
    
    def logp(self, x):
        log2pi = tf.math.log(2 * np.pi)
        logp_val = -0.5 * ((x - self.mean) ** 2. * tf.exp(-2. * self.log_std) 
                    + 2. * self.log_std + log2pi + 1e-10)
        return tf.reduce_sum(logp_val, axis=[1,2,3])
    
    def eps_recon(self, x):
        return (x - self.mean) / tf.exp(self.log_std)
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.keras.backend.random_normal(shape=self.mean.get_shape())
        return self.mean + tf.exp(self.log_std) * eps            
        
class LogisticDist(object):
    """A discretized logistic distribution that can be used for sampling and 
        calculating the log density.
    """
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale
        
    def logp(self, x):
        #x = (x - self.mean) / (2.0 * self.scale)
        #logp_val = - tf.math.log( self.scale + 1e-10) - tf.math.log(tf.math.exp(x) + tf.math.exp(-x) + 1e-10)
        logp_val = - tf.math.log(1 + tf.math.exp(x + 1e-10)) - tf.math.log(1 + tf.math.exp(-x + 1e-10))        
        return tf.reduce_sum(logp_val, axis=[1,2,3])
        
    def eps_recon(self, x):
        return (x - self.mean) / self.scale
    
    def sample(self, eps=None):
        if eps is None:
            eps_unif = tf.keras.backend.random_uniform(shape=self.mean.get_shape())
            eps = tf.math.log(eps_unif) - tf.math.log(1. - eps_unif)
        return self.mean + self.scale * eps            


def lu_decomposition(w_mat):
    shape = int_shape(w_mat)
    lu, p_inv = tf.linalg.lu(w_mat)
    p_mat = np.zeros(shape=shape, dtype='float32')
    for num, index in enumerate(p_inv.numpy()):
        p_mat[index, num] = 1
    p_mat = tf.constant(p_mat)    
    l_operator = tf.linalg.LinearOperatorLowerTriangular(lu)
    diag = l_operator.diag_part() 
    l_mat = l_operator.to_dense()
    
    u_mat = lu - l_mat       
    #l_mat = tf.linalg.set_diag(l_mat, tf.ones(shape=(shape[0],)))    
    #u_mat = tf.linalg.set_diag(u_mat, diag)    
    
    return p_mat, l_mat, u_mat, diag


class DetOne(tf.keras.constraints.Constraint):
    def __init__(self, channel_dim):
        self.channel_dim = channel_dim
        
    def __call__(self, s):
        # compute geometric mean for rescaling
        normalizer = tf.math.abs(tf.reduce_prod(s))
        #print('Normal: ' + str(normalizer))
        if normalizer == 0:
            scale = 1.
        else:
            scale = tf.math.pow(normalizer, 1. / float(self.channel_dim))
        #print('Scale: '+ str(scale))
        return s / scale
        
class LowerTriangularlWeights(tf.keras.constraints.Constraint):
    """Constrains the weights to be lower triangular, with ones on the diagonal.
    """
    def __init__(self, channel_dim):
        self.channel_dim = channel_dim
        
    def __call__(self, w):
        operator = tf.linalg.LinearOperatorLowerTriangular(w)
        w = operator.to_dense()
        return tf.linalg.set_diag(w, tf.ones(shape=(self.channel_dim,)))    
        
class UpperTriangularlWeights(tf.keras.constraints.Constraint):
    """Constrains the weights to be lower triangular, with ones on the diagonal.
    """
    def __init__(self, channel_dim):
        self.channel_dim = channel_dim
        
    def __call__(self, w):
        if tf.__version__ == '2.0.0-alpha0':
            transpose = tf.linalg.transpose
        else:
            transpose = tf.linalg.matrix_transpose
        w = transpose(w)
        operator = tf.linalg.LinearOperatorLowerTriangular(w)
        w = operator.to_dense()
        w = transpose(w)
        return tf.linalg.set_diag(w, tf.zeros(shape=(self.channel_dim,)))    
        