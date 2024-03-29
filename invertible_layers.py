# -*- coding: utf-8 -*-
"""
@author: tkapetano

Collection of invertable layer architectures:
    - Squeeze
    - Actnorm
    - Conv1x1
    - CouplingLayer2
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from helper import int_shape, split_along_channels, GaussianIsotrop, \
                    LogisticDist, lu_decomposition, DetOne, \
                    LowerTriangularlWeights, UpperTriangularlWeights

DTYPE = 'float32'


class Squeeze(tf.keras.layers.Layer):
    """Squeeze layer transformation trades spatial dimensions for a greater number of channels. 
    No trainable parameters.
    # Input shape: Requires even spatial dimensions. 
    # Output shape: If input is a h x w x c tensor, output shape is h/2 x w/2 x 4*c.
    """
    def __init__(self, name='squeeze', factor=2, **kwargs):
        super(Squeeze, self).__init__(name=name, **kwargs)
        self.factor = factor
                                               
    def call(self, inputs):
        h, w, c = int_shape(inputs)[-3:]
        assert h % self.factor == 0 and w % self.factor == 0
        y = tf.reshape(inputs, [-1, h//self.factor, self.factor, w//self.factor, self.factor, c])
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, [-1, h//self.factor, w//self.factor, self.factor*self.factor*c])
        return y
      
    def invert(self, outputs):
        h, w, c = int_shape(outputs)[-3:]
        square = self.factor*self.factor
        assert c >= square and c % square == 0
        x = tf.reshape(outputs, [-1, h, w, c//square, self.factor, self.factor])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, [-1, self.factor*h, self.factor*w, c//square])
        return x
        
    def compute_output_shape(self, input_shape):
        return (int(input_shape[-3]/self.factor), 
                int(input_shape[-2]/self.factor), 
                self.factor*self.factor*input_shape[-1])
                
class FlipChannels(tf.keras.layers.Layer):
    """Squeeze layer transformation trades spatial dimensions for a greater number of channels. 
    No trainable parameters.
    # Input shape: Requires even spatial dimensions. 
    # Output shape: If input is a h x w x c tensor, output shape is h/2 x w/2 x 4*c.
    """
    def __init__(self, name='flipchannel',  **kwargs):
        super(FlipChannels, self).__init__(name=name, **kwargs)
                                               
    def call(self, inputs):
        x_a, x_b = split_along_channels(inputs)
        return tf.concat([x_b, x_a], axis=-1)
      
    def invert(self, outputs):
        x_a, x_b = split_along_channels(outputs)
        return tf.concat([x_b, x_a], axis=-1)
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
class Actnorm(tf.keras.layers.Layer):
    """Activation normalization layer uses an affine channelwise transformation to 
    standardize mean and variance of activations, independent of batch sizes
    used during training.
    # Arguments: data_int: pair of initializer functions - first entry for scale,
    second for bias.
    # Output shape: Same shape as input.
    """
    def __init__(self, name='actnorm', ml=True, **kwargs):
        super(Actnorm, self).__init__(name=name, **kwargs)
        self.ml = ml
        
    def build(self, input_shape):
        channels = input_shape[-1]     
        self.scale = self.add_weight(name='scale',
                                    shape=(1, 1, channels),
                                    initializer='ones',
                                    trainable=True,
                                    dtype=DTYPE)
        self.bias = self.add_weight(name='bias',
                                    shape=(1, 1, channels),
                                    initializer='zeros',
                                    trainable=True,
                                    dtype=DTYPE)
        super(Actnorm, self).build(input_shape)
        
    def call(self, inputs):
        if self.ml: 
            w, h = int_shape(inputs)[-3:-1]
            log_det =  w * h * tf.reduce_sum(tf.math.log(tf.math.abs(self.scale)))
            self.add_loss(log_det)
        return inputs * self.scale + self.bias
        
    def invert(self, outputs):
        return (outputs - self.bias) / self.scale 
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
        
        
class Conv1x1(tf.keras.layers.Layer):
    """Invertible 1x1 convolutional layer transforms a generalized. learnable
    permutation of channels. Prepares flow for a coupling layer. 
    (cf. Kingma and Dhariwal, 2018)
    Can be easily made a simple permutation of channels by setting the 
    layer to non-trainable - use ml=False to save computations as log_det 
    is zero in this case. 
    # Output shape:  Same shape as input.
    """
    def __init__(self, name='conv1x1', ml=True, lu_decom=False, **kwargs):
        super(Conv1x1, self).__init__(name=name, **kwargs)
        self.ml = ml
        if lu_decom:
            self.ml = False
        self.lu = lu_decom
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.lu:
            self.w_mat = self.add_weight(name='w_mat',
                                 shape=(self.channels, self.channels),
                                 initializer='orthogonal',
                                 trainable=False,
                                 dtype=DTYPE)
            p, l, u, s = lu_decomposition(self.w_mat)
            self.p_mat = self.add_weight(name='p_mat',
                                 shape=p.get_shape(),
                                 initializer=tf.keras.initializers.Constant(value=p.numpy()),
                                 trainable=False,
                                 dtype=DTYPE)
            self.l_mat = self.add_weight(name='l_mat',
                                 shape=l.get_shape(),
                                 initializer=tf.keras.initializers.Constant(value=l.numpy()),
                                 trainable=True,
                                 constraint=LowerTriangularlWeights(self.channels),
                                 dtype=DTYPE)
            self.u_mat = self.add_weight(name='u_mat',
                                 shape=u.get_shape(),
                                 initializer=tf.keras.initializers.Constant(value=u.numpy()),
                                 trainable=True,
                                 constraint=UpperTriangularlWeights(self.channels),
                                 dtype=DTYPE)
            self.s = self.add_weight(name='s',
                                 shape=s.get_shape(),
                                 initializer=tf.keras.initializers.Constant(value=s.numpy()),
                                 trainable=True,
                                 constraint=DetOne(self.channels),
                                 dtype=DTYPE)
        else:
            self.w_mat = self.add_weight(name='w_mat',
                             shape=(self.channels, self.channels),
                             initializer='orthogonal',
                             trainable=True,
                             #constraint=tf.keras.constraints.UnitNorm(axis=[0,1]),
                             dtype=DTYPE)
                                 
        super(Conv1x1, self).build(input_shape)
                                               
    def call(self, inputs):
        if self.ml:
            w, h = int_shape(inputs)[-3:-1]
            if self.lu:
                log_det = w * h * tf.math.log(tf.math.abs(tf.reduce_prod(self.s)) + 1e-10)
            else:
                log_det = w * h * tf.math.log(tf.math.abs(tf.linalg.det(self.w_mat)))
            self.add_loss(log_det)
        if self.lu:
            u_plus =  tf.linalg.set_diag(self.u_mat, self.s)
            w_mat = tf.matmul(self.p_mat, tf.matmul(self.l_mat,  u_plus))
            #print(tf.math.abs(tf.linalg.det(w_mat)))
        else:
            w_mat = self.w_mat
        w_filter = tf.reshape(w_mat, [1,1, self.channels, self.channels])
        return tf.nn.conv2d(inputs, w_filter, [1,1,1,1], 'SAME')
               
    def invert(self, outputs):
        if self.lu:
            u_plus =  tf.linalg.set_diag(self.u_mat, self.s)
            w_mat = tf.matmul(self.p_mat, tf.matmul(self.l_mat,  u_plus))
        else:
            w_mat = self.w_mat
        w_inv = tf.linalg.inv(w_mat)
        w_filter = tf.reshape(w_inv, [1,1, self.channels, self.channels])
        return tf.nn.conv2d(outputs, w_filter, [1,1,1,1], 'SAME')
        
    def compute_output_shape(self, input_shape):
        return input_shape
        

class CouplingLayer(tf.keras.layers.Layer):
    """Affine Coupling layer (Dinh, Krueger, Bengio 2015).
    # Input shape:  Requieres an even number of channels. Recommended to have 
                    a permuation of the channels preceeding this layer.
    # Output shape: Same shape as input.
    """
    def __init__(self, name='coupling', ml=True, kernel_size=2, filters=(128,128), **kwargs):
        super(CouplingLayer, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.kernel_size = kernel_size
        self.filters = filters
        
        
    def build(self, input_shape):        
        channels = input_shape[-1]
        self.conv1 = tf.keras.layers.Conv2D(self.filters[0], 
                                   self.kernel_size,
                                   name='conv1',
                                   padding='same', 
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   dtype=DTYPE)
        self.conv2 = tf.keras.layers.Conv2D(self.filters[1], 
                                   self.kernel_size, 
                                   name='conv2',
                                   padding='same', 
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   dtype=DTYPE)
        self.conv3 = tf.keras.layers.Conv2D(channels, 
                                   self.kernel_size, 
                                   name='conv3',
                                   padding='same', 
                                   kernel_initializer='zeros', 
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                   dtype=DTYPE)
        super(CouplingLayer, self).build(input_shape)
                                               
                                               
    def call(self, inputs):
        x_a, x_b = split_along_channels(inputs)
        # apply the neural net to first partition component to get scaling 
        # and translation parameters
        intermediate = self.conv3(self.conv2(self.conv1(x_a)))
        bias = intermediate[:, :, :, 0::2]
        s = intermediate[:, :, :, 1::2]
        scale = tf.nn.sigmoid(s + 2.) + 1e-10
        y_b = x_b * scale + bias
        if self.ml:
            # add loss for max likelihood term
            log_det = tf.reduce_sum(tf.math.log(scale), axis=[1,2,3])
            self.add_loss(tf.reduce_mean(log_det))
        return tf.concat([x_a,y_b], axis=3)
        
        
    def invert(self, outputs):
        # split along channels
        y_a, y_b = split_along_channels(outputs)
        # apply nn
        intermediate = self.conv3(self.conv2(self.conv1(y_a)))
        t = intermediate[:, :, :, 0::2]
        s = intermediate[:, :, :, 1::2]
        # backward pass
        #scale = tf.math.exp(s)
        scale = tf.nn.sigmoid(s + 2.) + 1e-10
        x_b = (y_b - t) / scale
        return tf.concat([y_a,x_b], axis=3)
        
    def compute_output_shape(self, input_shape):
        return input_shape

        
class SplitLayer(tf.keras.layers.Layer):
    """Splits inputs channelwise, computes mean and log variance of first half
    and likelihood of the resulting gaussian density with second half as argument.
    # Input shape:  Requieres an even number of channels. 
    # Output shape: Three elements of input shape with half the number of channels
    """
    def __init__(self, name='split', ml=True, kernel_size=2, use_gauss=True, **kwargs):
        super(SplitLayer, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.kernel_size = kernel_size
        self.use_gauss = use_gauss
        
        
    def build(self, input_shape):        
        channels = input_shape[-1]
        self.conv = tf.keras.layers.Conv2D(channels, 
                                   self.kernel_size, 
                                   name='conv',
                                   padding='same', 
                                   kernel_initializer='zeros',
                                   dtype=DTYPE)
        super(SplitLayer, self).build(input_shape)
                                               
    def call(self, inputs):
        x_a, x_b = split_along_channels(inputs)
        # apply the neural net to first partition component to get scaling 
        # and translation parameters
        h = self.conv(x_a)
        mean = h[:, :, :, 0::2]
        log_std = h[:, :, :, 1::2]
        if self.use_gauss:
            dist = GaussianIsotrop(mean, log_std)
        else:
            dist = LogisticDist(mean, log_std)
        if self.ml:
            # add loss for max likelihood term
            log_det = dist.logp(x_b)
            log_det = tf.reduce_mean(log_det)
            self.add_loss(log_det)
        return x_a, x_b, dist.eps_recon(x_b)
        
        
    def invert(self, y_a, y_b, sample=False, eps=None):
        if sample:
            h = self.conv(y_a)
            mean = h[:, :, :, 0::2]
            log_std = h[:, :, :, 1::2]
            if self.use_gauss:
                dist = GaussianIsotrop(mean, log_std)
            else:
                dist = LogisticDist(mean, log_std)
            y_b = dist.sample(eps)
        return tf.concat([y_a, y_b], axis=3)
        
    def compute_output_shape(self, input_shape):
        input_shape[-1] /= 2
        return [int(input_shape)] * 3
        
