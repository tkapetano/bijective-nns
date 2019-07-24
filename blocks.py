# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:38:15 2019

@author: tkapetano

Collection of blocks and classifiers:
    - FlowstepACN
    - ClassifierInv
    - ClassifierBigInv
    - FlowstepBN
    - ClassifierBN

"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2
from helper import split_along_channels, int_shape


class FlowstepACN(layers.Layer):
    def __init__(self, ml=True, data_init=None, **kwargs):
        super(FlowstepACN, self).__init__(**kwargs)
        self.actn = Actnorm(ml, data_init)
        self.conv1x1 = Conv1x1(ml)
        self.coupling = CouplingLayer2(ml)
               
    def call(self, inputs):
        y = self.actn(inputs)
        y = self.conv1x1(y)
        return self.coupling(y)
        
    def invert(self, outputs):
        x = self.coupling.invert(outputs)
        x = self.conv1x1.invert(x) 
        return self.actn.invert(x)
        
               
       
class ClassifierInv(tf.keras.Model):
    def __init__(self, label_classes, ml=False, data_init=None, **kwargs):
        super(ClassifierInv, self).__init__(**kwargs)
        self.squeeze = Squeeze()
        self.flow_1 = FlowstepACN(ml, data_init)
        self.flow_2 = FlowstepACN(ml, data_init)
        self.split = split_along_channels
        self.flow_3 = FlowstepACN(ml, data_init)
        self.flat_first = layers.Flatten()
        self.flat_second = layers.Flatten()
        self.flat_last = layers.Flatten()
        self.dense = layers.Dense(label_classes)
          
    def call(self, inputs, training=None):
        y = self.squeeze(inputs)
        y = self.flow_1(y)
        y = self.flow_2(y)
        y_a, y_b = self.split(y)
        #y_b = self.flat_first(y_b)
        y = self.squeeze(y_a)
        y = self.flow_3(y)
        y_aa, y_bb = self.split(y)
        #y_bb = self.flat_second(y_bb)
        y = self.flat_last(y_aa)
        y = self.dense(y)
        return tf.nn.softmax(y)
        
                  
    def compute_z(self, inputs):
        y = self.squeeze(inputs)
        y = self.flow_1(y)
        y = self.flow_2(y)
        y_a, y_b = self.split(y)
        y = self.squeeze(y_a)
        y = self.flow_3(y)
        y_aa, y_bb = self.split(y)
        y = tf.concat((y_aa, y_bb), axis=3)
        y = tf.reshape(y, y_b.get_shape())
        return tf.concat((y, y_b), axis=3)
      
      
    def invert(self, z):    
        shape = int_shape(z)
        channels = shape[3]
        assert channels % 2 == 0 and shape[1] % 2 == 0 and shape[2] % 2 == 0
        z_1, z_2 = split_along_channels(z)
        z_1 = tf.reshape(z_1, [shape[0], shape[1]//2, shape[2]//2, channels*2])
        x_1 = self.flow_3.invert(z_1)
        x_1 = self.squeeze.invert(x_1)
        x = tf.concat((x_1, z_2), axis=3)
        x = self.flow_2.invert(x)
        x = self.flow_1.invert(x)
        x = self.squeeze.invert(x)
        return x
    
    
class ClassifierBigInv(tf.keras.Model):
    def __init__(self, label_classes, ml=False, data_init=None, **kwargs):
        super(ClassifierBigInv, self).__init__(**kwargs)
        self.squeeze = Squeeze()
        self.flow_1 = FlowstepACN(ml, data_init)
        self.flow_2 = FlowstepACN(ml, data_init)
        self.flow_3 = FlowstepACN(ml, data_init)
        self.split = split_along_channels
        self.flow_4 = FlowstepACN(ml, data_init)
        self.flow_5 = FlowstepACN(ml, data_init)
        self.flow_6 = FlowstepACN(ml, data_init)
        self.flat = layers.Flatten()
        self.dense = layers.Dense(label_classes)
          
    def call(self, inputs):
        y = self.squeeze(inputs)
        y = self.flow_1(y)
        y = self.flow_2(y)
        y = self.flow_3(y)
        y_a, _ = self.split(y)
        y = self.squeeze(y_a)
        y = self.flow_4(y)
        y = self.flow_5(y)
        y = self.flow_6(y)
        y_aa, _ = self.split(y)
        y = self.flat(y_aa)
        y = self.dense(y)
        return tf.nn.softmax(y)
        
    def compute_z(self, inputs):
        y = self.squeeze(inputs)
        y = self.flow_1(y)
        y = self.flow_2(y)
        y = self.flow_3(y)
        y_a, y_b = self.split(y)
        y = self.squeeze(y_a)
        y = self.flow_4(y)
        y = self.flow_5(y)
        y = self.flow_6(y)
        y_aa, y_bb = self.split(y)
        y = tf.concat((y_aa, y_bb), axis=3)
        y = tf.reshape(y, y_b.get_shape())
        return tf.concat((y, y_b), axis=3)
          
    def invert(self, z):    
        shape = int_shape(z)
        channels = shape[3]
        assert channels % 2 == 0 and shape[1] % 2 == 0 and shape[2] % 2 == 0
        z_1, z_2 = split_along_channels(z)
        z_1 = tf.reshape(z_1, [shape[0], shape[1]//2, shape[2]//2, channels*2])
        x_1 = self.flow_6.invert(z_1)
        x_1 = self.flow_5.invert(x_1)
        x_1 = self.flow_4.invert(x_1)
        x_1 = self.squeeze.invert(x_1)
        x = tf.concat((x_1, z_2), axis=3)
        x = self.flow_3.invert(x)
        x = self.flow_2.invert(x)
        x = self.flow_1.invert(x)
        x = self.squeeze.invert(x)
        return x
    
class FlowstepBN(layers.Layer):
    def __init__(self, input_shape, ml=False, **kwargs):
        super(FlowstepBN, self).__init__(**kwargs)
        self.bn = layers.BatchNormalization()
        self.conv1x1 = Conv1x1(ml)
        self.coupling = CouplingLayer2(ml)
                
    def call(self, inputs):
        y = self.bn(inputs)
        y = self.conv1x1(y)
        return self.coupling(y)
        
        
class ClassifierBN(tf.keras.Model):
    def __init__(self, label_classes, ml=False, **kwargs):
        super(ClassifierBN, self).__init__(**kwargs)
        self.squeeze = Squeeze()
        self.flow_1 = FlowstepBN(ml)
        self.flow_2 = FlowstepBN(ml)
        self.split = split_along_channels
        self.flow_3 = FlowstepBN(ml)
        self.flat = layers.Flatten()
        self.dense = layers.Dense(label_classes)
    
        
    def call(self, inputs):
        y = self.squeeze(inputs)
        y = self.flow_1(y)
        y = self.flow_2(y)
        y_a, _ = self.split(y)
        y = self.squeeze(y_a)
        y = self.flow_3(y)
        y_aa, _ = self.split(y)
        y = self.flat(y_aa)
        y = self.dense(y)
        return tf.nn.softmax(y)
        
