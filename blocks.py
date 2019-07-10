# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:38:15 2019

@author: tkapetano

Collection of blocks and classifiers:
    - FlowstepACN
    - ClassifierACN
    - FlowstepBN
    - ClassifierBN

"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2
from helper import split_along_channels


class FlowstepACN(layers.Layer):
    def __init__(self, ml=True, **kwargs):
        super(FlowstepACN, self).__init__(**kwargs)
        self.actn = Actnorm(ml)
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
        
        

class ClassifierACN(tf.keras.Model):
    def __init__(self, label_classes, ml=False, **kwargs):
        super(ClassifierACN, self).__init__(**kwargs)
        self.squeeze = Squeeze()
        self.flow_1 = FlowstepACN(ml)
        self.flow_2 = FlowstepACN(ml)
        self.split = split_along_channels
        self.flow_3 = FlowstepACN(ml)
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
        
