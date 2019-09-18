# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:38:00 2019

@author: tempo
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2, SplitLayer
from helper import split_along_channels, int_shape
from blocks import FlowstepACN, FlowstepSqueeze

class Encoder(tf.keras.layers.Layer):
    def __init__(self, label_classes, input_shape, name='encoder', ml=False, data_init=None, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.flow_1 = FlowstepSqueeze(ml)
        self.flow_2 = FlowstepACN(ml, data_init)
        self.flow_3 = FlowstepACN(ml, data_init)
        self.flow_4 = FlowstepACN(ml, data_init)
        self.flow_5 = FlowstepACN(ml, data_init)
        self.split_1 = SplitLayer(ml)
        self.flow_6 = FlowstepSqueeze(ml)
        self.flow_7 = FlowstepACN(ml, data_init)
        self.flow_8 = FlowstepACN(ml, data_init)
        self.flow_9 = FlowstepACN(ml, data_init)
        self.flow_10 = FlowstepACN(ml, data_init)
        self.split_2 = SplitLayer(ml)
          
    def call(self, inputs):
        y = self.flow_1(inputs)
        y = self.flow_2(y)
        y = self.flow_3(y)
        y = self.flow_4(y)
        y = self.flow_5(y)
        y_a, y_b, eps_1 = self.split_1(y)
        y = self.flow_6(y_a)
        y = self.flow_7(y)
        y = self.flow_8(y)
        y = self.flow_9(y)
        y = self.flow_10(y)
        y_aa, y_ab, eps_2 = self.split_2(y)
        return y_aa, y_ab, y_b, eps_1, eps_2
          
    def invert(self, z_a, z_ab, z_b, sample=False):    
        shape = int_shape(z_a)
        channels = shape[3]
        assert channels % 2 == 0 and shape[1] % 2 == 0 and shape[2] % 2 == 0
        if sample:
            x_a = self.split_2.invert_sample(z_a, None)     
        else:
            x_a = self.split_2.invert(z_a, z_ab)        
        x_a = self.flow_10.invert(x_a)
        x_a = self.flow_9.invert(x_a)
        x_a = self.flow_8.invert(x_a)
        x_a = self.flow_7.invert(x_a)
        x_a = self.flow_6.invert(x_a)
        if sample:
            x = self.split_1.invert_sample(x_a, None)
        else:
            x = self.split_1.invert(x_a, z_b)
        x = self.flow_5.invert(x)
        x = self.flow_4.invert(x)
        x = self.flow_3.invert(x)
        x = self.flow_2.invert(x)
        x = self.flow_1.invert(x)
        return x
        
    #def compute_output_shapes

  
class Glow(tf.keras.Model):
  def __init__(self, label_classes, input_shape, ml):
    super(Glow, self).__init__()
    self.encoder = Encoder(label_classes, input_shape=input_shape, ml=ml, data_init=None)
    self.flatten =  tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(label_classes)
    #self.nuisance_classifier = tf.keras.Sequential(
    #    [tf.keras.layers.InputLayer(input_shape=)
    #])
    
      
  def call(self, inputs):
      y, _, _, _, _ = self.encoder(inputs)
      y = self.flatten(y)
      y = self.dense(y)
      #ml_loss = self.encoder.losses
      #print(ml_loss)
      #self.add_loss(ml_loss)
      return tf.nn.softmax(y)
      
#  @tf.function
#  def sample(self):
#    z_a = tf.random.normal(shape=(28,28,1))
#    return self.encoder.invert(z_a, None, None, sample=True)


model = Glow(10, (28,28,1), ml=True)