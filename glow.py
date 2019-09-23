# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:38:00 2019

@author: tempo
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from invertible_layers import SplitLayer
from helper import int_shape
from blocks import FlowstepACN, FlowstepSqueeze

class Encoder(tf.keras.layers.Layer):
    def __init__(self, name='encoder', ml=True, blocks_per_level=[4,4], **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.level_1 = [FlowstepSqueeze(ml=ml)]
        for i in range(blocks_per_level[0]):
            self.level_1.append(FlowstepACN(ml=ml))
        self.split_1 = SplitLayer(ml=ml)
        self.level_2 =  [FlowstepSqueeze(ml=ml)]
        for i in range(blocks_per_level[1]):
            self.level_2.append(FlowstepACN(ml=ml))
        self.split_2 = SplitLayer(ml=ml)
          
    def call(self, inputs):
        shape = int_shape(inputs)
        assert shape[1] % 4 == 0 and shape[2] % 4 == 0
        y = inputs
        for block in self.level_1:
            y = block(y)
        y, y_b, eps_1 = self.split_1(y)
        for block in self.level_2:
            y = block(y)
        y_aa, y_ab, eps_2 = self.split_2(y)
        return y_aa, y_ab, y_b, eps_1, eps_2
          
    def invert(self, z_a, z_ab, z_b, sample=False):    
        shape = int_shape(z_a)
        assert shape[-1] % 4 == 0
        x = self.split_2.invert(z_a, z_ab, sample=sample)        
        for block in reversed(self.level_2):
            x = block.invert(x)
        x = self.split_1.invert(x, z_b, sample=sample)
        for block in reversed(self.level_1):
            x = block.invert(x)
        return x
        
    def data_dependent_init(self, init_data_batch):
        inputs = init_data_batch
        for block in self.level_1:
            inputs = block.data_dependent_init(inputs)
        inputs, _, _ = self.split_1(inputs)
        for block in self.level_2:
            inputs = block.data_dependent_init(inputs)
    
    def compute_output_shapes(self, input_shape):
        out_fst_level = [input_shape[-2]/2, input_shape[-2]/2, input_shape[-1]*2]
        out_scd_level = [input_shape[-2]/4, input_shape[-2]/4, input_shape[-1]*4]
        return 2*[out_scd_level] + 2*[out_fst_level] + [out_scd_level]
        

  
class GlowNet(tf.keras.Model):
    def __init__(self, label_classes, input_shape, name='glownet', ml=True, **kwargs):
        super(GlowNet, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.encoder = Encoder(ml=ml)
        self.flatten =  tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(label_classes)
        self.flatten_nuisance =  tf.keras.layers.Flatten()
        self.dense_nuisance = tf.keras.layers.Dense(label_classes)
      
    def call(self, inputs):
        self.enable_only_classification(ml=self.ml)
        y, _, _, _, _ = self.encoder(inputs)
        y = self.flatten(y)
        y = self.dense(y)
        return tf.nn.softmax(y)
        
    def call_all(self, inputs):
        y_aa, y_ab, y_b, _, _ = self.encoder(inputs)
        y_aa = self.flatten(y_aa)
        y_aa = self.dense(y_aa)
        y_classification = tf.nn.softmax(y_aa)
    
        y_bb = tf.concat([self.flatten(y_ab), self.flatten(y_b)], axis=-1) 
        y_bb = self.dense_nuisance(y_bb)
        y_nuisance_classification = tf.nn.softmax(y_bb)
      
        return  y_classification, y_nuisance_classification
        
    def enable_only_classification(self, ml=True):
        self.encoder.trainiable = True  
        self.dense.trainable = True  
        self.dense_nuisance.trainable = False
        if ml:
            self.split_1.trainable = True
            self.split_2.trainable = True   
        else:
            self.split_1.trainable = False
            self.split_2.trainable = False 
         
    def enable_only_nuisance_classification(self, ml=True):
        self.encoder.trainiable = False 
        self.dense.trainable = False  
        self.dense_nuisance.trainable = True
        if ml:
            self.split_1.trainable = True
            self.split_2.trainable = True   
        else:
            self.split_1.trainable = False
            self.split_2.trainable = False 
    
    
      
  
