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
        self.classifier = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(label_classes, activation='softmax')])
        self.flatten =  tf.keras.layers.Flatten()
        self.nuisance_classifier = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(label_classes, activation='softmax')])
      
    def call(self, inputs):
        """Allows for usage of keras API: model.fit, model.evaluate etc.
            Make sure to call model.enable_only_classification() and 
            initialize the model with ml=False in this case.
        """
        y, y_ab, y_b, _, _ = self.encoder(inputs)
        # nuisance classifier is only called here, to assure the model is build
        # correctly and dimensions can be automatically infered
        y_bb = tf.concat([self.flatten(y_ab), self.flatten(y_b)], axis=-1) 
        self.nuisance_classifier(y_bb)
        return self.classifier(y)
        
    def call_nuisance(self, inputs):
        _, y_ab, y_b, _, _ = self.encoder(inputs)    
        y_bb = tf.concat([self.flatten(y_ab), self.flatten(y_b)], axis=-1) 
        return self.nuisance_classifier(y_bb)
        
    def call_all(self, inputs):
        y, y_ab, y_b, _, _ = self.encoder(inputs)    
        y_bb = tf.concat([self.flatten(y_ab), self.flatten(y_b)], axis=-1) 
        return self.classifier(y), self.nuisance_classifier(y_bb)
        
    def enable_only_classification(self):
        """Freezes all weight that do not feed into the semantic variables."""
        self.encoder.trainiable = True  
        self.encoder.split_1.trainable = False
        self.encoder.split_2.trainable = False
        self.classifier.trainable = True
        self.nuisance_classifier.trainable = False
         
    def enable_only_nuisance_classification(self):
        """Freezes all weight that do not feed into the nuisance variables."""
        self.nuisance_classifier.trainable = True
        self.encoder.trainable = False 
        self.classifier.trainable = False
        
        
      
  
