# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:38:00 2019

@author: tempo
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt

from helper import int_shape, split_along_channels
from blocks import FlowstepACN, FlowstepSqueeze


class Encoder(tf.keras.layers.Layer):
    def __init__(self, name='encoder', ml=False, blocks_per_level=[4,4], **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.level_1 = [FlowstepSqueeze(ml=ml)]
        for i in range(blocks_per_level[0]):
            self.level_1.append(FlowstepACN(ml=ml))
        self.level_2 =  [FlowstepSqueeze(ml=ml)]
        for i in range(blocks_per_level[1]):
            self.level_2.append(FlowstepACN(ml=ml))
          
    def call(self, inputs):
        shape = int_shape(inputs)
        assert shape[1] % 4 == 0 and shape[2] % 4 == 0
        y = inputs
        for block in self.level_1:
            y = block(y)
        y, y_b = split_along_channels(y)
        for block in self.level_2:
            y = block(y)
        return y, y_b
          
    def invert(self, z, z_b):    
        shape = int_shape(z)
        assert shape[-1] % 4 == 0
        x = z
        for block in reversed(self.level_2):
            x = block.invert(x)
        x = tf.concat([x,z_b], axis=-1)
        for block in reversed(self.level_1):
            x = block.invert(x)
        return x
        
    def data_dependent_init(self, init_data_batch):
        inputs = init_data_batch
        for block in self.level_1:
            inputs = block.data_dependent_init(inputs)
        inputs, _ = split_along_channels(inputs)
        for block in self.level_2:
            inputs = block.data_dependent_init(inputs)
    
    def compute_output_shape(self, input_shape):
        out_fst_level = [int(input_shape[-2]/2), int(input_shape[-2]/2), int(input_shape[-1]*2)]
        out_scd_level = [int(input_shape[-2]/4), int(input_shape[-2]/4), int(input_shape[-1]*8)]
        return [out_fst_level, out_scd_level] 
        

  
class IRevNet(tf.keras.Model):
    def __init__(self, num_of_labels, input_shape, name='irevnet', ml=False, **kwargs):
        super(IRevNet, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.num_of_labels = num_of_labels
      
        self.encoder = Encoder(ml=ml, input_shape=input_shape)  
        [self.level_1_out_shape, self.level_2_out_shape] = self.encoder.compute_output_shape(input_shape)
        self.flatten_1 =  tf.keras.layers.Flatten()
        self.flatten_2 = tf.keras.layers.Flatten()
        out_dims = input_shape[0] * input_shape[1] * input_shape[2] - num_of_labels
        self.nuisance_classifier = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(256, input_shape=(out_dims,), activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(num_of_labels)])
      
    def call(self, inputs):
        """Allows for usage of keras API: model.fit, model.evaluate etc.
            Make sure to call model.enable_only_classification() and 
            initialize the model with ml=False in this case.
        """
        y, y_b = self.encoder(inputs)
        # nuisance classifier is only called here, to assure the model is build
        # correctly and dimensions can be automatically infered
        y_b_flat = self.flatten_1(y_b)        
        y_flat = self.flatten_2(y)
        y_total = tf.concat([y_flat, y_b_flat], axis=-1)
        y_logits = y_total[:, :self.num_of_labels]
        y_nuisance = y_total[:, self.num_of_labels:]
        self.nuisance_classifier(y_nuisance)
        return y_logits
        
        
    def call_all(self, inputs):
        y, y_b = self.encoder(inputs)
        # nuisance classifier is only called here, to assure the model is build
        # correctly and dimensions can be automatically infered
        y_b_flat = self.flatten_1(y_b)        
        y_flat = self.flatten_2(y)
        y_total = tf.concat([y_flat, y_b_flat], axis=-1)
        y_logits = y_total[:, :self.num_of_labels]
        y_nuisance = y_total[:, self.num_of_labels:]
        y_nuisance_logits = self.nuisance_classifier(y_nuisance)
        return y_logits, y_nuisance, y_nuisance_logits
        
    def metameric_sampling(self, logits, nuisance):
        batch = int_shape(logits)[0]
        y_total = tf.concat([logits, nuisance], axis=-1)
        length = y_total.get_shape()[-1]
        y, y_b = y_total[:, :int(length/2)], y_total[:, int(length/2):]
        # compute necessary shapes
        y = tf.reshape(y, shape=[batch] + self.level_2_out_shape)
        y_b = tf.reshape(y_b, shape=[batch] + self.level_1_out_shape)
        return self.encoder.invert(y, y_b)

        
    def enable_only_classification(self):
        """Freezes all weight that do not feed into the semantic variables."""
        self.encoder.trainable = True  
        self.nuisance_classifier.trainable = False
         
    def enable_only_nuisance_classification(self):
        """Freezes all weight that do not feed into the nuisance variables."""
        self.nuisance_classifier.trainable = True
        self.encoder.trainable = False 
        
    def plot_and_save_adv_examples(self, images, trial_num, epoch):
        assert int_shape(images)[0] == 2
        image_logits, image_nuisance, _ = self.call_all(images)
        image_list = []
        
        logit_1 = image_logits[0, :]
        logit_2 = image_logits[1, :]
        nuis_1 = image_nuisance[0, :]
        nuis_2 = image_nuisance[1, :]
        image_list.append(self.metameric_sampling(logit_1, nuis_1))
        image_list.append(self.metameric_sampling(logit_1, nuis_2))
        image_list.append(self.metameric_sampling(logit_2, nuis_1))
        image_list.append(self.metameric_sampling(logit_2, nuis_2))
        
        fig, axes = plt.subplots(1, 4, figsize=(16,16))
        axes = axes.flatten()
        for img, ax in zip(image_list, axes):
            ax.imshow(tf.reshape(img, [28,28]))
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        #fig.savefig('model' + str(trial_num) + '_metameres_epoch_' + str(epoch) +'.png')
            
        
      
  
