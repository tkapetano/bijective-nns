# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:28:53 2019

@author: tempo
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from invertible_layers import Squeeze, Actnorm, Conv1x1, CouplingLayer, FlipChannels
from blocks import FlowstepACN, FlowstepSqueeze
from helper import int_shape, split_along_channels, GaussianIsotrop, LogisticDist


NAME = 's16'

class GenerativeFlow(tf.keras.Model):
    def __init__(self, input_shape, encoder, use_gauss=True, name='genflow', ml=True, **kwargs):
        super(GenerativeFlow, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.encoder = encoder
        out_shape = input_shape
        for layer in self.encoder:
            out_shape = layer.compute_output_shape(out_shape)
        self.out_shape = out_shape
        zeros = tf.zeros(shape=self.out_shape)
        if use_gauss:
            self.dist = GaussianIsotrop(zeros, zeros)
        else:
            ones = tf.ones(shape=self.out_shape)
            self.dist = LogisticDist(zeros, ones)
        
    def call(self, inputs):
        for layer in self.encoder:
            inputs = layer(inputs) 
        return inputs
        
    def invert(self, outputs):
        for layer in reversed(self.encoder):
            outputs = layer.invert(outputs) 
        return outputs
        
    def sample(self):
        z = self.dist.sample()
        z = tf.reshape(z, shape=[1] + list(self.out_shape))
        x = self.invert(z)
        return x
        
    @classmethod
    def buildSimpleNet(cls, input_shape, use_gauss=True, blocks=3, use_permutations=False):
        encoder = []
        for i in range(blocks):
            if use_permutations:
                perm_layer = FlipChannels()
                encoder.append(perm_layer)       
            else:
                encoder.append(Conv1x1(ml=False, lu_decom=True))
            encoder.append(CouplingLayer(ml=True, filters=(32,32)))
        return cls(input_shape, encoder, use_gauss)
        
    @classmethod
    def buildMnistNet(cls, input_shape=(28, 28, 1), use_gauss=False, blocks=5, use_permutations=False):
        encoder = []
        encoder.append(Squeeze())
        encoder.append(Squeeze())
        for i in range(blocks):
            if use_permutations:
                perm_layer = FlipChannels()
                encoder.append(perm_layer)       
            else:
                encoder.append(Conv1x1(ml=False, lu_decom=True))
            encoder.append(CouplingLayer(ml=True, filters=(12,12)))
        return cls(input_shape, encoder, use_gauss)
        

batch_size = 256

    
def test_sampling(model, trial_num=0, epoch=0):
    images = [model.sample() for i in range(4)]
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(tf.reshape(img, [28,28]), cmap='Greys',  interpolation='nearest')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    fig.savefig('model' + str(trial_num) + '_gen_samples_epoch_' + str(epoch) +'.png')
    


        
def training_mnist(model, dataset, epochs, trial_num):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(epochs):
        for batch, _ in dataset:
            with tf.GradientTape() as tape:
                z = model(batch)
                log_det = sum(model.losses)
                log_prior = tf.reduce_mean(model.dist.logp(z))
                loss_value =  - log_prior - log_det 
                print('Epoch {} has a NLL of {}'.format(epoch+1, loss_value))
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  
        if epoch % 3 == 0:
            model.save_weights('./' + str(trial_num) + '_' + str(epoch+1) )
            test_sampling(model, trial_num, epoch+1)
            

    
