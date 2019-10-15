# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:28:53 2019

@author: tempo
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from invertible_layers import Squeeze, Actnorm, Conv1x1, CouplingLayer, SplitLayer
from blocks import FlowstepACN, FlowstepSqueeze
from helper import int_shape, split_along_channels, GaussianIsotrop
from glow import Encoder, GlowNet



class TrivialNet1(tf.keras.Model):
    def __init__(self, input_shape, name='trivialnet1', ml=True, blocks=1, **kwargs):
        super(TrivialNet1, self).__init__(name=name, **kwargs)
        self.ml = ml
        self.encoder = []
        for i in range(blocks):
            self.encoder.append(Conv1x1(ml=ml))
            self.encoder.append(CouplingLayer(ml=ml))
        #mean = tf.zeros(shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]//2))
        #self.gauss = GaussianIsotrop(mean, mean)
        zeros = tf.zeros(shape=input_shape)
        self.gauss = GaussianIsotrop(zeros, zeros)
        
    def call(self, inputs):
        for layer in self.encoder:
            inputs = layer(inputs) 
            #self.plot_samples(inputs)
        return inputs
        
    def invert(self, outputs):
        for layer in reversed(self.encoder):
            outputs = layer.invert(outputs) 
        return outputs
        
    @staticmethod
    def plot_samples(tensor):
        batch_size = int_shape(tensor)[0]
        x_vals = []
        y_vals = []
        for i in range(batch_size):
            x_vals.append(tensor[i, 0, 0, 0])
            y_vals.append(tensor[i, 0, 0, 1])
            
        plt.plot(x_vals, y_vals, 'ro')
        plt.show()
        
#    
#inputs = tf.ones(shape=(1,1,1,2))
#model = TrivialNet1(input_shape=int_shape(inputs), blocks=4)
#outputs = model(inputs)
#input_recon = model.invert(outputs)
#print(input_recon)
batch_size = 256



def generate_data(batch_size):
    shape = (batch_size, 1, 1, 1)
    x_2 = tf.keras.backend.random_normal(
        shape=shape,
        mean=0.0,
        stddev=4.0)
        
    x_1 = tf.keras.backend.random_normal(
        shape=shape,
        mean=0.25 * tf.square(x_2),
        stddev=1.0)
    
    return tf.concat([x_1, x_2], axis=3)

dataset = [generate_data(batch_size) for i in range(100)]
        
def training(model, dataset, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                z = model(batch)
           
                log_det = sum(model.losses)
                log_prior = sum(model.gauss.logp(z))
                
                loss_value =  (- log_prior - log_det ) / float(batch_size) / 2.
                print('Epoch {} has a NLL of {}'.format(epoch+1, loss_value))
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
x_samples = generate_data(batch_size)
TrivialNet1.plot_samples(x_samples)
model = TrivialNet1(input_shape=int_shape(x_samples), blocks=3)
training(model, dataset, 30)

def test_sampling(model):
    z = model.gauss.sample()
    z = z /20.
    TrivialNet1.plot_samples(z)    
    x = model.invert(z)
    TrivialNet1.plot_samples(x)
    
def test_temperature_sampling():
    TrivialNet1.plot_samples(x_samples)
    TrivialNet1.plot_samples(model(x_samples))
    TrivialNet1.plot_samples(model.invert(model(x_samples)))
    
test_temperature_sampling()
test_sampling()