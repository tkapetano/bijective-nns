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
from helper import int_shape, split_along_channels, GaussianIsotrop, LogisticDist
from glow import Encoder, GlowNet


NAME = 's13'

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
        # forward pass to fill the losses with logdet values
        self.call(self.invert(z))
        logdet = sum(tf.math.exp(self.losses))
        #z /= logdet
        x = self.invert(z)
        return x
        
    @classmethod
    def buildSimpleNet(cls, input_shape, use_gauss=True, blocks=3, use_permutations=False):
        encoder = []
        for i in range(blocks):
            if use_permutations:
                perm_layer = Conv1x1(ml=False, trainable=False)
                encoder.append(perm_layer)       
            else:
                encoder.append(Conv1x1(ml=True))
            encoder.append(CouplingLayer(ml=True, filters=(32,32)))
        return cls(input_shape, encoder, use_gauss)
        
    @classmethod
    def buildMnistNet(cls, input_shape=(28, 28, 1), use_gauss=False, blocks=5, use_permutations=False):
        encoder = []
        encoder.append(Squeeze())
        encoder.append(Squeeze())
        for i in range(blocks):
            if use_permutations:
                perm_layer = Conv1x1(ml=False, trainable=False)
                encoder.append(perm_layer)       
            else:
                encoder.append(Conv1x1(ml=True))
            encoder.append(CouplingLayer(ml=True, filters=(16,16)))
        return cls(input_shape, encoder, use_gauss)
        
    @classmethod
    def buildMnistNet2(cls, input_shape=(28, 28, 1), use_gauss=False, blocks=5, use_permutations=False):
        encoder = []
        encoder.append(Squeeze())
        encoder.append(Squeeze())
        for i in range(blocks):
            encoder.append(FlowstepACN(ml=True, filters=(8,8)))
        return cls(input_shape, encoder, use_gauss)
        
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


        
def training_mnist(model, dataset, epochs):
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

def training(model, dataset, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                z = model(batch)
                log_det = sum(model.losses)
                log_prior = tf.reduce_mean(model.dist.logp(z))
                loss_value =  - log_prior - log_det 
                print('Epoch {} has a NLL of {}'.format(epoch+1, loss_value))
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
#x_samples = generate_data(batch_size)
#TrivialNet1.plot_samples(x_samples)
#model = TrivialNet1(input_shape=int_shape(x_samples), blocks=3)
#training(model, dataset, 45)
    
    
def re_shape(tensor):
    batch_size = int_shape(tensor)[0]
    x_vals = []
    y_vals = []
    for i in range(batch_size):
        x_vals.append(tensor[i, 0, 0, 0])
        y_vals.append(tensor[i, 0, 0, 1])
    return x_vals, y_vals
    
def plot_dist_samples(dist):
    points = [dist.sample() for i in range(batch_size)]
    x_vals = []
    y_vals = []
    for p in points:
        x_vals.append(p[ 0, 0, 0])
        y_vals.append(p[ 0, 0, 1])
    fig = plt.figure()
    plt.plot(x_vals, y_vals, 'ro')
    plt.show()
    fig.savefig('dist' + '_generate.png')
    
    
def test_sampling(model):
    images = [model.sample() for i in range(4)]
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(tf.reshape(img, [28,28]))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def sample_points(model):
    points = [model.sample() for i in range(batch_size)]
    x_vals = []
    y_vals = []
    for p in points:
        x_vals.append(p[0, 0, 0, 0])
        y_vals.append(p[0, 0, 0, 1])
    fig = plt.figure()
    plt.plot(x_vals, y_vals, 'ro')
    plt.show()
    fig.savefig(NAME + '_generate.png')

    
def show_space_contraction(model, all_layers=False):
    horizontal_lines = []
    vertical_lines = []
    line = np.linspace(-2.0, 2.0, 256)
    #line = np.linspace(-16.0, 16.0, 256)
    line = tf.cast(tf.reshape(line, shape=(256,1,1,1)), 'float32')
    for i in range(3):
        y_plus = tf.zeros(shape=line.get_shape(), dtype=tf.dtypes.float32) + float(i)
        y_minus = tf.zeros(shape=line.get_shape(), dtype=tf.dtypes.float32) - float(i)
        line_plus_h = tf.concat([line, y_plus], axis=-1)
        line_minus_h = tf.concat([line, y_minus], axis=-1)
        line_plus_v = tf.concat([y_plus, line], axis=-1)
        line_minus_v = tf.concat([y_minus, line], axis=-1)
        horizontal_lines += [line_plus_h, line_minus_h]
        vertical_lines += [line_plus_v, line_minus_v]
        
    if all_layers:
        num_of_figures = len(model.layers) + 1
        fig, axes = plt.subplots(1, num_of_figures, figsize=(5*num_of_figures,5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes = axes.flatten()

    for line in horizontal_lines:
        x,y = re_shape(line)
        axes[0].plot(x, y, 'go--', linewidth=2, markersize=3)
        if all_layers:
            for num, layer in enumerate(reversed(model.layers)):
                line = layer.invert(line)
                x,y = re_shape(line)
                axes[num+1].plot(x, y, 'go--', linewidth=2, markersize=3)
        else:
            z = model.invert(line)
            x,y = re_shape(z)
            axes[1].plot(x, y, 'go--', linewidth=2, markersize=3)
        
    for line in vertical_lines:
        x,y = re_shape(line)
        axes[0].plot(x, y, 'ro--', linewidth=2, markersize=3)
        if all_layers:
            for num, layer in enumerate(reversed(model.layers)):
                line = layer.invert(line)
                x,y = re_shape(line)
                axes[num+1].plot(x, y, 'ro--', linewidth=2, markersize=3)
        else:
            z = model.invert(line)
            x,y = re_shape(z)
            axes[1].plot(x, y, 'ro--', linewidth=2, markersize=3)  
        
    plt.show()
    if all_layers:
        name_add = '_all'
    else:
        name_add = ''
    
    fig.savefig(NAME + name_add + '_contract_back.png')
    
def run():
    dataset = [generate_data(batch_size) for i in range(100)]
    model = GenerativeFlow.buildSimpleNet(input_shape=(1, 1, 2), use_gauss=False, blocks=3, use_permutations=False)
    training(model, dataset, 20)
    sample_points(model)
    show_space_contraction(model)
    show_space_contraction(model, True)
    
