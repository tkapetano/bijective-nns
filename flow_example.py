# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:23:47 2019

@author: tempo
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from invertible_layers import Squeeze, Actnorm, Conv1x1, CouplingLayer, SplitLayer
from blocks import FlowstepACN, FlowstepSqueeze
from helper import int_shape, split_along_channels, GaussianIsotrop, LogisticDist
from gen_flow import GenerativeFlow

NAME = 's19'
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
    

def training(model, dataset, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                z = model(batch)
                #print(model.losses)
                log_det = sum(model.losses)
                log_prior = tf.reduce_mean(model.dist.logp(z))
                loss_value =  - log_prior - log_det 
                print('Epoch {} has a NLL of {}'.format(epoch+1, loss_value))
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            

    
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


def _compute_likelihood(x,y):
    nll = tf.math.log(8 * np.pi)
    nll += (y- x**2./4.)**2./ 2. + x**2. / 32.
    return nll
    
def compute_likelihood(tensor):
    np_array = tf.reshape(tensor, (batch_size, 2)).numpy()
    mean = 0.0
    for row in np_array:
        mean += _compute_likelihood(row[0], row[1])
    return mean / batch_size
    
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
    model = GenerativeFlow.buildSimpleNet(input_shape=(1, 1, 2), use_gauss=False, blocks=3, use_permutations=True)
    training(model, dataset, 20)
    sample_points(model)
    show_space_contraction(model)
    show_space_contraction(model, True)
    
    
#mean = 0
#for i in range(100):
#    p = [model.sample() for i in range(batch_size)]
#    mean += compute_likelihood(p).numpy()
#mean /= 100.    

    
def alter():
    encoder = []
    for i in range(5):
        encoder.append(FlowstepACN(ml=True, filters=(12,12), lu_decom=True))
    model = GenerativeFlow(input_shape=(1, 1, 2), encoder=encoder, use_gauss=True, ml=True)    
    
    dataset = [generate_data(batch_size) for i in range(100)]
    
    training(model, dataset, 20)
    sample_points(model)
    show_space_contraction(model)
    show_space_contraction(model, True)