# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:52:30 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from model import Encoder

glow = Encoder((28,28,1))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# training
def train(epochs):
    for epoch in range(epochs):
        print('Start of epoch%d' % (epoch,))
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                encoding = glow(x_batch_train)
                loss = tf.reduce_sum(glow.losses)
            
            grads = tape.gradient(loss, glow.trainable_variables)
            optimizer.apply_gradients(zip(grads, glow.trainable_variables))
        print( 'Epoch: {}, ml: {}'.format(epoch, loss))
        glow.sampleplot()
            
            
train(5)