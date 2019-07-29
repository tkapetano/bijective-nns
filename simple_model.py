# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:40:58 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from blocks import ClassifierInv

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = tf.cast(train_images, tf.float32) / 255.0, tf.cast(test_images, tf.float32) / 255.0
train_labels, test_labels = tf.constant(train_labels), tf.constant(test_labels)

# inspect data
print(train_images.shape)
print(test_images.shape)
print(train_labels[0])
print(train_labels.shape)
print(test_labels[0])

def run_model(epochs):
    model = ClassifierInv(label_classes=10, ml=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
                  
    model.fit(train_images, train_labels, epochs=epochs, batch_size=32, 
              validation_data=(test_images, test_labels))
    

