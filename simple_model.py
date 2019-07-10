# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:40:58 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from blocks import ClassifierACN

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# inspect data
print(train_images.shape)
print(test_images.shape)
print(train_labels[0])
print(train_labels.shape)
print(test_labels[0])

def run_model(epochs):
    model = ClassifierACN(label_classes=10, ml=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
                  
    model.fit(train_images, train_labels, epochs=epochs, batch_size=32, 
              validation_data=(test_images, test_labels))
    

