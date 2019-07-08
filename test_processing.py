# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:48:50 2019

@author: tempo
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# pre - post processing check

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)

def preprocess1(train_data, discrete_vals=256):
    x = tf.cast(train_data, 'float32')
    x = x / discrete_vals 
    x += tf.random.uniform(x.shape, 0, 1. / discrete_vals)
    return x
     
def help_plot(x):
    plt.figure()
    plt.imshow(x)
    plt.colorbar()
    plt.grid(False)
    plt.show()

# before preprocessing
img = tf.reshape(x_train[0], (28,28))
help_plot(img)  
 
# after preprocessing   
x_train = preprocess1(x_train)   
img_pre = tf.reshape(x_train[0], (28,28))
help_plot(img_pre) 

def postprocess(z, discrete_vals=256):
    return tf.cast(tf.clip_by_value(tf.floor(z*discrete_vals), 0, discrete_vals-1), 'uint8')
    
# after postprocessing
img_post = postprocess(img_pre)
img_post = tf.reshape(img_post, (28,28))
help_plot(img_post) 

dist = np.linalg.norm(img - img_post)
print('Difference of an image before and after pre+postprocessing: {}'.format(dist))