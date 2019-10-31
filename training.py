# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:27:38 2019

@author: tempo
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from helper import int_shape
from glow import IRevNet

from time import time

NUM_OF_LABELS = 10
BATCH_SIZE = 128
INPUT_SHAPE = [28,28,1]
FACTOR = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2] * tf.math.log(2.)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def prepare_mnist_dataset():
    """Loads the MNIST dataset, dequantizes, rescales to [0,1], shuffles 
        and batches it to return a training and a test dataset 
        of type tf.data.Dataset.    
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    def rescale(image, label):
        image = tf.cast(image, tf.float32)
        #image += tf.keras.backend.random_uniform(image.get_shape())
        image /= 266.
        return image, label
    
    def make_dataset(data, labels, total_num, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(total_num).map(rescale).batch(batch_size)  
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 
    
    train_dataset = make_dataset(train_images, train_labels, 60000, BATCH_SIZE)
    test_dataset = make_dataset(test_images, test_labels, 10000, BATCH_SIZE)
    return train_dataset, test_dataset


def track_gradients_and_losses(model, inputs, targets, inner_train_loop=5,
                               optimizer=tf.keras.optimizers.Adam(),
                               multi_task_loss_weights=[1, 1]):
    """Computes gradients w.r.t. to the three losses from the classification task,
        the nuisance classification task and the generative flow part. 
        # Inputs: multi_task_loss_weights is a list of length 3, that weights the
                  three loss terms by multiplying with the provided factor.
        # Outputs: the total loss, a list of the raw(unweighted) loss terms, 
                  and the gradient tape of the total loss wrt the model's trainable
                  variables.
    """
    with tf.GradientTape() as tape:
        loss_value = 0
        y_pred, y_nuisance, _ = model.call_all(inputs)
        
        loss_classification = LOSS(targets, y_pred)
        #print(loss_classification)
        loss_value += multi_task_loss_weights[0] * loss_classification
        
        # interleaving with 
        loss_nuisance = 0 
        for i in range(inner_train_loop):
            with tf.GradientTape() as inner_tape:
           
                loss_nuisance = LOSS(targets, model.nuisance_classifier(y_nuisance))
                inner_grads = inner_tape.gradient(loss_nuisance, 
                                                  model.nuisance_classifier.trainable_variables)
                optimizer.apply_gradients(zip(inner_grads, 
                                              model.nuisance_classifier.trainable_variables))
                #print(loss_nuisance)
                
        y_pred_nuisance = model.nuisance_classifier(y_nuisance)    
        loss_nuisance = - LOSS(targets, y_pred_nuisance)
        #print(loss_nuisance)
        loss_value += multi_task_loss_weights[1] * loss_nuisance 
        
    return loss_value, [loss_classification, loss_nuisance], \
            tape.gradient(loss_value, model.trainable_variables)
            
def train(model, trial_num, train_dataset, num_epochs, 
          optimizer=tf.keras.optimizers.Adam(),
          multi_task_loss_weights=[1., 1./ (28*28-10)]):
    train_loss_results = []
    train_class_loss_results = []
    train_nuisance_class_loss_results = []
    train_accuracy_results = []
    train_nuisance_accuracy_results = []
    
    for epoch in range(1, num_epochs):
        print('Starting epoch {} ..'.format(epoch))
        start = time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_class_loss_avg = tf.keras.metrics.Mean()
        epoch_nuisance_class_loss_avg = tf.keras.metrics.Mean()
        epoch_class_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_nuisance_class_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        steps = 0
    
        # Training loop - using batches
        for image_batch, label_batch in train_dataset:
            steps += 1
            if steps % 100:
                print('Processed {} of {} steps'.format(steps, int(60000/BATCH_SIZE)))
        # Optimize the model
            loss_value, loss_list, grads = track_gradients_and_losses(model, 
                                                                      image_batch, 
                                                                      label_batch, 
                                                                      multi_task_loss_weights=multi_task_loss_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
            # Track progress
            epoch_loss_avg(loss_value) 
            epoch_class_loss_avg(loss_list[0])
            epoch_nuisance_class_loss_avg(loss_list[1])

            # suppress when only training is the goal
            y_pred, _, y_nuisance_pred = model.call_all(image_batch)
            epoch_class_accuracy(label_batch, y_pred)
            epoch_nuisance_class_accuracy(label_batch, y_nuisance_pred)
    
        # End epoch
        end = time()
        print('Epoch {} finished after {:3f}'.format(epoch, end-start))
        
        train_loss_results.append(epoch_loss_avg.result())
        train_class_loss_results.append(epoch_class_loss_avg.result())
        train_nuisance_class_loss_results.append(epoch_nuisance_class_loss_avg.result())
        train_accuracy_results.append(epoch_class_accuracy.result())
        train_nuisance_accuracy_results.append(epoch_nuisance_class_accuracy.result())
    
        print("Epoch {:03d}: Loss: {:.3f}, Acc.: {:.3%}, Nuisance Acc.: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_class_accuracy.result(),
                                                                    epoch_nuisance_class_accuracy.result()))
        print("Class. Loss: {:.3f}, Nuisance Class. Loss: {:.3f}".format(epoch_class_loss_avg.result(),
                                                                    epoch_nuisance_class_loss_avg.result()))   
                                                                    
        if epoch % 5 == 0:
            model.save_weights('./trial_{}_epoch_{}'.format(trial_num, epoch))
            #images, _ = next(iter(train_dataset))
            #model.plot_and_save_adv_examples(images[0:2, :, :, :])
            
                                                                    
    return train_loss_results, train_class_loss_results, \
                train_nuisance_class_loss_results, \
                train_accuracy_results,  train_nuisance_accuracy_results
                
train_dataset, test_dataset = prepare_mnist_dataset()
#model = IRevNet(NUM_OF_LABELS, [28,28,1])
#train(model, train_dataset, num_epochs=2)