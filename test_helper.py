# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:31:03 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2
from helper import int_shape, split_along_channels, data_init_acn
from blocks import FlowstepSqueeze

import unittest

class TestCaseHelper(unittest.TestCase):
    """
    Testing the individual layers of the module 'helper.py'
    """
    def setUp(self):
        ml = True # the most general setting
        self.flowSqueeze = FlowstepSqueeze(ml=False)
        self.gaussians = tf.random.normal((7, 4, 4, 3), mean=3.0, stddev=2.0) 
        self.squeeze = Squeeze()
        self.actn = Actnorm(ml)
        self.conv1x1 = Conv1x1(ml)
        self.coupling = CouplingLayer2(ml)
    
        self.dist = lambda x, x_approx: np.linalg.norm(x - x_approx)
    
    def testShapes(self):
        # positive
        inputs = tf.ones([2, 2, 2, 4])
        out_1, out_2 = split_along_channels(inputs)
        self.assertEqual(out_1.shape, out_2.shape)
        self.assertEqual((2,2,2,2), out_1.shape)

#TODO:        
#        scale_init, bias_init = actn_init(inputs)
#        s_init = scale_init(inputs.shape)
#        b_init = bias_init(inputs.shape)
#        print(s_init)
#        print(b_init)
        
    def testDataInitACN(self):
        #batch = tf.ones([7, 4, 4, 3])
        list_of_inits = data_init_acn(self.flowSqueeze, self.gaussians)
        # now new init of the Flow block
        model_inited = FlowstepSqueeze(ml=False, data_init=list_of_inits[0])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_inited.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])      
        layers = model_inited.layers
        squeeze_layer = layers[0]
        flow_layer = layers[1]
        flow_output = flow_layer(squeeze_layer(self.gaussians))
        stddev = tf.ones((12,))
        mean = tf.zeros((12,))
        post_stddev = tf.math.reduce_std(flow_output, axis=(0,1,2))
        post_mean = tf.math.reduce_mean(flow_output, axis=(0,1,2))
        self.assertLessEqual(self.dist(stddev, post_stddev), 1e-5)
        self.assertLessEqual(self.dist(mean, post_mean), 1e-5)

        
        
if __name__ == '__main__':
    unittest.main()