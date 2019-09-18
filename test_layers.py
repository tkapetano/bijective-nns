# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:47:15 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2, SplitLayer
from helper import int_shape, split_along_channels

import unittest

class TestCaseLayers(unittest.TestCase):
    """
    Testing individual layers of the module 'layers.py'. 
    """
    def setUp(self):
        ml = True # the most general setting
        self.squeeze = Squeeze()
        self.actn = Actnorm(ml)        
        self.conv1x1 = Conv1x1(ml)
        self.coupling = CouplingLayer2(ml)
        self.split = SplitLayer(ml)
        
        self.dist = lambda x, x_approx: np.linalg.norm(x - x_approx)
    
        
    def testShapes(self):
        # positive    
        inputs = tf.ones([4, 2, 2, 3])
        squeezed = self.squeeze(inputs)
        self.assertEqual([4,1,1,12], int_shape(squeezed))
        desqueezed = self.squeeze.invert(squeezed)
        self.assertEqual([4,2,2,3], int_shape(desqueezed))
        
        actnormed = self.actn(inputs)
        self.assertEqual([4,2,2,3], int_shape(actnormed))
            
        convolved = self.conv1x1(inputs)
        self.assertEqual([4,2,2,3], int_shape(convolved))
        
        coupled = self.coupling(squeezed)
        self.assertEqual([4,1,1,12], int_shape(coupled))
        
        split_a, split_b = split_along_channels(squeezed)
        self.assertEqual([4,1,1,6], int_shape(split_a))
        self.assertEqual([4,1,1,6], int_shape(split_b))
        
        input_a, input_b, eps_b = self.split(squeezed)
        self.assertEqual([4,1,1,6], int_shape(input_a))
        self.assertEqual([4,1,1,6], int_shape(input_b))
        self.assertEqual([4,1,1,6], int_shape(eps_b))
        
        recon = self.split.invert_sample(input_a, eps_b)
        self.assertEqual([4,1,1,12], int_shape(recon))        
        
        # negative
        inputs = tf.ones([4, 3, 3, 2])
        try: 
            _ = self.squeeze(inputs)
        except AssertionError:
            self.assertTrue(True)
            
        inputs = tf.ones([4, 2, 2, 6])
        try: 
            _ = self.squeeze.invert(inputs)
        except AssertionError:
            self.assertTrue(True)
    
    def testInvertion(self):
        inputs = tf.ones([4, 2, 2, 3])
        squeezed = self.squeeze(inputs)
        desqueezed = self.squeeze.invert(squeezed)
        self.assertLessEqual(self.dist(inputs, desqueezed), 1e-5)
        
        actnormed = self.actn(inputs)
        deactnormed = self.actn.invert(actnormed)
        self.assertLessEqual(self.dist(inputs, deactnormed), 1e-5)
        
        convolved = self.conv1x1(inputs)
        deconvolved = self.conv1x1.invert(convolved)
        self.assertLessEqual(self.dist(inputs, deconvolved), 1e-5)
        
        coupled = self.coupling(squeezed)
        decoupled = self.coupling.invert(coupled)
        self.assertLessEqual(self.dist(squeezed, decoupled), 1e-5)
    
        input_a, input_b, eps_b = self.split(squeezed)
        split_1, _ = split_along_channels(squeezed)
        self.assertLessEqual(self.dist(input_a, split_1), 1e-5)
        
        recon = self.split.invert_sample(input_a, eps_b)
        recon_1, _ = split_along_channels(recon)
        self.assertLessEqual(self.dist(split_1, (recon_1)), 1e-5)     
        
    
        
    def testLosses(self):
        # Due to initialization Actnorm, Conv1x1 and Coupling have 0 loss at the beginning
        inputs = tf.ones([4, 2, 2, 2])
        _ = self.actn(inputs)
        loss_acn = self.actn.losses
        loss_expected = tf.zeros([4])
        print(loss_acn)
        self.assertLessEqual(self.dist(loss_expected, loss_acn), 1e-8)
        
        _ = self.conv1x1(inputs)
        loss_conv = self.conv1x1.losses
        print(loss_conv)
        self.assertLessEqual(self.dist(loss_expected, loss_conv), 1e-5)
        # small error here, why?
        
        _ = self.coupling(inputs)
        loss_coupling = self.coupling.losses
        print(loss_coupling)
        self.assertLessEqual(self.dist(loss_expected, loss_coupling), 1e-8)
        
        # mean and log_var are initialized with 0s, hence loss boils down to 
        # log det of density of standard normal
        _, _, _ = self.split(inputs)
        loss_split = self.split.losses
        dims = int_shape(inputs)
        factor = float(dims[0] * dims[1] * dims[2] * dims[3]) / 2.
        loss_val = -0.5 * factor * (1. ** 2. + tf.math.log(2. * np.pi))
        loss_expected = loss_val * tf.ones([4])
        print(loss_split)
        self.assertLessEqual(self.dist(loss_expected, loss_split), 1e-8)
        
        inputs = 2. * tf.ones([4, 2, 2, 2])
        _, _, _ = self.split(inputs)
        loss_split = self.split.losses
        loss_val = -0.5 * factor * (2. ** 2. + tf.math.log(2. * np.pi))
        loss_expected = loss_val * tf.ones([4])
        self.assertLessEqual(self.dist(loss_expected, loss_split), 1e-8)
        
    
        
if __name__ == '__main__':
    unittest.main()