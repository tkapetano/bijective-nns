# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:47:15 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2
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
    
        
    
        
#TODO: test Losses 
#    def testLosses(self):
#        inputs = tf.ones([4, 2, 2, 2])
#        actnormed = self.actn(inputs)
#        #print(self.actn.losses)
    
        
if __name__ == '__main__':
    unittest.main()