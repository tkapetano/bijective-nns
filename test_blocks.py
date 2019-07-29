# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:47:15 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from blocks import FlowstepACN, ClassifierACN, ClassifierInv, FlowstepSqueeze
from layers import Squeeze
from helper import int_shape

import unittest

class TestCaseBlocks(unittest.TestCase):
    """
    Testing block components of the module 'blocks.py'
    """
    def setUp(self):
        ml = True # the most general setting
        self.squeeze = Squeeze()
        self.flow = FlowstepACN(ml)
        self.flowSqueeze = FlowstepSqueeze(ml=False)
        self.classify = ClassifierACN(10, ml=False)
        self.classifyInv = ClassifierInv(10, ml=False)
    
        self.dist = lambda x, x_approx: np.linalg.norm(x - x_approx)
        
    def testShapes(self):
        # positive
        inputs = tf.ones([4, 2, 2, 2])
        outputs = self.flow(inputs)
        self.assertEqual(int_shape(inputs), int_shape(outputs))
        
        inputs = tf.ones([4, 4, 4, 1])
        outputs = self.classify(inputs)
        self.assertEqual((4, 10) , outputs.get_shape())
        
        inputs = tf.ones([4, 4, 4, 1])
        outputs = self.classifyInv(inputs)
        self.assertEqual((4, 1, 1, 4) , outputs.get_shape())
        
        
        
         
    def testLosses(self):
        inputs = tf.ones([4, 2, 2, 2])
        _ = self.flow(inputs)
        #print('losses: ' + str(self.flow.losses) + ' end')
        self.assertEqual(len(self.flow.losses), 3)
        self.assertLessEqual(self.flow.losses[1], 1e-5)
        self.assertLessEqual(self.flow.losses[2], 1e-5)
        
                
    def testInvertion(self):
        inputs = tf.ones([4, 2, 2, 2])
        flowacn = self.flow(inputs)
        recon_flowacn = self.flow.invert(flowacn)
        self.assertLessEqual(self.dist(inputs, recon_flowacn), 1e-5)
   
      
        
if __name__ == '__main__':
    unittest.main()
