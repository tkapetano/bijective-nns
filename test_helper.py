# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:31:03 2019

@author: tkapetano
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import numpy as np
#from layers import Squeeze, Actnorm, Conv1x1, CouplingLayer2
from helper import int_shape, split_along_channels, actn_init

import unittest

class TestCaseHelper(unittest.TestCase):
    """
    Testing the individual layers of the module 'helper.py'
    """
#    def setUp(self):
#        ml = True # the most general setting
#        self.squeeze = Squeeze()
#        self.actn = Actnorm(ml)
#        self.conv1x1 = Conv1x1(ml)
#        self.coupling = CouplingLayer2(ml)
#    
#        self.dist = lambda x, x_approx: np.linalg.norm(x - x_approx)
    
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
        
        
       
if __name__ == '__main__':
    unittest.main()