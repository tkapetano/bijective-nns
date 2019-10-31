# -*- coding: utf-8 -*-
"""
@author: tkapetano

Unittest Testsuite for modules 'invertible_layers' , 'blocks', 'glow' and 'helper'
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from invertible_layers import Squeeze, Actnorm, Conv1x1, CouplingLayer, SplitLayer
from blocks import FlowstepACN, FlowstepSqueeze
from helper import int_shape, split_along_channels, lu_decomposition, DetOne
from glow import Encoder, IRevNet #GlowNet

import unittest

L2NORM = lambda x, x_approx: np.linalg.norm(x - x_approx)

class TestCaseLayers(unittest.TestCase):
    """
    Testing individual layers of the module 'invertible_layers.py'. 
    """
    def setUp(self):
        ml = True
        self.squeeze = Squeeze()
        self.actn = Actnorm(ml=ml)        
        self.conv1x1 = Conv1x1(ml=ml)
        self.conv_lu =  Conv1x1(ml=ml, lu_decom=True)
        self.coupling = CouplingLayer(ml=ml)
        self.split = SplitLayer(ml=ml)
        
        
    def test_layer_in_and_output_shapes(self):
        inputs = tf.ones([4, 2, 2, 3])
        squeezed = self.squeeze(inputs)
        self.assertEqual([4,1,1,12], int_shape(squeezed))
        desqueezed = self.squeeze.invert(squeezed)
        self.assertEqual([4,2,2,3], int_shape(desqueezed))
        
        actnormed = self.actn(inputs)
        self.assertEqual([4,2,2,3], int_shape(actnormed))
            
        convolved = self.conv1x1(inputs)
        self.assertEqual([4,2,2,3], int_shape(convolved))
        convolved_lu = self.conv_lu(inputs)
        self.assertEqual([4,2,2,3], int_shape(convolved_lu))
        
        
        coupled = self.coupling(squeezed)
        self.assertEqual([4,1,1,12], int_shape(coupled))
        
        split_a, split_b = split_along_channels(squeezed)
        self.assertEqual([4,1,1,6], int_shape(split_a))
        self.assertEqual([4,1,1,6], int_shape(split_b))
        
        input_a, input_b, eps_b = self.split(squeezed)
        self.assertEqual([4,1,1,6], int_shape(input_a))
        self.assertEqual([4,1,1,6], int_shape(input_b))
        self.assertEqual([4,1,1,6], int_shape(eps_b))
        
        recon = self.split.invert(input_a, input_b)
        self.assertEqual([4,1,1,12], int_shape(recon))        
        
        # test for assertion errors, if layers are called on tensors
        # that do not fulfil the dimension requierements 
        inputs = tf.ones([4, 3, 3, 2])
        with self.assertRaises(AssertionError):
            self.squeeze(inputs)
            
        inputs = tf.ones([2, 2, 2, 3])
        with self.assertRaises(AssertionError):
            self.coupling(inputs)
        
        with self.assertRaises(AssertionError):
            self.split(inputs)
        
    
    def test_layer_inversion(self):
        inputs = tf.ones([4, 2, 2, 3])
        squeezed = self.squeeze(inputs)
        desqueezed = self.squeeze.invert(squeezed)
        self.assertLessEqual(L2NORM(inputs, desqueezed), 1e-5)
        
        actnormed = self.actn(inputs)
        deactnormed = self.actn.invert(actnormed)
        self.assertLessEqual(L2NORM(inputs, deactnormed), 1e-5)
        
        convolved = self.conv1x1(inputs)
        deconvolved = self.conv1x1.invert(convolved)
        self.assertLessEqual(L2NORM(inputs, deconvolved), 1e-5)
        convolved_lu = self.conv_lu(inputs)
        deconvolved_lu = self.conv_lu.invert(convolved_lu)
        self.assertLessEqual(L2NORM(inputs, deconvolved_lu), 1e-5)
        
        coupled = self.coupling(squeezed)
        decoupled = self.coupling.invert(coupled)
        self.assertLessEqual(L2NORM(squeezed, decoupled), 1e-5)
    
        input_a, input_b, eps_b = self.split(squeezed)
        split_1, _ = split_along_channels(squeezed)
        self.assertLessEqual(L2NORM(input_a, split_1), 1e-5)
        
        recon = self.split.invert(input_a, input_b)
        recon_1, _ = split_along_channels(recon)
        self.assertLessEqual(L2NORM(split_1, recon_1), 1e-5)     
        
    
    def test_add_losses(self):
        # Due to initialization Actnorm and Conv1x1  have 0 loss at the beginning
        inputs = tf.ones([4, 2, 2, 2])
        self.actn(inputs)
        [loss_acn] = self.actn.losses
        loss_expected = 0.0
        self.assertEqual(loss_expected, loss_acn.numpy())
        #print(loss_acn.numpy())

        self.conv1x1(inputs)
        [loss_conv] = self.conv1x1.losses
        # not exactly zero because the randomly initialized orthogonal matrix
        # has only det = 1 up to a small error
        self.assertLessEqual(L2NORM(loss_expected, loss_conv.numpy()), 1e-6)
        #print(loss_conv.numpy())
        
        loss_expected =  4. * tf.math.log(tf.nn.sigmoid(2.) + 1e-10)
        self.coupling(inputs)
        [loss_coupling] = self.coupling.losses
        self.assertLessEqual(L2NORM(loss_expected, loss_coupling.numpy()), 1e-6)
        #print(loss_coupling.numpy())
        
        # mean and log_var are initialized with zeros, hence loss boils down to 
        # log det of density of standard normal
        self.split(inputs)
        [loss_split] = self.split.losses
        dims = int_shape(inputs)
        factor =  float(dims[1] * dims[2] * dims[3]) / 2.
        loss_expected = -0.5 * factor * (1. ** 2. + tf.math.log(2. * np.pi))
        self.assertEqual(loss_expected.numpy(), loss_split.numpy())
        #print(loss_split.numpy())
        
        inputs = 2. * tf.ones([4, 2, 2, 2])
        self.split(inputs)
        [loss_split] = self.split.losses
        loss_expected = -0.5 * factor * (2. ** 2. + tf.math.log(2. * np.pi))
        self.assertEqual(loss_expected.numpy(), loss_split.numpy())
        
   
class TestCaseBlocks(unittest.TestCase):
    """
    Testing block components of the module 'blocks.py'
    """
    def setUp(self):
        ml = True 
        self.flow = FlowstepACN(ml=ml, lu_decom=False)
        self.flow_squeeze = FlowstepSqueeze(ml=ml, lu_decom=False)
        
    def test_block_in_and_output_shapes(self):
        inputs = tf.ones([4, 2, 2, 2])
        outputs = self.flow(inputs)
        self.assertEqual(int_shape(inputs), int_shape(outputs))
        
        outputs = self.flow_squeeze(inputs)
        self.assertEqual([4, 1, 1, 8], int_shape(outputs))
        
              
    def test_block_add_losses(self):
        inputs = tf.ones([4, 2, 2, 2])
        self.flow(inputs)
        self.assertEqual(len(self.flow.losses), 3)
        self.assertLessEqual(self.flow.losses[0], 1e-5)
        self.assertLessEqual(self.flow.losses[1], 1e-5)
        
        self.flow_squeeze(inputs)
        self.assertEqual(len(self.flow_squeeze.losses), 3)
        self.assertLessEqual(self.flow_squeeze.losses[0], 1e-5)
        self.assertLessEqual(self.flow_squeeze.losses[1], 1e-5)
        
                
    def test_block_inversion(self):
        inputs = tf.ones([4, 2, 2, 2])
        flowacn = self.flow(inputs)
        recon_flowacn = self.flow.invert(flowacn)
        self.assertLessEqual(L2NORM(inputs, recon_flowacn), 1e-5)
        
        flowsqueezed = self.flow(inputs)
        recon_flowsqueezed = self.flow.invert(flowsqueezed)
        self.assertLessEqual(L2NORM(inputs, recon_flowsqueezed), 1e-5)
        
    def test_data_dependent_init(self):
        inputs = tf.random.normal((4, 2, 2, 2), mean=3.0, stddev=2.5, seed=1)
        # before data dependent initialization, mean and stddev should be 
        # passed through 
        acn_output = self.flow.acn(inputs)
        post_stddev = tf.math.reduce_std(acn_output)
        post_mean = tf.math.reduce_mean(acn_output)    
        self.assertLessEqual(L2NORM(post_mean, 3.0), 0.75)
        self.assertLessEqual(L2NORM(post_stddev, 2.5), 0.75)
        # after data dependent init activation should be normalized
        self.flow.data_dependent_init(inputs)
        acn_output = self.flow.acn(inputs)
        post_stddev = tf.math.reduce_std(acn_output)
        post_mean = tf.math.reduce_mean(acn_output)
        self.assertLessEqual(L2NORM(post_mean, 0), 0.75)
        self.assertLessEqual(L2NORM(post_stddev, 1), 0.75)
        
        acn_output = self.flow_squeeze.acn(self.flow_squeeze.squeeze(inputs))
        post_stddev = tf.math.reduce_std(acn_output)
        post_mean = tf.math.reduce_mean(acn_output)    
        self.assertLessEqual(L2NORM(post_mean, 3.0), 0.75)
        self.assertLessEqual(L2NORM(post_stddev, 2.5), 0.75)
        self.flow_squeeze.data_dependent_init(inputs)
        acn_output = self.flow_squeeze.acn(self.flow_squeeze.squeeze(inputs))
        post_stddev = tf.math.reduce_std(acn_output)
        post_mean = tf.math.reduce_mean(acn_output)
        self.assertLessEqual(L2NORM(post_mean, 0), 0.75)
        self.assertLessEqual(L2NORM(post_stddev, 1), 0.75)
  
class TestCaseGlow(unittest.TestCase):
    """
    Testing the module 'glow.py'
    """
    def setUp(self):
        ml = False 
        self.encoder = Encoder(ml=ml)
        self.model = IRevNet(10, [28,28,1])
        #self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        t = tf.TensorShape([4, 28, 28, 1])
        self.model.build(t)
        
    def test_in_and_output_shapes(self):
        inputs = tf.ones([4, 4, 4, 3])
        # shapes are tranformed: 4,4,3 -> 2,2,12 -split- 2,2,6 -> 1,1,24 
        out_1, out_2 = self.encoder(inputs)
        self.assertEqual([4,1,1,24], int_shape(out_1))        
        self.assertEqual([4,2,2,6], int_shape(out_2))
        
        inputs = tf.ones((2, 28, 28, 1))
        out_logits, out_nuisance, out_nui_logits = self.model.call_all(inputs)
        self.assertEquals([2, 10], int_shape(out_logits))
        self.assertEquals([2, 28*28-10], int_shape(out_nuisance))
        self.assertEquals([2, 10], int_shape(out_nui_logits))
                    
    def test_encoder_inversion(self):
        inputs = tf.ones([4, 4, 4, 3])
        out_1, out_2 = self.encoder(inputs)
        inputs_reconstruct = self.encoder.invert(out_1, out_2)
        self.assertLessEqual(L2NORM(inputs, inputs_reconstruct), 1e-4)
        
        
    def test_encoder_data_dependent_init(self):
        inputs = tf.random.normal((4, 4, 4, 3), mean=5.0, stddev=3.0, seed=1)
        # check for third block as representive
        # after data dependent init activation should be normalized
        self.encoder.data_dependent_init(inputs)
        inputs = self.encoder.level_1[0](inputs)
        inputs = self.encoder.level_1[1](inputs)
        acn_output = self.encoder.level_1[2].acn(inputs)
        post_stddev = tf.math.reduce_std(acn_output)
        post_mean = tf.math.reduce_mean(acn_output)
        self.assertLessEqual(L2NORM(post_mean, 0), 0.75)
        self.assertLessEqual(L2NORM(post_stddev, 1), 0.75)
        
    def test_metameric_sampling(self):
        inputs = tf.random.normal((2, 28, 28, 1), mean=1.0, stddev=1.0, seed=1)
        self.model.call(inputs)
        out_logits, out_nuisance, out_nui_logits = self.model.call_all(inputs)
        input_recon = self.model.metameric_sampling(out_logits, out_nuisance)
        self.assertLessEqual(L2NORM(inputs, input_recon), 1e-4)
        
        
    
class TestCaseHelper(unittest.TestCase):
    """
    Testing the module 'helper.py'
    """
         
    def setUp(self):
        self.con = DetOne(3)
        
    def testShapes(self):
        inputs = tf.ones([2, 2, 2, 4])
        out_1, out_2 = split_along_channels(inputs)
        self.assertEqual(out_1.shape, out_2.shape)
        self.assertEqual([2,2,2,2], out_1.shape)
        
    def testLUDecomposition(self):
        w = tf.keras.backend.random_normal(shape=(3,3))
        p, l, u, diag = lu_decomposition(w)
        # Correct composition yields W = P * L * (U + diag)
        u_plus = tf.linalg.set_diag(u, diag)   
        l_with_diag = tf.linalg.set_diag(l, tf.ones(shape=diag.shape))  
        w_recon = tf.matmul(p, tf.matmul(l_with_diag, u_plus))
        self.assertLessEqual(L2NORM(w, w_recon), 1e-6)
        
    def testDetOneConstraint(self):
        s = tf.constant([1., 2., 3.])
        s_con = self.con(s)
        self.assertLessEqual(1. - tf.reduce_prod(s_con).numpy(), 1e-6)
        
if __name__ == '__main__':
    unittest.main()