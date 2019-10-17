# -*- coding: utf-8 -*-
"""
@author: tkapetano

This module comprises blocks to build invertible steps of flow.

"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from invertible_layers import Squeeze, Actnorm, Conv1x1, CouplingLayer
from helper import int_shape

class FlowstepACN(tf.keras.layers.Layer):
    """ This block performs a step of flow by applying activation normalization, 
        a 1x1 convolution and an affine coupling layer transformation.
        # Output shape: Same shape as input.
    """
    def __init__(self, name='flowstep_actnorm', ml=True, filters=(64, 64), **kwargs):
        super(FlowstepACN, self).__init__(name=name, **kwargs)
        self.acn = Actnorm(ml=ml)
        self.conv1x1 = Conv1x1(ml=ml)
        self.coupling = CouplingLayer(ml=ml, filters=filters)
        
    def call(self, inputs):
        y = self.acn(inputs)
        y = self.conv1x1(y)
        return self.coupling(y)
        
    def invert(self, outputs):
        x = self.coupling.invert(outputs)
        x = self.conv1x1.invert(x) 
        return self.acn.invert(x)
        
        
    def data_dependent_init(self, inputs):
        """Resets the weights of the activation normalization layer to have 
        normalized, i.e. mean 0 and stddev 1, outgoing activations. 
        Works also for subclasses with a preceeding squeeze layer.
        # Input is a single data batch
        # Output is the block's output after normalization (in order to iterate
        this procedure through several blocks)
        """
        in_activations = inputs
        if hasattr(self, 'squeeze'):
            in_activations = self.squeeze(in_activations)
        out_activations = self.acn(in_activations)

        # normalize
        shape = int_shape(out_activations)
        scale = tf.math.reduce_std(out_activations, axis=(0,1,2))
        scale = tf.reshape(scale, [1, 1, shape[-1]])
        scale = tf.math.reciprocal(scale)
        bias = tf.math.reduce_mean(out_activations, axis=(0,1,2))
        bias = tf.reshape(bias, [1, 1, shape[-1]])
        bias = -bias * scale 
        
        # reset the weights of acn layer
        self.acn.set_weights([scale, bias])       
     
        # calculate and return the new output values
        return self.call(inputs)
        
            
class FlowstepSqueeze(FlowstepACN):
    """ This block performs a step of flow by squeezing and then applying the 
        parent class FlowstepACN step of flow.
        # Input shape: Requires even spatial dimensions. 
        # Output shape: If input is a h x w x c tensor, output shape is h/2 x w/2 x 4*c.
    """
    squeeze = Squeeze()
    
    def __init__(self, name='flowstep_squeeze', ml=True,filters=(64,64), **kwargs):
        super(FlowstepSqueeze, self).__init__(name=name, ml=ml, filters=filters, **kwargs)
        
               
    def call(self, inputs):
        y = self.squeeze(inputs)
        return super(FlowstepSqueeze, self).call(y)
       
    def invert(self, outputs):
        x = super(FlowstepSqueeze, self).invert(outputs)
        return self.squeeze.invert(x)        
       
        
       

