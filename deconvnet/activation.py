import numpy as np
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K

class DActivation(object):
    '''
    A class to define forward and backward operation on Activation
    '''

    def __init__(self, layer, linear=False):
        '''
        # Arguments
            layer: an instance of Activation layer, whose configuration
                   will be used to initiate DActivation(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape=layer.output_shape)

        output = self.activation(input)
        # According to the original paper,
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = K.function(
            [input, K.learning_phase()], [output])
        self.down_func = K.function(
            [input, K.learning_phase()], [output])

    # Compute activation in forward pass
    def up(self, data, learning_phase=0):
        '''
        function to compute activation in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        '''
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    # Compute activation in backward pass
    def down(self, data, learning_phase=0):
        '''
        function to compute activation in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        '''
        self.down_data = self.down_func([data, learning_phase])
        return self.down_data
