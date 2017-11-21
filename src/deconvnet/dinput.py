import numpy as np
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K

class DInput(object):
    '''
    A class to define forward and backward operation on Input
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Input layer, whose configuration
                   will be used to initiate DInput(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer

    # input and output of Inputl layer are the same
    def up(self, data, learning_phase=0):
        '''
        function to operate input in forward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            data
        '''
        self.up_data = data
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to operate input in backward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            data
        '''
        self.down_data = data
        return self.down_data
