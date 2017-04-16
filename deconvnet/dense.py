import numpy as np
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K

class DDense(object):
    '''
    A class to define forward and backward operation on Dense
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Dense layer, whose configuration
                   will be used to initiate DDense(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DDense
        input = Input(shape=layer.input_shape[1:])
        output = Dense(output_dim=layer.output_shape[1],
                       weights=[W, b])(input)
        self.up_func = K.function([input, K.learning_phase()], [output])

        # Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape=self.output_shape[1:])
        output = Dense(
            output_dim=self.input_shape[1],
            weights=flipped_weights)(input)
        self.down_func = K.function([input, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        '''
        function to compute dense output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Result of dense layer
        '''
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to compute dense output in backward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Result of reverse dense layer
        '''
        # data = data - self.bias
        self.down_data = self.down_func([data, learning_phase])
        return self.down_data
