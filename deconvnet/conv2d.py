import numpy as np
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K

class DConvolution2D(object):
    '''
    A class to define forward and backward operation on Convolution2D
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Convolution2D layer, whose configuration
                   will be used to initiate DConvolution2D(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer

        weights = layer.get_weights()
        config = layer.get_config()

        W = weights[0]
        b = weights[1]

        # Set up_func for DConvolution2D
        input = Input(shape=layer.input_shape[1:])
        output = Conv2D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            padding='same',
            weights=[W, b]
        )(input)
        self.up_func = K.function([input, K.learning_phase()], [output])

        # Flip W horizontally and vertically,
        # and set down_func for DConvolution2D
        W = np.transpose(W, (0, 1, 3, 2))
        W = W[::-1, ::-1, :, :]
        config['filters'] = W.shape[3]
        b = np.zeros(config['filters'])

        input = Input(shape=layer.output_shape[1:])
        output = Conv2D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            padding='same',
            weights=[W, b]
        )(input)
        self.down_func = K.function([input, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        '''
        function to compute Convolution output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Convolved result
        '''
        self.up_data = self.up_func([data, learning_phase])
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to compute Deconvolution output in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Deconvolved result
        '''
        self.down_data = self.down_func([data, learning_phase])
        return self.down_data
