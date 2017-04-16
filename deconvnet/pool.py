import numpy as np
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K

class DPooling(object):
    '''
    A class to define forward and backward operation on Pooling
    '''

    def __init__(self, layer):
        '''
        # Arguments
            layer: an instance of Pooling layer, whose configuration
                   will be used to initiate DPooling(input_shape,
                   output_shape, weights)
        '''
        self.layer = layer
        self.poolsize = layer.pool_size

    def up(self, data, learning_phase=0):
        '''
        function to compute pooling output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Pooled result
        '''
        [self.up_data, self.switch] = \
            self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data, learning_phase=0):
        '''
        function to compute unpooling output in backward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Unpooled result
        '''
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data

    def __max_pooling_with_switch(self, input, poolsize):
        '''
        Compute pooling output and switch in forward pass, switch stores
        location of the maximum value in each poolsize * poolsize block
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
        # Returns
            Pooled result and Switch
        '''
        switch = np.zeros(input.shape)
        out_shape = list(input.shape)
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        out_shape[1] = out_shape[1] / poolsize[0]
        out_shape[2] = out_shape[2] / poolsize[1]
        pooled = np.zeros(out_shape)

        for sample in range(input.shape[0]):
            for dim in range(input.shape[3]):
                for row in range(out_shape[1]):
                    for col in range(out_shape[2]):
                        patch = input[sample,
                                      row * row_poolsize: (row + 1) * row_poolsize,
                                      col * col_poolsize: (col + 1) * col_poolsize,
                                      dim]
                        max_value = patch.max()
                        pooled[sample, row, col, dim] = max_value
                        max_col_index = patch.argmax(axis=1)
                        max_cols = patch.max(axis=1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample,
                               row * row_poolsize + max_row,
                               col * col_poolsize + max_col,
                               dim] = 1
        return [pooled, switch]

    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):
        '''
        Compute unpooled output using pooled data and switch
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
            switch: switch storing location of each elements
        # Returns
            Unpooled result
        '''
        tile = np.ones((switch.shape[1] / input.shape[1],
                        switch.shape[2] / input.shape[2]))
        input = input.transpose((0, 3, 1, 2))
        switch = switch.transpose((0, 3, 1, 2))
        out = np.kron(input, tile)
        unpooled = out * switch
        unpooled = unpooled.transpose((0, 2, 3, 1))
        return unpooled
