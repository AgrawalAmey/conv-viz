from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from keras.utils import plot_model
import os
import matplotlib.pyplot as plt
from keras.layers import Conv2D

from utils import *


class GradientVisualizer(object):
    def __init__(self, model):
        self.model = model
        # dimensions of the generated pictures for each filter.
        self.img_width = 224
        self.img_height = 224

    def process_layer(self, layer_num):
        # Select the layer
        layer_output = self.model.layers[layer_num].output

        # Find number of filters
        if K.image_data_format() == 'channels_first':
            filter_num = layer_output.get_shape().as_list()[1]
        else:
            filter_num = layer_output.get_shape().as_list()[3]

        print("The number of filters in the layer: ", filter_num, sep="")

        # this is the placeholder for the input images
        input_img = self.model.input

        # To be returned
        output = []

        # for filter_index in range(filter_num):
        for filter_index in range(2):
            print('Processing filter %d' % filter_index)
            start_time = time.time()

            # we build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some random noise
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random(
                    (1, 3, self.img_width, self.img_height))
            else:
                input_img_data = np.random.random(
                    (1, self.img_width, self.img_height, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # we run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    break

            # decode the resulting input image
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                output.append(
                    [img, loss_value, self.model.layers[layer_num].name, filter_index])
            else:
                output.append(
                    [np.zeros_like(img), loss_value, self.model.layers[layer_num].name, filter_index])
            end_time = time.time()

            print('Filter %d processed in %ds' %
                  (filter_index, end_time - start_time))

        return np.asarray(output)

    def save_images(self, output):
        for i in range(len(output)):
            imsave('./static/viz/{}_{}.png'.format(
                output[i][2], output[i][3]), output[i][0])

        return output[:, 1:]

    def process_net(self):
        results = []
        for layer_num in range(3):
            # for layer_num in range(len(self.model.layers)):
            if isinstance(self.model.layers[layer_num], Conv2D):
                output = self.process_layer(layer_num )
                # Show image
                results.append(self.save_images(output))

        return np.asarray(results)
