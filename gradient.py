from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from keras.utils import plot_model
import os
import matplotlib.pyplot as plt
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
        kept_filters = []

        for filter_index in range(filter_num):
            # we only scan through the first 200 filters,
            # but there are actually 512 of them
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
                kept_filters.append((img, loss_value))
            end_time = time.time()
            print('Filter %d processed in %ds' %
                  (filter_index, end_time - start_time))

        return kept_filters

    def show_image(self, kept_filters, grid_size=2):
        # we will stich the best 9 filters on a 8 x 8 grid.
        n = grid_size

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = n * self.img_width + (n - 1) * margin
        height = n * self.img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                stitched_filters[(self.img_width + margin) * i: (self.img_width + margin) * i + self.img_width,
                                 (self.img_height + margin) * j: (self.img_height + margin) * j + self.img_height, :] = img

        # display the image
        plt.imshow(stitched_filters)
        plt.show()
        # save the result to disk
        imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
