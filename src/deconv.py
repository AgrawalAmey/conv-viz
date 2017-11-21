import numpy as np
import sys
from PIL import Image
from keras.layers import *
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K
from keras.applications import imagenet_utils
from deconvnet import *
import matplotlib.pyplot as plt
from scipy.misc import imsave, imshow

class DeconvVisualizer(object):
    def __init__(self, model, layer_name, filter_to_visualize, visualize_mode):
        self.deconv_layers = []
        self.visualize_mode = visualize_mode
        self.layer_name = layer_name
        self.filter_to_visualize = filter_to_visualize
        # Stack layers
        for i in range(len(model.layers)):
            if isinstance(model.layers[i], Conv2D):
                self.deconv_layers.append(
                    dconv2d.DConvolution2D(model.layers[i]))
                self.deconv_layers.append(
                    dactivation.DActivation(model.layers[i]))
            elif isinstance(model.layers[i], MaxPooling2D):
                self.deconv_layers.append(dpool.DPooling(model.layers[i]))
            elif isinstance(model.layers[i], Dense):
                self.deconv_layers.append(ddense.DDense(model.layers[i]))
                self.deconv_layers.append(
                    dactivation.DActivation(model.layers[i]))
            elif isinstance(model.layers[i], Activation):
                self.deconv_layers.append(
                    dactivation.DActivation(model.alyers[i]))
            elif isinstance(model.layers[i], Flatten):
                self.deconv_layers.append(dflatten.DFlatten(model.layers[i]))
            elif isinstance(model.layers[i], InputLayer):
                self.deconv_layers.append(dinput.DInput(model.layers[i]))
            else:
                print('Cannot handle this type of layer')
                print(model.layers[i].get_config())
                sys.exit()
            if layer_name == model.layers[i].name:
                break

    def load_image(self, filename):
        # Open the image and preprocess
        img = Image.open(filename)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array[np.newaxis, :]
        img_array = img_array.astype(np.float)
        img_array = imagenet_utils.preprocess_input(img_array)

        return img_array

    def forward(self, data):
        # Forward pass
        self.deconv_layers[0].up(data)
        for i in range(1, len(self.deconv_layers)):
            data = np.asarray(self.deconv_layers[i - 1].up_data)
            if(len(data.shape) == 5):
                data = data.reshape(
                    (data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
            self.deconv_layers[i].up(data)

        output = np.asarray(self.deconv_layers[-1].up_data)
        if(len(output.shape) == 5):
            output = output.reshape(
                (output.shape[1], output.shape[2], output.shape[3], output.shape[4]))
        assert output.ndim == 2 or output.ndim == 4
        if output.ndim == 2:
            feature_map = output[:, self.filter_to_visualize]
        else:
            feature_map = output[:, :, :, self.filter_to_visualize]
        if 'max' == self.visualize_mode:
            max_activation = feature_map.max()
            temp = feature_map == max_activation
            feature_map = feature_map * temp
        elif 'all' != self.visualize_mode:
            print('Illegal visualize mode')
            sys.exit()
        output = np.zeros_like(output)
        if 2 == output.ndim:
            output[:, self.filter_to_visualize] = feature_map
        else:
            output[:, :, :, self.filter_to_visualize] = feature_map

        return output

    def backward(self, output):
        # Backward pass
        self.deconv_layers[-1].down(output)
        for i in range(len(self.deconv_layers) - 2, -1, -1):
            data = np.asarray(self.deconv_layers[i + 1].down_data)
            if(len(data.shape) == 5):
                data = data.reshape(
                    (data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
            self.deconv_layers[i].down(data)
        deconv = self.deconv_layers[0].down_data
        deconv = deconv.squeeze()

        return deconv

    def show_image(self, data):
        # Postprocess
        data = data - data.min()
        data *= 1.0 / (data.max() + 1e-8)
        data = data[:, :, ::-1]
        data = data * 255

        # display the image
        imshow(data)

        # Save image
        imsave('{}_{}_{}.png'.format(self.layer_name,
                                     self.filter_to_visualize, self.visualize_mode), data)
