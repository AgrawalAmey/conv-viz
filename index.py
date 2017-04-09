from Tkinter import Tk
from tkFileDialog import askopenfilename
from keras.models import load_model
import numpy as np
from activation import *

# just to that the loop starts
contin = 1

while contin:
    # Quick and hacky menu
    print("-------------------------------------------------------------------")
    print("                    Welcome To CNN Viz Toolkit                     ")
    print("-------------------------------------------------------------------")
    print("Please select one of the input mode:")

    print("1. Load a keras model from hdf5 file.")
    print("2. Load the default VGG16 model.")

    # Take input
    mode_num = input('Please enter the option: ')
    mode_num = int(mode_num)

    if(mode_num == 1):
        # we don't want a full GUI, so keep the root window from appearing
        Tk().withdraw()
        # show an "Open" dialog box and return the path to the selected file
        filename = askopenfilename()
        # Load model
        model = load_model(filename)
    else:
        model = vgg16.VGG16(weights='imagenet', include_top=False)

    print("-------------------------------------------------------------------")

    print("Model loaded.")
    model.summary()
    plot_model(model)
    plot_model(model, to_file='model.png')

    print("-------------------------------------------------------------------")

    print('Please select the layer to be visualized:')
    for i in range(len(model.layers[1:])):
        print("{}. {}".format(i + 1, model.layers[i + 1].name))

    layer_num = input('Please enter the option: ')
    layer_num = int(layer_num)

    print("-------------------------------------------------------------------")

    print("Select the visalization mode:")
    print("1. Max activation")

    # Take input
    viz_mode = input('Please enter the visalization mode number: ')
    viz_mode = int(viz_mode)

    # Visualize
    if(viz_mode == 1):
        av = ActivationVisualizer(model)
        kept_filters = av.process_layer(layer_num)
        av.show_filter_viz(kept_filters)

    print("-------------------------------------------------------------------")
    contin = input('Enter 1 to continue or 0 to exit: ')
    contin = int(contin)
