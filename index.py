from Tkinter import Tk
from tkFileDialog import askopenfilename
from keras.models import load_model
from keras.applications import *
import numpy as np
from gradient import *
from deconv import *
from tsne import *
import os

# Make it run on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        model = vgg16.VGG16(weights='imagenet', include_top=True)

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
    print("1. Gradient acent to maximize activation")
    print("2. Deconvolution keeping all gradient activation")
    print("3. Deconvolution keeping only max gradient activation")
    print("4. t-SNE visualize of last fully connected layer")

    # Take input
    viz_mode = input('Please enter the visalization mode number: ')
    viz_mode = int(viz_mode)

    # Visualize
    if(viz_mode == 1):
        # Init
        gv = GradientVisualizer(model)
        # Process layers
        kept_filters = gv.process_layer(layer_num)
        # Show image
        gv.show_image(kept_filters)
        # Delete
        del gv
    elif(viz_mode == 2 or viz_mode == 3):
        # Get the filter number
        filter_num = input('Please enter the filter number to be visualized: ')
        filter_num = int(filter_num)
        # Select the image
        print('Please select the image')
        # we don't want a full GUI, so keep the root window from appearing
        Tk().withdraw()
        # show an "Open" dialog box and return the path to the selected file
        filename = askopenfilename()
        # Init the object
        if(viz_mode == 2):
            dv = DeconvVisualizer(model, layer_num, filter_num, "all")
        else:
            dv = DeconvVisualizer(model, layer_num, filter_num, "max")
        # Load imgae
        image = dv.load_image(filename)
        # Forward
        encoding = dv.forward(image)
        # Backward
        output = dv.backward(encoding)
        # Show
        dv.show_image(output)
        # Delete
        del dv
    elif(viz_mode == 4):
        last_layer = model.layers[-1]
        if isinstance(last_layer, Dense) or isinstance(last_layer, Activation) or isinstance(last_layer, Flatten):
            ts = TSNE(model)
            ts.plot()
            del ts
        else:
            print('Cannot handle this type of layer')
            print(model.layers[i].get_config())
            sys.exit()

    print("-------------------------------------------------------------------")
    contin = input('Enter 1 to continue or 0 to exit: ')
    contin = int(contin)
