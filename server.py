import os
from flask import Flask, request, redirect, url_for, jsonify, render_template
import json
from werkzeug.utils import secure_filename
from Tkinter import Tk
from tkFileDialog import askopenfilename
from keras.models import load_model
from keras.applications import *
import numpy as np
from gradient import *
from deconv import *
from tsne import *
import keras.backend as K
from keras.layers import Conv2D, InputLayer, Dense

# Make it run on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='', static_folder="static")

MODELS_UPLOAD_FOLDER = "./uploaded_models/"
PICTURES_UPLOAD_FOLDER = "./uploaded_pictures/"


def get_layer_data(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, InputLayer) or isinstance(layer, Dense):
            temp = {}
            temp['name'] = layer.name
            if isinstance(layer, Dense):
                temp['shape'] = [layer.output.get_shape().as_list()[1], 1, 1]
            else:
                temp['shape'] = layer.output.get_shape().as_list()[1:]
            layers.append(temp)
    return layers


@app.route('/')
def root():
    return app.send_static_file('upload_file.html')


@app.route('/upload_model', methods=['POST'])
def upload_file():
    if request.form['modelType'] == "1":
        f = request.files['model']
        filename = MODELS_UPLOAD_FOLDER + secure_filename(f.filename)
        f.save(filename)
        model = load_model(filename)
    else:
        model = vgg16.VGG16(weights='imagenet', include_top=True)

    if request.form['vizType'] == "1":
        f = request.files['picture']
        f.save(PICTURES_UPLOAD_FOLDER + secure_filename(f.filename))

    layer_data = get_layer_data(model)
    layer_data = json.dumps(layer_data)


    if request.form['vizType'] == "0":
        gv = GradientVisualizer(model)
        filter_data = gv.process_net()
        filter_data = filter_data.tolist()

        for i in range(len(filter_data)):
            for j in range(len(filter_data[i])):
                filter_data[i][j][0] = str(filter_data[i][j][0])
                filter_data[i][j][2] = str(filter_data[i][j][2])

        filter_data = json.dumps(filter_data)

        return render_template('canvas.html', layer_data=layer_data, filter_data=filter_data)


    return "hello"






app.run()
