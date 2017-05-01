from __future__ import absolute_import
from __future__ import print_function
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, Input, TimeDistributed
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.models import load_model

import os
import csv
import StringIO

from PIL import Image
import numpy as np
import keras


def label2string(label):
    if label is 0:
        return "ANGRY"
    if label is 1:
        return "DISGUST"
    if label is 2:
        return "FEAR"
    if label is 3:
        return "HAPPY"
    if label is 4:
        return "SAD"
    if label is 5:
        return "SURPRISE"
    if label is 6:
        return "NEUTRAL"


def make_prediction(num):
    data = np.empty((num, 1, 48, 48), dtype="float32")
    predict_label = np.empty((num,), dtype="uint8")

    for i in range(num):
        img = Image.open("./" + str(i) + ".jpg", 'r').convert('L')
        img.thumbnail((48, 48))

        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr

    model = Sequential()
    model = load_model("cnn_kaggle_50.h5")

    predict_label = model.predict(data, batch_size=1, verbose=1)
    print (predict_label)
    print("**********************************")
    print("Facial Expression Prediction Start")
    print("**********************************")
    for i in range(num):
        print(str(i) + ".jpg could be " +
              label2string(int(np.argmax(predict_label[i]))))
    return "finish"


def make_prediction_Byname(str_name):
    data = np.empty((1, 1, 48, 48), dtype="float32")
    predict_label = np.empty((1,), dtype="uint8")
    img = Image.open("./" + str_name, 'r').convert('L')
    img.thumbnail((48, 48))

    arr = np.asarray(img, dtype="float32")
    data[0, :, :, :] = arr

    model = Sequential()
    model = load_model("cnn_kaggle_50.h5")

    predict_label = model.predict(data, batch_size=1, verbose=1)
    print("**********************************")
    print("Facial Expression Prediction Start")
    print("**********************************")
    print("This image could be " + slabel2string(int(np.argmax(predict_label))))
    return "finish"
