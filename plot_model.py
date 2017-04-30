
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

import csv
import StringIO

import os
from PIL import Image
import numpy as np
import keras

from keras.models import load_model
from keras.utils import plot_model


model = Sequential()

model = load_model("cnn_kaggle_50.h5")
plot_model(model, to_file='model.png')