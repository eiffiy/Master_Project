
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


def label_reader(file_string):
    file = file_string
    label = 0
    if file[3] is 'A':
        label = 0
    if file[3] is 'D':
        label = 1
    if file[3] is 'F':
        label = 2
    if file[3] is 'H':
        label = 3
    if file[3] is 'N':
        label = 4
    if file[3] is 'S' and file[4] is 'A':
        label = 5
    if file[3] is 'S' and file[4] is 'U':
        label = 6
    return label


def load_data_shared(path="/Users/eiffiy/Master_Project/jaffe"):

    data = np.empty((213, 1, 48, 48), dtype="float32")
    label = np.empty((213,), dtype="uint8")

    files = os.listdir(path)

    j = 0

    for file in files:
        if file is ".DS_Store":
            continue
        if not os.path.isdir(file):
            label[j] = int(label_reader(file))
            img = Image.open(path + "/" + file, "r")
            img.thumbnail((48, 48))
            arr = np.asarray(img, dtype="float32")
            data[j, :, :, :] = arr
            j = j + 1
    print (label)
    return [data, label]


data, label = load_data_shared()
data = data.reshape(213, 1, 48, 48)

label = np_utils.to_categorical(label, 7)

model = Sequential()

model.add(Conv2D(32, kernel_size=3, padding='same',
                 input_shape=(1, 48, 48), data_format='channels_first'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

# converts 3D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

# sgd = optimizers.SGD(lr=0.01, decay=0.01, momentum=0.1, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data, label, batch_size=10, epochs=50,
          shuffle=True, verbose=1, validation_split=0.2)
