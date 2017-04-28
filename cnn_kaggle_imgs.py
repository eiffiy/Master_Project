
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


def dataLoader():
    path = "./kaggle_img/All"
    imgs = os.listdir(path)
    num = len(imgs)

    data = np.empty((num, 1, 48, 48), dtype="float32")
    label = np.empty((num,), dtype="uint8")

    for i in range(num):
        img = Image.open(path + "/" + imgs[i], "r").convert('L')
        arr = np.asarray(img, dtype="float32")
        print (imgs[i])
        data[i, :, :, :] = arr
        label[i] = int(imgs[i].split(".")[1])
    # img.show()
    return [data, label]


data, label = dataLoader()
num = len(label)
data = data.reshape(num, 1, 48, 48)
print(data.shape[0], ' samples')

label = np_utils.to_categorical(label, 7)

model = Sequential()

model.add(Conv2D(32, kernel_size=3, padding='same',
                 input_shape=(1, 48, 48), data_format='channels_first'))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# converts 3D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

# sgd = optimizers.SGD(lr=0.1, decay=0.1, momentum=0.1, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data, label, batch_size=100, epochs=50,
          shuffle=True, verbose=1, validation_split=0.3)
