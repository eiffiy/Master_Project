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


class PredictClass:

    def __init__(self):
        # init model
        self.model = Sequential()
        self.model = load_model("cnn_kaggle_50.h5")

    # print fixed information
    def StartPrintResult(self):
        print("**********************************")
        print("Facial Expression Prediction Start")
        print("**********************************")

    # relationship between label and expressions
    def label2string(self, label):
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

    # def predictTrainImgs():

    # this method uses for default cropped img in dir ./CroppedImgs/folderName/
    # *************************************
    def makePredictionFromFolder(self, num, folderName):
        # init data and predict_result np array
        data = np.empty((num, 1, 48, 48), dtype="float32")
        predict_result = np.empty((num,), dtype="uint8")
        for i in range(num):
            # this dir could need to change to a folder which use for cropped
            # imgs
            img = Image.open("./CroppedImgs/" + folderName + '/' + str(i) +
                             ".jpg", 'r').convert('L')
            img.thumbnail((48, 48))

            arr = np.asarray(img, dtype="float32")
            data[i, :, :, :] = arr

        predict_result = self.model.predict(data, batch_size=1, verbose=1)
        # print Prediction information
        self.StartPrintResult()

        # label array for return
        predict_label = []
        for i in range(num):
            # print every expresion
            predict_label.append(int(np.argmax(predict_result[i])))
            print('Image ' + str(i) + " could be " +
                  self.label2string(predict_label[i]))

        return ['Prediction is finish', predict_label]

    # analyze the expression from the input path, and img should be cropped
    # single facial img
    def makePredictionByName(self, path):
        data = np.empty((1, 1, 48, 48), dtype="float32")
        predict_result = np.empty((1,), dtype="uint8")
        img = Image.open(path, 'r').convert('L')
        img.thumbnail((48, 48))

        arr = np.asarray(img, dtype="float32")
        data[0, :, :, :] = arr

        predict_result = self.model.predict(data, batch_size=1, verbose=1)
        self.StartPrintResult()

        predict_label = int(np.argmax(predict_result))
        print("This expression could be " +
              self.label2string(predict_label))

        return ['Prediction is finish', predict_label]

    # the input is PIL.Image object, from the web camera, the cropped img
    # would not be saved
    def makePredictionFromCam(self, img):
        # init np array
        data = np.empty((1, 1, 48, 48), dtype="float32")
        predict_result = np.empty((1,), dtype="uint8")

        arr = np.asarray(img, dtype="float32")
        data[0, :, :, :] = arr

        predict_result = self.model.predict(data, batch_size=1, verbose=1)
        self.StartPrintResult()

        predict_label = int(np.argmax(predict_result))
        print("This expression could be " +
              self.label2string(predict_label))

        return ['Prediction is finish', predict_label]
