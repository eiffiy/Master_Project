#coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe
file:data.py
"""
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, Input, TimeDistributed
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
# from data import load_data

import os
from PIL import Image
import numpy as np

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
    if file[3] is 'S':
        label = 5
    return label

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    data = np.empty((213,1,256,256),dtype="float32")
    label = np.empty((213,),dtype="uint8")

    path = "/root/Master_Project/jaffe"
    imgs = os.listdir(path)
    num = len(imgs)
    for i in range(num):
        img = Image.open(path+"/"+imgs[i],"r")
        arr = np.asarray(img,dtype="float32")
        print (arr)
        print ("****************")
        data[i,:,:,:] = arr
        label[i] = int(label_reader(imgs[i]))
        print (label_reader(imgs[i]))
    return [data,label]



#加载数据
data, label = load_data()
data = data.reshape(213,1,256,256)
print(data.shape[0], ' samples')


#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 6)

###############
#开始建立CNN模型
###############

#生成一个model
model = Sequential()

model.add(Conv2D(32, kernel_size = 4, strides = 1, input_shape=(1,256,256), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, label, batch_size=5, nb_epoch=500, shuffle=True, verbose=1, validation_split=0.15)


