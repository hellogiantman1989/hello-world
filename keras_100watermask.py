# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:20:13 2018

@author: Administrator
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from PIL import Image
import os
from keras import backend as K
import matplotlib as plt
from keras.utils.vis_utils import plot_model
import numpy
import cv2
import keras.utils.vis_utils
import graphviz
import pydot
import pydot_ng

pos_num = 1000
neg_num = 1000
imageheight = 60
imagewidth = 60

imagepath_pos = r'E:\LiuBingLe_Data\Start_Programme\Face_Recognition\Sample\vs_crop\zrt_watermark_head'
imagepath_neg = r'E:\LiuBingLe_Data\Start_Programme\Face_Recognition\Sample\vs_crop\zrt_watermark_head_neg'

train_data = numpy.empty((pos_num+neg_num,imageheight*imagewidth))
train_data_pos = numpy.empty((pos_num,imageheight*imagewidth))
train_data_neg = numpy.empty((neg_num,imageheight*imagewidth))
label=numpy.empty(pos_num+neg_num,dtype=int)


def load_data_pos(dataset_path):
    images = os.listdir(dataset_path)
    for idx,imagename in enumerate(images[0:pos_num]):
        if imagename.endswith(('.bmp','.jpg')):
            img = Image.open(os.path.join(dataset_path,imagename))
            img_ndarray = numpy.array(img, dtype='float64')
            #img_ndarray = (img_ndarray-img_ndarray.min())/(img_ndarray.max()-img_ndarray.min())
            train_data_pos[idx] = numpy.ndarray.flatten(img_ndarray)
            
def load_data_neg(dataset_path):
    images = os.listdir(dataset_path)
    for idx,imagename in enumerate(images[0:neg_num]):
        if imagename.endswith(('.bmp','.jpg')):
            img = Image.open(os.path.join(dataset_path,imagename))
            img_ndarray = numpy.array(img, dtype='float64')
            #img_ndarray = (img_ndarray-img_ndarray.min())/(img_ndarray.max()-img_ndarray.min())
            train_data_neg[idx] = numpy.ndarray.flatten(img_ndarray)

def gen_label():
    label[0:pos_num]=0
    label[pos_num:pos_num+neg_num]=1
    label_y = np_utils.to_categorical(label,2)
    return label_y

def Net_model(lr=0.05,decay=1e-6,momentum=0.9):
    model = Sequential()
    model.add(Convolution2D(5, 5, 5,border_mode='valid',input_shape=(imageheight, imagewidth,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(10, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000)) #Full connection
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
#	model.compile(loss='categorical_crossentropy', optimizer=sgd)	
    return model

def train_model(model,X_train,Y_train,batch_size,nb_epoch):
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
#    model.save_weights('model_weights.h5',overwrite=True)
    return model
    
if __name__ == '__main__':
    
    load_data_pos(imagepath_pos)
    load_data_neg(imagepath_neg)
    train_data = numpy.concatenate((train_data_pos,train_data_neg))
    Y_train = gen_label()
    X_train = train_data.reshape(train_data.shape[0], imageheight, imagewidth,1)
    
#    model=Net_model()
    model = Sequential()
    model.add(Convolution2D(5, 3, 3,border_mode='valid',input_shape=(imageheight, imagewidth,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
'''    
    model.add(Convolution2D(5, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
'''     
    model.add(Convolution2D(20, 7, 7))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   
#   model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(500)) #Full connection
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    sgd = SGD(lr=0.5, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])	
    model.fit(X_train, Y_train, batch_size=30, nb_epoch=5,verbose=1)
    
#    intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[1].output])
    
    plot_model(model, to_file='keras_model1.png',show_shapes=True)

#可视化特征    
    modeltest = Model(input=model.input, output=model.layers[5].output)
    features = modeltest.predict(numpy.expand_dims(X_train[1050], axis=0))
    features1 = numpy.squeeze(features)
    features2 = features1[:,:,5]
    plt.pyplot.imshow(features2)
    plt.pyplot.colorbar()
    plt.pyplot.show()
    
#    plt.pyplot.plot(numpy.arange(0,3380),numpy.squeeze(features,axis=0))
    
   
    
    