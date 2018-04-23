# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:41:17 2018

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

cat = cv2.imread('cat.jpg')
print (cat.shape)

model = Sequential()
model.add(Convolution2D(1,3,3,input_shape=cat.shape))

cat_batch = np.expand_dims(cat,axis=0)
conv_cat = model.predict(cat_batch)

def visualize_cat(cat_batch):
    cat = np.squeeze(cat_batch,axis=[0,3])
    print (cat.shape)
    plt.pyplot.imshow(cat)
    plt.pyplot.colorbar()
    plt.pyplot.show()
    
def nice_cat_printer(model,cat):
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat2 = model.predict(cat_batch)
    conv_cat2 = np.squeeze(conv_cat2,axis=0)
    print (conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])
    print (conv_cat2.shape)
    plt.pyplot.imshow(conv_cat2)
    plt.pyplot.colorbar()
    plt.pyplot.show()

model = Sequential()
model.add(Convolution2D(1,3,3,input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(1,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
    
visualize_cat(conv_cat)
nice_cat_printer(model,cat)