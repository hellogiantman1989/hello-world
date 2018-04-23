# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:33:04 2017

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import _pickle as pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


'''
    Train a simple deep NN on the MNIST dataset.
    Get to 98.30% test accuracy after 20 epochs (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 10


def read_data(data_file):  
    import gzip  
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f,encoding='bytes')  
    f.close()  
    train_x = train[0]  
    train_y = train[1]
    test_x = test[0]  
    test_y = test[1]  
    return train_x, train_y, test_x, test_y
    
    
# the data, shuffled and split between tran and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
train_x, train_y, test_x, test_y = read_data(r"E:\python_spider_file\py3\mnist.pkl.gz")
X_train = train_x
X_test = test_x
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)

model = Sequential()
model.add(Dense(input_dim=784, output_dim=128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms,metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

