# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:26:07 2018

@author: Administrator
"""

import scipy.io as scio
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy

dataFile = r'E:\LiuBingLe_Data\Start_Programme\Face_Recognition\Sample\OCR\ocr_2000_in_out.mat'
data = scio.loadmat(dataFile)

X = data['in_put']
Y = data['out_put']

# create model
'''
model = Sequential()
model.add(Dense(20, input_dim=150, init='uniform', activation='sigmoid'))
model.add(Dense(34, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=150, batch_size=10)
'''
model = Sequential()
model.add(Dense(20, input_dim=150, init='uniform', activation='relu'))
model.add(Dense(34, init='uniform', activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
hist = model.fit(X, Y, nb_epoch=50, batch_size=30,callbacks=[early_stopping])
print(hist.history)

first_layer_weights = model.layers[0].get_weights()[0]
first_layer_biases  = model.layers[0].get_weights()[1]
second_layer_weights = model.layers[1].get_weights()[0]
second_layer_biases  = model.layers[1].get_weights()[1]

first_layer_weights = first_layer_weights.transpose()
first_layer_biases = first_layer_biases.transpose()
second_layer_weights = second_layer_weights.transpose()
second_layer_biases = second_layer_biases.transpose()


scio.savemat('savewb_keras_srn_20180124.mat', {'w1': first_layer_weights,'b1': first_layer_biases,'w2': second_layer_weights,'b2': second_layer_biases})  