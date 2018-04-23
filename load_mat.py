# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:18:34 2017

@author: Administrator
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from pybrain.datasets import ClassificationDataSet,SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer,FullConnection
from pybrain.structure import *
from pybrain.utilities import percentError

#matlab文件名
matfn=r'E:\LiuBingLe_Data\Start_Programme\Face_Recognition\Sample\OCR\in_out_data_20171103.mat'
data=sio.loadmat(matfn)

input = data['in_put']
output = data['out_put']

# definite the dataset
DS = SupervisedDataSet(150,34)

# add data element to the dataset
for i in np.arange(len(input)):
    DS.addSample(input[i],output[i])
    
# you can get your input/output this way
X = DS['input']
Y = DS['target']

# createa neural network
fnn = FeedForwardNetwork()

# create three layers
inLayer = LinearLayer(150,name='inLayer')
hiddenLayer0 = SigmoidLayer(50,name='hiddenLayer0')
outLayer = SigmoidLayer(34,name='outLayer')

# add three layers to the neural network
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer0)
fnn.addOutputModule(outLayer)

# link three layers
in_to_hidden0 = FullConnection(inLayer,hiddenLayer0)
hidden0_to_out = FullConnection(hiddenLayer0, outLayer)

# add the links to neural network
fnn.addConnection(in_to_hidden0)
fnn.addConnection(hidden0_to_out)

# make neural network come into effect
fnn.sortModules()


# train the NN
# we use BP Algorithm
# verbose = True means print th total error
trainer = BackpropTrainer(fnn, DS, verbose=True,learningrate=0.01)
# set the epoch times to make the NN  fit
trainer.trainUntilConvergence(maxEpochs=10)



NetworkWriter.writeToFile(net, 'srnNet.xml')

for mod in fnn.modules:
  print ("Module:", mod.name)
  if mod.paramdim > 0:
    print ("--parameters:", mod.params)
  for conn in fnn.connections[mod]:
    print ("-connection to", conn.outmod.name)
    if conn.paramdim > 0:
       print ("- parameters", conn.params)
  if hasattr(fnn, "recurrentConns"):
    print ("Recurrent connections")
    for conn in fnn.recurrentConns:
       print ("-", conn.inmod.name, " to", conn.outmod.name)
       if conn.paramdim > 0:
          print ("- parameters", conn.params)


