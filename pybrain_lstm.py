# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:55:09 2017

@author: Administrator
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LSTMLayer,LinearLayer

net = buildNetwork(3,3,3, hiddenclass=LSTMLayer,outclass=LinearLayer, bias=True, recurrent=True)
dataSet = SupervisedDataSet(3, 3)
dataSet.addSample((0, 0, 0), (0, 0, 0))
dataSet.addSample((1, 1, 1), (0, 0, 0))
dataSet.addSample((1, 0, 0), (1, 0, 0))
dataSet.addSample((0, 1, 0), (0, 1, 0))
dataSet.addSample((0, 0, 1), (0, 0, 1))

trainer = BackpropTrainer(net, dataSet)
trained = False
acceptableError = 0.001

howmanytries = 0
# train until acceptable error reached
while (trained == False) and (howmanytries < 1000):
    error = trainer.train()
    if error < acceptableError :
        trained = True
    else:
        howmanytries += 1

result = net.activate([0.5, 0.4, 0.7])
net['in'].outputbuffer[net['in'].offset]
net['hidden0'].outputbuffer[net['hidden0'].offset]
print result