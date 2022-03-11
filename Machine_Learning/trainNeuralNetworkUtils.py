import numpy as np
import constants as const

'''
File: trainNeuralNetworkUtils.py
Brief: Extra utility functions for training neural network.
'''

def getTrainAndTestSets(xAll, yAll):

    numTestExamples = np.int(np.floor(xAll.shape[1] * const.testSetFraction))

    xTrain = xAll[:, numTestExamples:]
    yTrain = yAll[:, numTestExamples:]

    xTest = xAll[:, :numTestExamples]
    yTest = yAll[:, :numTestExamples]

    return (xTrain, yTrain, xTest, yTest)
