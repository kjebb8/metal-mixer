from enum import Enum
import numpy as np

'''
File: neuralNetworkUtils.py
Brief: Extra utility functions for the neural network that don't need to be
       part of the NeuralNetwork Class.
'''
class Activations(Enum):
    sigmoid = 0
    relu = 1


def sigmoid(Z):
    return (1 / (1 + np.exp(-Z)))


def relu(Z):
    return (np.maximum(0, Z))


def sigmoidBackward(Z):
    sig = sigmoid(Z)
    dSig = sig * (1 - sig)
    return dSig


def reluBackward(Z):
    dRelu = np.full(Z.shape, 1.)
    dRelu[Z <= 0] = 0
    return dRelu
