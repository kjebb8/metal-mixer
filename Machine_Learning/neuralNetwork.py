import os
import cPickle as pickle
import numpy as np
from neuralNetworkUtils import *

'''
File: neuralNetwork.py
Brief: Functions to train and apply a neural network for machine learning.
       Follows the code on the Coursera Deep Learning Specialization #1 closely.
'''
class NeuralNetwork:

    class AdamParams:
        def __init__(self, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon

            self.momentumWeights = []
            self.momentumBiases = []
            self.rmsWeights = []
            self.rmsBiases = []


    def __init__(self, layerDims):
        self.layerDims = layerDims
        self.weights = []
        self.biases = []

        print("Creating Neural Network with layerDims: " + str(layerDims))

        '''
        Layer dims - Input dimension | Hidden Layer Dimensions | Out Dimension
        Arrays at index 0:
            weights[0] and biases[0] - Weights applied to the input
            A_prev_cache[0] - Input X values
            Z_cache[0] - Linear combination for first hidden layer
        Sizes:
            weights and biases - number of layers - 1 (in between each layer)
            A_prev_cache - number of layers - 1 (Does not include output layer)
            Z_cache - number of layers - 1 (Does not include input layer)
        '''

    # Basic full batch training algorithm
    def trainModel(self, X, Y, alpha, lambd = 0, numIterations = 3000, miniBatchSize = None,
                   adamOn = False, plotCostInterval = None, checkGradIters = 0):

        computeCost = (plotCostInterval != None)
        costs = []

        self.initializeParameters()
        print("Number of weight layers: " + str(len(self.weights)))
        print("Weights shape: " + str(self.weights[0].shape))
        print("Number of bias layers: " + str(len(self.biases)))
        print("Biases shape: " + str(self.biases[0].shape))

        if adamOn:
            adamParams = self.AdamParams()
            self.initializeAdam(adamParams)

        if miniBatchSize is None:
            miniBatchSize = X.shape[1]

        miniBatches = self.randomMiniBatches(X, Y, miniBatchSize)
        numMiniBatches = len(miniBatches)
        print("numMiniBatches: " + str(numMiniBatches))

        for i in range(numIterations):
            batchX = miniBatches[i % numMiniBatches][0]
            batchY = miniBatches[i % numMiniBatches][1]

            if i < checkGradIters:
                print("Iteration: " + str(i))
                self.gradCheck(batchX, batchY, lambd)

            (Z_cache, A_prev_cache, AL) = self.forwardProp(batchX)
            (_, _, dW_cache, db_cache) = self.backProp(AL, batchY, Z_cache, A_prev_cache, lambd)
            if adamOn:
                self.updateParamsAdam(dW_cache, db_cache, alpha, adamParams, i + 1)
            else:
                self.updateParams(dW_cache, db_cache, alpha)

            if computeCost and (i % plotCostInterval == 0):
                cost = self.computeCost(AL, batchY, lambd)
                costs.append(cost)

        return costs


    def initializeParameters(self):
        # Initialize weights to small values so the activation has a large slope
        # Also good to avoid exploding gradients
        # Reasonable approach for shallow networks
        numLayers = len(self.layerDims)
        for layer in range(1, numLayers):
            if layer == numLayers - 1:
                # For sigmoid activation on final layer
                initFactor = np.sqrt(1. / self.layerDims[layer - 1])
            else:
                # For relu on inner hidden layers
                initFactor = np.sqrt(2. / self.layerDims[layer - 1])

            newWeights = np.random.randn(self.layerDims[layer], self.layerDims[layer - 1]) * \
                         initFactor
            self.weights.append(newWeights)

            newBias = np.zeros((self.layerDims[layer], 1))
            self.biases.append(newBias)


    def forwardLinear(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z


    def forwardActivation(self, Z, activation):
        if activation == Activations.sigmoid:
            A = sigmoid(Z)
        elif activation == Activations.relu:
            A = relu(Z)
        else:
            assert(0)

        return A


    def forwardProp(self, X):
        # Use ReLu for hidden layers and sigmoid for output layer
        numReluLayers = len(self.weights) - 1
        A = X
        Z_cache = []
        A_prev_cache = []

        for layer in range(numReluLayers):
            A_prev = A
            A_prev_cache.append(A_prev)
            Z = self.forwardLinear(A_prev, self.weights[layer], self.biases[layer])
            A = self.forwardActivation(Z, Activations.relu)
            Z_cache.append(Z)

        A_prev = A
        A_prev_cache.append(A_prev)
        Z = self.forwardLinear(A_prev, self.weights[-1], self.biases[-1])
        AL = self.forwardActivation(Z, Activations.sigmoid)
        Z_cache.append(Z)

        return (Z_cache, A_prev_cache, AL)


    def computeCost(self, AL, Y, lambd = 0):
        # Different cost function for multiclass classification
        # This one works for 1D and 2D where every value is independent
        # (i.e. Sum down the columns of Y does not equal 1)
        numYVals = Y.size
        cost = (-1. / numYVals) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        if lambd != 0:
            assert(self.weights)
            regCost = 0
            for layer in range(len(self.weights)):
                regCost += np.sum(np.square(self.weights[layer]))
            regCost *= lambd / (2 * numYVals)
            cost += regCost

        cost = np.squeeze(cost)

        return cost


    def backLinear(self, numYVals, dZ, A_prev, layer, lambd):
        # layer corresponds to the layer of dZ
        dW = 1. / numYVals * np.dot(dZ, A_prev.T) + lambd / numYVals * self.weights[layer]
        db = 1. / numYVals * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.weights[layer].T, dZ)

        return (dA_prev, dW, db)


    def backActivation(self, Z, dA, activation):
        if activation == Activations.sigmoid:
            dActivation = sigmoidBackward(Z)
        elif activation == Activations.relu:
            dActivation = reluBackward(Z)
        else:
            assert(0)
        dZ = dActivation * dA

        return dZ


    def backProp(self, AL, Y, Z_cache, A_prev_cache, lambd = 0):
        Y = Y.reshape(AL.shape)
        numParamLayers = len(self.weights)
        numYVals = Y.size

        dA_prev_cache = [None] * numParamLayers
        dZ_cache = [None] * numParamLayers
        dW_cache = [None] * numParamLayers
        db_cache = [None] * numParamLayers

        dAL = -Y / AL + (1 - Y) / (1 - AL)
        dZ_cache[-1] = self.backActivation(Z_cache[-1], dAL, Activations.sigmoid)
        (dA_prev_cache[-1], dW_cache[-1], db_cache[-1]) = \
            self.backLinear(numYVals, dZ_cache[-1], A_prev_cache[-1], numParamLayers - 1, lambd)

        for layer in reversed(range(0, numParamLayers - 1)):
            dZ_cache[layer] = self.backActivation(Z_cache[layer],
                                                  dA_prev_cache[layer + 1],
                                                  Activations.relu)
            (dA_prev_cache[layer], dW_cache[layer], db_cache[layer]) = \
                self.backLinear(numYVals, dZ_cache[layer], A_prev_cache[layer], layer, lambd)

        return (dA_prev_cache, dZ_cache, dW_cache, db_cache)


    def updateParams(self, dW_cache, db_cache, alpha):
        numParamLayers = len(self.weights)
        for layer in range(numParamLayers):
            self.weights[layer] = self.weights[layer] - alpha * dW_cache[layer]
            self.biases[layer] = self.biases[layer] - alpha * db_cache[layer]


    def randomMiniBatches(self, X, Y, miniBatchSize):

        numExamples = X.shape[1]

        assert(miniBatchSize > 0)
        assert(miniBatchSize <= numExamples)

        miniBatches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(numExamples))
        shuffledX = X[:, permutation]
        shuffledY = Y[:, permutation]

        # Step 2: Partition (shuffledX, shuffledY). Minus the end case.
        numFullMiniBatches = int(np.floor(numExamples / miniBatchSize))
        startBatchIdx = 0
        for _ in range(0, numFullMiniBatches):
            miniBatchX = shuffledX[:, startBatchIdx : startBatchIdx + miniBatchSize]
            miniBatchY = shuffledY[:, startBatchIdx : startBatchIdx + miniBatchSize]
            miniBatches.append((miniBatchX, miniBatchY))
            startBatchIdx += miniBatchSize

        # Step 3: Handling the end case (last mini-batch < miniBatchSize)
        if numExamples % miniBatchSize != 0:
            miniBatchX = shuffledX[:, startBatchIdx:]
            miniBatchY = shuffledY[:, startBatchIdx:]
            miniBatches.append((miniBatchX, miniBatchY))

        return miniBatches



    def initializeAdam(self, adamParams) :
        numParamLayers = len(self.weights)
        for layer in range(numParamLayers):
            adamParams.momentumWeights.append(np.zeros(self.weights[layer].shape))
            adamParams.momentumBiases.append(np.zeros(self.biases[layer].shape))
            adamParams.rmsWeights.append(np.zeros(self.weights[layer].shape))
            adamParams.rmsBiases.append(np.zeros(self.biases[layer].shape))


    def updateParamsAdam(self, dW_cache, db_cache, alpha, adamParams, numIter):
        numParamLayers = len(self.weights)
        for layer in range(numParamLayers):
            # Momentum calculation
            adamParams.momentumWeights[layer] = adamParams.beta1 * adamParams.momentumWeights[layer] + \
                                                (1 - adamParams.beta1) * dW_cache[layer]
            adamParams.momentumBiases[layer] = adamParams.beta1 * adamParams.momentumBiases[layer] + \
                                               (1 - adamParams.beta1) * db_cache[layer]

            # Momentum Bias correction
            momentumWeightsCorrected = adamParams.momentumWeights[layer] / \
                                       (1 - adamParams.beta1 ** numIter)
            momentumBiasesCorrected = adamParams.momentumBiases[layer] / \
                                      (1 - adamParams.beta1 ** numIter)
            # RMS calculation
            adamParams.rmsWeights[layer] = adamParams.beta2 * adamParams.rmsWeights[layer] + \
                                           (1 - adamParams.beta2) * dW_cache[layer] ** 2
            adamParams.rmsBiases[layer] = adamParams.beta2 * adamParams.rmsBiases[layer] + \
                                          (1 - adamParams.beta2) * db_cache[layer] ** 2

            # RMS Bias correction
            rmsWeightsCorrected = adamParams.rmsWeights[layer] / \
                                  (1 - adamParams.beta2 ** numIter)
            rmsBiasesCorrected = adamParams.rmsBiases[layer] / \
                                 (1 - adamParams.beta2 ** numIter)

            # Param Updates
            self.weights[layer] = self.weights[layer] - alpha * momentumWeightsCorrected / \
                                  (np.sqrt(rmsWeightsCorrected) + adamParams.epsilon)

            self.biases[layer] = self.biases[layer] - alpha * momentumBiasesCorrected / \
                                  (np.sqrt(rmsBiasesCorrected) + adamParams.epsilon)


    def gradCheck(self, X, Y, lambd):
        epsilon = 1e-7

        (Z_cache, A_prev_cache, AL) = self.forwardProp(X)
        (_, _, dW_cache, db_cache) = self.backProp(AL, Y, Z_cache, A_prev_cache, lambd)

        gradApprox = self.getGradApprox(X, Y, lambd, epsilon)

        grad = np.array([])
        for gradWLayer in dW_cache:
            grad = np.append(grad, gradWLayer.flatten())
        for gradBLayer in db_cache:
            grad = np.append(grad, gradBLayer.flatten())

        numerator = np.linalg.norm(gradApprox - grad)
        denominator = np.linalg.norm(gradApprox) + np.linalg.norm(grad)
        difference = numerator / denominator

        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + \
                  str(difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + \
                  str(difference) + "\033[0m")


    def getGradApprox(self, X, Y, lambd, epsilon):

        params = [self.weights, self.biases]
        gradApprox = np.array([])

        for paramType in params:
            for paramLayer in paramType:
                for paramI in range(paramLayer.shape[0]):
                    for paramJ in range(paramLayer.shape[1]):

                        paramVal = paramLayer[paramI][paramJ]
                        paramLayer[paramI][paramJ] += epsilon
                        (_, _, AL) = self.forwardProp(X)
                        costPlus = self.computeCost(AL, Y, lambd)
                        paramLayer[paramI][paramJ] = paramVal

                        paramLayer[paramI][paramJ] -= epsilon
                        (_, _, AL) = self.forwardProp(X)
                        costMinus = self.computeCost(AL, Y, lambd)
                        paramLayer[paramI][paramJ] = paramVal

                        newGradApprox = (costPlus - costMinus) / (2 * epsilon)
                        gradApprox = np.append(gradApprox, newGradApprox)

        return gradApprox


    def predict(self, X):
        (_, _, prediction) = self.forwardProp(X)
        prediction[prediction > 0.5] = 1
        prediction[prediction != 1] = 0

        return prediction


    def getPredictionAccuracy(self, prediction, Y):
        accuracy = np.sum(prediction == Y) / float(Y.size)

        return accuracy


    def saveParams(self, paramsPath):
        weightsPath = paramsPath + "weights.pkl"
        biasesPath = paramsPath + "biases.pkl"

        # Overwrites any existing file.
        with open(weightsPath, 'wb') as output:
            pickle.dump(self.weights, output, pickle.HIGHEST_PROTOCOL)

        with open(biasesPath, 'wb') as output:
            pickle.dump(self.biases, output, pickle.HIGHEST_PROTOCOL)


    def loadParams(self, paramsPath):

        weightsPath = paramsPath + "weights.pkl"
        biasesPath = paramsPath + "biases.pkl"

        if os.path.exists(weightsPath):
            with open(weightsPath, 'rb') as input:
                self.weights = pickle.load(input)
        else:
            print("Cannot load weights. No file " + weightsPath)

        if os.path.exists(biasesPath):
            with open(biasesPath, 'rb') as input:
                self.biases = pickle.load(input)
        else:
            print("Cannot load weights. No file " + biasesPath)