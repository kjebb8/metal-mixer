import os
import cPickle as pickle
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork import NeuralNetwork
from trainNeuralNetworkUtils import *
from mLConfig import MLConfig
from ..Separation_Config.separationConfig import SeparationConfig
from ..Separation_Config.separationConfigOutputs import SeparationConfigOutputs

'''
File: runNeuralNetwork.py
Brief: Script that takes input data and trains a neural network model.
         - Splits the input data into train and test set.
         - Tests the accuracy and prints the training time.
         - Save the trained weights to a file.
'''

def trainNeuralNetwork(mLConfig, mLDataInputsFolder, mLDataOutputsPath):

    dataFolderName = getMLDataOutputsFolderName(mLConfig, mLDataInputsFolder)
    mLDataOutputsFolder = mLDataOutputsPath + dataFolderName + "/"
    createMLDataOutputsFolder(mLDataOutputsFolder)

    np.random.seed(0)

    # 1. Import the data from the Input Data Path
    (xAll, yAll) = importInputData(mLDataInputsFolder)

    # 2. Divide the data into the Training Set and the Test Set
    (xTrain, yTrain, xTest, yTest) = getTrainAndTestSets(xAll, yAll)

    print("X training shape: " + str(xTrain.shape))
    print("Y training shape: " + str(yTrain.shape))
    print("X test shape: " + str(xTest.shape))
    print("Y test shape: " + str(yTest.shape))

    # 3. Train the network on the Training Set
    nn = NeuralNetwork(mLConfig.layerDims)

    startTime = time.time()
    costs = nn.trainModel(xTrain,
                          yTrain,
                          mLConfig.alpha,
                          mLConfig.lambd,
                          mLConfig.numIters,
                          mLConfig.miniBatchSize,
                          mLConfig.adamOn,
                          mLConfig.plotCostInterval)
    endTime = time.time()

    print("Time to Train the Model: " + str(endTime - startTime))

    plotCosts(costs, mLConfig.plotCostInterval)

    # 4. Get the prediction accuracy on the Training Set
    prediction = nn.predict(xTrain)
    accuracy = nn.getPredictionAccuracy(prediction, yTrain)
    print("The Training Set Accuracy is: " + str(accuracy))

    # 5. Get the prediction accuracy on the Test Set
    prediction = nn.predict(xTest)
    accuracy = nn.getPredictionAccuracy(prediction, yTest)
    print("The Test Set Accuracy is: " + str(accuracy))

    # 6. Save the trained model parameters to a file
    nn.saveParams(mLDataOutputsFolder)

    # 7. Save the mLConfig in the SeparationConfigOutputs
    saveSeparationConfig(mLConfig, mLDataInputsFolder, mLDataOutputsFolder)


def getMLConfig(mLDataInputsFolder):
    mLConfig = MLConfig()
    mLConfig.numHiddenLayers = 0
    mLConfig.alpha = 0.01
    mLConfig.lambd = 0.0
    mLConfig.numIters = 200
    mLConfig.miniBatchSize = None
    mLConfig.adamOn = True
    mLConfig.plotCostInterval = 2

    separationConfig = SeparationConfig.loadConfig(mLDataInputsFolder)
    numSpectrogramValues = separationConfig.genIn.numSpectrogramValues

    mLConfig.layerDims = [numSpectrogramValues]
    for _ in range(mLConfig.numHiddenLayers):
        mLConfig.layerDims.append(numSpectrogramValues)
    mLConfig.layerDims.append(numSpectrogramValues)

    return mLConfig


def getMLDataOutputsFolderName(mLConfig, mLDataInputsFolder):

    dataFolderName = ""

    for key, val in mLConfig.__dict__.iteritems():
        if key == "plotCostInterval":
            continue
        dataFolderName += str(key) + "-"
        if key == "layerDims":
            dataFolderName += str(val)[1:-1].replace(", ", "-")
        else:
            dataFolderName += str(val)
        dataFolderName += "_"

    dataFolderName += "inputs-"
    dataFolderName += mLDataInputsFolder.split("/")[-2]

    return dataFolderName


def createMLDataOutputsFolder(mLDataOutputsFolder):
    if not os.path.isdir(mLDataOutputsFolder):
        os.makedirs(mLDataOutputsFolder)
    else:
        print("Error: folder " + mLDataOutputsFolder + " already exists")
        assert(0)


def importInputData(mLDataInputsFolder):
    xAll = np.load(mLDataInputsFolder + "inputSpectrogramVecs.npy")
    yAll = np.load(mLDataInputsFolder + "idealMasks.npy")

    assert(xAll.dtype == np.uint16)
    assert(yAll.dtype == np.bool)

    xAll = (xAll.astype(np.float64) / \
           np.iinfo(np.uint16).max).astype(np.float64)
    assert(xAll.dtype == np.float64)
    assert(np.all(xAll <= 1) and np.all(xAll >= 0))

    yAll = yAll.astype(np.float64)
    assert(yAll.dtype == np.float64)
    assert(len(yAll[yAll == 1]) + len(yAll[yAll == 0]) == yAll.size)

    assert(xAll.shape == yAll.shape)

    return (xAll, yAll)


def plotCosts(costs, plotCostInterval):
    # print(costs)
    plt.plot(costs)
    plt.xlabel('Iterations (per ' + str(plotCostInterval) + ')' )
    plt.ylabel('Cost')
    plt.title('Learning Graph')
    plt.show()


def saveSeparationConfig(mLConfig, mLDataInputsFolder, mLDataOutputsFolder):
    separationConfig = SeparationConfigOutputs(mLConfig, mLDataInputsFolder, mLDataOutputsFolder)
    separationConfig.saveConfig()


def trainNeuralNetworkTest():
    print("\nTrain Neural Network Test Started\n")

    testPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Code/" + \
        "Machine_Learning/Train_NN_Test/"

    refDataFolder = testPath + "Test_Data_Ref/"
    with open(refDataFolder + "weights.pkl", 'rb') as input:
        weightsRef = pickle.load(input)
    with open(refDataFolder + "biases.pkl", 'rb') as input:
        biasesRef = pickle.load(input)
    separationConfigRef = SeparationConfigOutputs.loadConfig(refDataFolder)

    mLDataInputsFolder = testPath + "Test_Data_Inputs/"

    mLConfig = getMLConfigTest(mLDataInputsFolder)
    trainNeuralNetwork(mLConfig, mLDataInputsFolder, testPath)

    dataFolderName = \
        "miniBatchSize-25_lambd-0.5_layerDims-100-100_numIters-10_alpha-0.01_adamOn-True_numHiddenLayers-0_inputs-Test_Data_Inputs"
    mLDataOutputsFolder = testPath + dataFolderName + "/"
    with open(mLDataOutputsFolder + "weights.pkl", 'rb') as input:
        weights = pickle.load(input)
    with open(mLDataOutputsFolder + "biases.pkl", 'rb') as input:
        biases = pickle.load(input)
    separationConfig = SeparationConfigOutputs.loadConfig(mLDataOutputsFolder)

    shutil.rmtree(mLDataOutputsFolder)

    assert(np.array_equal(weights, weightsRef))
    assert(np.array_equal(biases, biasesRef))
    assert(separationConfig == separationConfigRef)

    print("\nTrain Neural Network Test Finished\n")


def getMLConfigTest(mLDataInputsFolder):
    mLConfig = MLConfig()
    mLConfig.numHiddenLayers = 0
    mLConfig.alpha = 0.01
    mLConfig.lambd = 0.5
    mLConfig.numIters = 10
    mLConfig.miniBatchSize = 25
    mLConfig.adamOn = True
    mLConfig.plotCostInterval = 2

    separationConfig = SeparationConfig.loadConfig(mLDataInputsFolder)
    numSpectrogramValues = separationConfig.genIn.numSpectrogramValues

    mLConfig.layerDims = [numSpectrogramValues]
    for _ in range(mLConfig.numHiddenLayers):
        mLConfig.layerDims.append(numSpectrogramValues)
    mLConfig.layerDims.append(numSpectrogramValues)

    return mLConfig


def main():
    print("\n\n----------Train Neural Network Starting----------\n\n")

    trainNeuralNetworkTest()

    mLDataInputsFolder = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/" + \
        "ML_Experiments/ML_Data/Exp_4/drums-piano_1_duration-200ms/"

    mLDataOutputsPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/" + \
        "ML_Experiments/ML_Data/Exp_4/drums-piano_1_duration-200ms/"

    mLConfig = getMLConfig(mLDataInputsFolder)
    # trainNeuralNetwork(mLConfig, mLDataInputsFolder, mLDataOutputsPath)

    print("\n\n----------Train Neural Network Finished----------\n\n")

if __name__ == "__main__": main()