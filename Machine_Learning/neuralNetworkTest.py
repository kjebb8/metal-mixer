import numpy as np
import h5py
import matplotlib.pyplot as plt
from neuralNetwork import NeuralNetwork

'''
File: neuralNetworkTest.py
Brief: Test the NeuralNetwork Class functions. Tests match the test cases on
       Coursera Deep Learning Specialization #1.
'''
def initializationTest():
    print("\nInitialization Test Start\n")

    layerDims = [2, 4, 1]
    numLayers = len(layerDims)
    np.random.seed(3)

    nn = NeuralNetwork(layerDims)
    nn.initializeParameters()

    for l in range(numLayers - 1):
        assert(np.array_equal(\
            np.around(nn.weights[l], decimals=8),
            np.around(initializationWeightRef1[l], decimals=8)))

    assert(np.array_equal(nn.biases[0], np.zeros((4, 1))))
    assert(np.array_equal(nn.biases[1], np.zeros((1, 1))))

    print("\nInitialization Test Finish\n")


def forwardPropTest():
    print("\nForward Prop Test Start\n")

    layerDims = [5, 4, 3, 1]
    np.random.seed(6)
    X = np.random.randn(5, 4)

    weights = [np.random.randn(4, 5)]
    biases = [np.random.randn(4, 1)]
    weights.append(np.random.randn(3, 4))
    biases.append(np.random.randn(3, 1))
    weights.append(np.random.randn(1, 3))
    biases.append(np.random.randn(1, 1))

    nn = NeuralNetwork(layerDims)
    nn.weights = weights
    nn.biases = biases

    (Z_cache, A_prev_cache, AL) = nn.forwardProp(X)

    assert(len(Z_cache) == 3)
    assert(len(A_prev_cache) == 3)

    assert(np.array_equal(np.around(AL, decimals=8),
                          np.around(forwardPropRef, decimals=8)))

    print("\nForward Prop Test Finish\n")


def computeCostTest():
    print("\nCompute Cost Test Start\n")

    layerDims = [5, 4, 3, 1]
    Y = np.array([[1, 1, 0]])
    AL = np.array([[0.8, 0.9, 0.4]])
    nn = NeuralNetwork(layerDims)
    cost = nn.computeCost(AL, Y)
    assert(np.around(cost, decimals=8) == 0.27977656)

    print("\nCompute Cost Test Finish\n")


def backPropTest():
    print("\nBack Prop Test Start\n")
    np.random.seed(3)
    layerDims = [4, 3, 1]
    Y = np.array([[1, 0]])

    A_prev_cache = [None] * 2
    Z_cache = [None] * 2
    weights = [None] * 2
    biases = [None] * 2

    AL = np.random.randn(1, 2)
    A_prev_cache[0] = np.random.randn(4, 2)
    weights[0] = np.random.randn(3, 4)
    biases[0] = np.random.randn(3, 1)
    Z_cache[0] = np.random.randn(3, 2)

    A_prev_cache[1] = np.random.randn(3, 2)
    weights[1] = np.random.randn(1, 3)
    biases[1] = np.random.randn(1, 1)
    Z_cache[1] = np.random.randn(1, 2)

    nn = NeuralNetwork(layerDims)
    nn.weights = weights
    nn.biases = biases
    (dA_prev_cache, dZ_cache, dW_cache, db_cache) = \
        nn.backProp(AL, Y, Z_cache, A_prev_cache)

    assert(np.array_equal(np.around(dW_cache[0], decimals=8),
                          np.around(backPropRefdW0, decimals=8)))
    assert(np.array_equal(np.around(db_cache[0], decimals=8),
                          np.around(backPropRefdb0, decimals=8)))
    assert(np.array_equal(np.around(dA_prev_cache[1], decimals=8),
                          np.around(backPropRefdA1, decimals=8)))

    print("\nBack Prop Test Finish\n")


def updateParamsTest():
    print("\nUpdate Params Test Start\n")
    np.random.seed(2)
    layerDims = [4, 3, 1]
    alpha = 0.1

    weights = [None] * 2
    biases = [None] * 2
    dW_cache = [None] * 2
    db_cache = [None] * 2

    weights[0] = np.random.randn(3, 4)
    biases[0] = np.random.randn(3, 1)
    weights[1] = np.random.randn(1, 3)
    biases[1] = np.random.randn(1, 1)

    np.random.seed(3)
    dW_cache[0] = np.random.randn(3, 4)
    db_cache[0] = np.random.randn(3, 1)
    dW_cache[1] = np.random.randn(1, 3)
    db_cache[1] = np.random.randn(1, 1)

    nn = NeuralNetwork(layerDims)
    nn.weights = weights
    nn.biases = biases

    nn.updateParams(dW_cache, db_cache, alpha)

    assert(np.array_equal(np.around(nn.weights[0], decimals=8),
                          np.around(updateParamsRefdW0, decimals=8)))
    assert(np.array_equal(np.around(nn.biases[0], decimals=8),
                          np.around(updateParamsRefdb0, decimals=8)))

    print("\nUpdate Params Test Finish\n")


def predictTest():
    print("\nPredict Test Start\n")
    np.random.seed(0)
    layerDims = [5, 4, 3]

    nn = NeuralNetwork(layerDims)
    nn.initializeParameters()

    X = np.random.randn(5, 4)
    prediction = nn.predict(X)

    assert(np.array_equal(prediction, predictRef))

    print("\nPredict Test Finish\n")


def accuracyTest():
    print("\nAccuracy Test Start\n")
    np.random.seed(1)
    layerDims = [5, 4, 3]

    nn = NeuralNetwork(layerDims)
    nn.initializeParameters()

    X = np.random.randn(5, 4)
    prediction = nn.predict(X)
    accuracy = nn.getPredictionAccuracy(prediction, predictRef)

    assert(accuracy == 7. / 12.)

    print("\nAccuracy Test Finish\n")


def trainModelTest():
    print("\nTrain Model Test Start\n")
    np.random.seed(1)
    layerDims = [5, 4, 3]
    plotCostInterval = 100

    nn = NeuralNetwork(layerDims)

    X = np.random.randn(5, 10)
    Y = np.random.randn(3, 10)
    Y[Y > 0.5] = 1
    Y[Y != 1] = 0

    costs = nn.trainModel(X, Y, 0.0075, 0, 2500, None, None, plotCostInterval, 10)
    # print(costs)
    # plt.plot(costs)
    # plt.xlabel('Iterations (per ' + str(plotCostInterval) + ')' )
    # plt.ylabel('Cost')
    # plt.title('Learning Graph')
    # plt.show()

    prediction = nn.predict(X)
    accuracy = nn.getPredictionAccuracy(prediction, Y)

    assert(np.around(accuracy, decimals=3) == 0.8)

    print("\nTrain Model Test Finish\n")


def trainModelTestCourse():

    '''
    To pass the test, set:
    newWeights = np.random.randn(self.layerDims[layer],
                                 self.layerDims[layer - 1]) / \
                                 np.sqrt(self.layerDims[layer-1])
    '''

    print("\nTrain Mode Test Course Start\n")

    np.random.seed(1)
    layerDims = [12288, 20, 7, 5, 1]

    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    dataPath = "/Users/keeganjebb/Documents/Programming_2/" + \
                 "Metal_Mixer/Code/Machine_Learning/"
    train_dataset = h5py.File(dataPath + 'train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # train set features
    train_y = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File(dataPath + 'test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    # index = 2
    # plt.imshow(train_x[index])
    # print ("y = " + str(train_y[0,index]) + ". It's a " + \
    #        classes[train_y[0,index]].decode("utf-8") +  " picture.")
    # plt.show()

    # Reshape the training and test examples
    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x / 255.
    test_x = test_x / 255.

    nn = NeuralNetwork(layerDims)
    nn.trainModel(train_x, train_y, 0.0075, 2500)

    prediction = nn.predict(train_x)
    accuracy = nn.getPredictionAccuracy(prediction, train_y)
    assert(np.around(accuracy, decimals=5) == 0.98565)

    prediction = nn.predict(test_x)
    accuracy = nn.getPredictionAccuracy(prediction, test_y)
    assert(np.around(accuracy, decimals=5) == 0.8)

    print("\nTrain Model Test Course Finish\n")


def computeCostL2RegTest():
    print("\nCompute Cost L2 Regularization Test Start\n")

    np.random.seed(1)

    layerDims = [3, 2, 3, 1]

    weights = []
    biases = []
    for layer in range(1, len(layerDims)):
        weights.append(np.random.randn(layerDims[layer], layerDims[layer - 1]))
        biases.append(np.random.randn(layerDims[layer], 1))

    Y = np.array([[1, 1, 0, 1, 0]])
    AL = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    lambd = 0.1

    nn = NeuralNetwork(layerDims)
    nn.weights = weights
    nn.biases = biases
    cost = nn.computeCost(AL, Y, lambd)
    assert(np.around(cost, decimals=11) == 1.78648594516)

    print("\nCompute Cost L2 Regularization Test Finish\n")


def backPropL2RegTest():
    print("\nBack Prop L2 Regularization Test Start\n")

    layerDims = [3, 2, 3, 1]
    AL = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    Y = np.array([[1, 1, 0, 1, 0]])
    lambd = 0.70

    nn = NeuralNetwork(layerDims)
    nn.weights = backPropL2_weights
    nn.biases = backPropL2_biases
    (_, _, dW_cache, _) = nn.backProp(AL, Y, backPropL2_Z_cache, backPropL2_A_prev_cache, lambd)

    for layer in range(len(backPropL2Ref)):
        assert(np.array_equal(np.around(dW_cache[layer], decimals=5),
                              np.around(backPropL2Ref[layer], decimals=5)))

    print("\nBack Prop L2 Regularization Test Finish\n")


def gradCheckTest():
    print("\nGrad Check Test Start\n")
    np.random.seed(1)
    layerDims = [5, 4, 3]
    lambd = 0.70

    X = np.random.randn(5, 10)
    Y = np.random.randn(3, 10)
    Y[Y > 0.5] = 1
    Y[Y != 1] = 0

    nn = NeuralNetwork(layerDims)
    nn.initializeParameters()
    nn.gradCheck(X, Y, lambd)

    print("\nGrad Check Test Finish\n")


def randMiniBatchTest():
    print("\nRandom Mini Batch Test Start\n")
    np.random.seed(1)
    layerDims = [5, 4, 3]

    X = np.random.randn(5, 32)
    Y = np.random.randn(3, 32)

    nn = NeuralNetwork(layerDims)
    miniBatches = nn.randomMiniBatches(X, Y, 10)
    assert(len(miniBatches) == 4)
    for batchIdx in range(0, len(miniBatches)):
        batchX = miniBatches[batchIdx][0]
        batchY = miniBatches[batchIdx][1]
        if batchIdx is not 3:
            assert(batchX.shape == (5, 10))
            assert(batchY.shape == (3, 10))
        else:
            assert(batchX.shape == (5, 2))
            assert(batchY.shape == (3, 2))

    print("\nRandom Mini Batch Test Finish\n")

def trainModelMiniBatchesTest():
    print("\nTrain Model Mini Batches Test Start\n")
    np.random.seed(1)
    layerDims = [5, 4, 3]
    plotCostInterval = 100

    nn = NeuralNetwork(layerDims)

    X = np.random.randn(5, 10)
    Y = np.random.randn(3, 10)
    Y[Y > 0.5] = 1
    Y[Y != 1] = 0

    costs = nn.trainModel(X, Y, 0.0075, 0, 500, 4, None, plotCostInterval, 10)
    # print(costs)
    # plt.plot(costs)
    # plt.xlabel('Iterations (per ' + str(plotCostInterval) + ')' )
    # plt.ylabel('Cost')
    # plt.title('Learning Graph')
    # plt.show()

    costsRef = [0.6551861053135983, 0.6794896383841215, 0.6915884752680705, 0.6107036607777991, 0.6769154206405951]

    assert(costs == costsRef)

    print("\nTrain Model Mini Batches Test Finish\n")


def updatelAdamTest():
    print("\nUpdate Adam Test Start\n")

    layerDims = [5, 4, 3]

    nn = NeuralNetwork(layerDims)
    nn.weights = [
        np.array([[ 1.62434536, -0.61175641, -0.52817175],
                  [-1.07296862,  0.86540763, -2.3015387 ]]),
        np.array([[ 0.3190391,  -0.24937038,  1.46210794],
                  [-2.06014071, -0.3224172,  -0.38405435],
                  [ 1.13376944, -1.09989127, -0.17242821]])
    ]

    nn.biases = [
        np.array([[ 1.74481176], [-0.7612069 ]]),
        np.array([[-0.87785842], [ 0.04221375], [ 0.58281521]])
    ]

    dW_cache = [
      np.array([[-1.10061918,  1.14472371,  0.90159072],
                [ 0.50249434,  0.90085595, -0.68372786]]),
      np.array([[-0.26788808,  0.53035547, -0.69166075],
                [-0.39675353, -0.6871727,  -0.84520564],
                [-0.67124613, -0.0126646,  -1.11731035]])
    ]

    db_cache = [
      np.array([[-0.12289023], [-0.93576943]]),
      np.array([[ 0.2344157 ], [ 1.65980218], [ 0.74204416]])
    ]

    adamParams = NeuralNetwork.AdamParams()
    nn.initializeAdam(adamParams)
    nn.updateParamsAdam(dW_cache, db_cache, 0.01, adamParams, 2)

    for layer in range(len(adamWeights)):
        assert(np.array_equal(np.around(nn.weights[layer], decimals=6),
                              np.around(adamWeights[layer], decimals=6)))
        assert(np.array_equal(np.around(nn.biases[layer], decimals=6),
                                np.around(adamBiases[layer], decimals=6)))

    print("\nUpdate Adam Test Finish\n")


def trainModelAdam():
    print("\nTrain Model Adam Test Start\n")
    np.random.seed(1)
    layerDims = [5, 4, 3]
    plotCostInterval = 100

    nn = NeuralNetwork(layerDims)

    X = np.random.randn(5, 10)
    Y = np.random.randn(3, 10)
    Y[Y > 0.5] = 1
    Y[Y != 1] = 0

    costs = nn.trainModel(X, Y, 0.0075, 0, 500, 5, True, plotCostInterval, 10)
    # print(costs)
    # plt.plot(costs)
    # plt.xlabel('Iterations (per ' + str(plotCostInterval) + ')' )
    # plt.ylabel('Cost')
    # plt.title('Learning Graph')
    # plt.show()

    costsRef = [0.6584912281050297, 0.41581982054185024, 0.19743734471946872, 0.11370499627801604,
                0.076180057840628]

    assert(costs == costsRef)

    print("\nTrain Model Adam Test Finish\n")


initializationWeightRef1 = [
    np.array([[ 1.78862847, 0.43650985], [ 0.09649747, -1.8634927 ],
             [-0.2773882, -0.35475898], [-0.08274148, -0.62700068]]),
    np.array([[-0.02190908, -0.23860902, -0.65693238,  0.44231119]])
]

forwardPropRef = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])

backPropRefdW0 = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                           [0, 0, 0, 0],
                           [0.05283652, 0.01005865, 0.01777766, 0.0135308]])
backPropRefdb0 = np.array([[-0.22007063], [0], [-0.02835349]])
backPropRefdA1 = np.array([[0.12913162, -0.44014127], [-0.14175655, 0.48317296],
                           [0.01663708, -0.05670698]])

updateParamsRefdW0 = np.array(\
    [[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
     [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
     [-1.0535704 , -0.86128581,  0.68284052,  2.20374577]])
updateParamsRefdb0 = np.array([[-0.04659241], [-1.28888275], [ 0.53405496]])

predictRef = np.array([[ 0.,  1.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.],
                       [ 1.,  1.,  1.,  1.]])

backPropL2_Z_cache = [
    np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
    np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
                [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
    np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]])
]
np.random.seed(1)
backPropL2_A_prev_cache = [
    np.random.randn(3, 5),
    np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],
                [ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]]),
    np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
]
backPropL2_weights = [
    np.array([[-1.09989127, -0.17242821, -0.87785842],
                [ 0.04221375,  0.58281521, -1.10061918]]),
    np.array([[ 0.50249434,  0.90085595], [-0.68372786, -0.12289023], [-0.93576943, -0.26788808]]),
    np.array([[-0.6871727 , -0.84520564, -0.67124613]])
]
backPropL2_biases = [
    np.array([[ 1.14472371], [ 0.90159072]]),
    np.array([[ 0.53035547], [-0.69166075], [-0.39675353]]),
    np.array([[-0.0126646]])
]
backPropL2Ref = [
    np.array([[-0.25604646,  0.12298827, -0.28297129], [-0.17706303, 0.34536094, -0.4410571 ]]),
    np.array([[ 0.79276486,  0.85133918], [-0.0957219, -0.01720463], [-0.13100772, -0.03750433]]),
    np.array([[-1.77691347, -0.11832879, -0.09397446]])
]
adamWeights = [
    np.array([[ 1.63178673, -0.61919778, -0.53561312], [-1.08040999,  0.85796626, -2.29409733]]),
    np.array([[ 0.32648047, -0.25681175,  1.46954931],
              [-2.05269934, -0.31497583, -0.37661298],
              [ 1.14121081, -1.09244991, -0.16498684]])
]
adamBiases = [
    np.array([[ 1.75225313], [-0.75376553]]),
    np.array([[-0.88529979], [ 0.03477238], [ 0.57537384]])
]

def main():
    print("\n\n----------Neural Network Test Starting----------\n\n")

    initializationTest()
    forwardPropTest()
    computeCostTest()
    backPropTest()
    updateParamsTest()
    predictTest()
    accuracyTest()
    trainModelTest()
    computeCostL2RegTest()
    backPropL2RegTest()
    gradCheckTest()
    randMiniBatchTest()
    trainModelMiniBatchesTest()
    updatelAdamTest()
    trainModelAdam()

    # trainModelTestCourse()
    print("\n\n----------Neural Network Test Finished----------\n\n")

if __name__ == "__main__": main()
