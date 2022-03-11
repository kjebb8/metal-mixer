import numpy as np

'''
File: equalLoudness.py
Brief: Approximate human hearing of loudness as a function of frequency. Use
       linear interpolation of an equal loudness curve to correct the power of a
       particualr frequency to what a human would hear.

       Humans hear certain frequencies more easily with a constant sound
       pressure dB. It also varies with the dB of the sound. This model takes
       the average change in real dB vs perceived dB across 4 dB levels (done
       in the calculateAverageEqualLoudnessCorrection function and saved as
       equalLoudnessCorrectionPerFreq) to approxiamte the change. The function
       used by spectrograms to apply equal loudness is
       getEqualLoudnessCorrection.
'''
def getEqualLoudnessCorrection(freqVec):
    freqVec[freqVec == 0] = 1
    freqVecEql = np.copy(freqVec)
    freqVecEql = freqVecEql[freqVecEql < np.max(freqRange)]
    logFreq = np.log10(freqVecEql)

    closestIdx = np.zeros((len(logFreq),)).astype(int)
    for i in range(len(logFreq)):
        closestIdx[i] = (np.abs(logFreqRange - logFreq[i])).argmin()

    lowerIdx = np.copy(closestIdx)
    lowerIdx[logFreq < logFreqRange[closestIdx]] -= 1

    upperIdx = closestIdx
    upperIdx[logFreq >= logFreqRange[closestIdx]] += 1

    # Linear interpolation
    slope = (equalLoudnessCorrectionPerFreq[upperIdx] - \
                equalLoudnessCorrectionPerFreq[lowerIdx]) / \
            (logFreqRange[upperIdx] - \
                logFreqRange[lowerIdx])
    deltaF = logFreq - logFreqRange[lowerIdx]

    correction = np.zeros((len(freqVec), 1))
    correction[:len(freqVecEql), 0] = \
        slope * deltaF + equalLoudnessCorrectionPerFreq[lowerIdx]

    return correction


def calculateAverageEqualLoudnessCorrection():
    # Use the middle yValues to create the interpolation model
    yValsUsed = np.array([20, 40, 60, 80])
    inputVals = np.array([twentyLevelDb, fortyLevelDb,
                          sixtyLevelDb, eightyLevelDb]).astype(float).T
    averageCorrectionPerFreq = np.zeros((len(freqRange),))
    averageCorrectionPerFreq = \
        np.sum(yValsUsed - inputVals, axis=1) / len(yValsUsed)

    logFreqRange = np.log10(freqRange)
    # plt.plot(logFreqRange, averageCorrectionPerFreq)
    # plt.show()
    # print(averageCorrectionPerFreq)
    # print(logFreqRange)


# Regression version that didn't really work. Makes more sense to do linear
# interpolation to solve this problem.
def equalLoudnessRegression():

    featureExponents = 3
    numFeatures = 4 + 2 * featureExponents
    yVals = [0, 20, 40, 60, 80, 100]
    numExamplesPerY = len(zeroLevelDb)
    numExamples = numExamplesPerY * len(yVals)
    dbInLevels = [zeroLevelDb, twentyLevelDb, fortyLevelDb, sixtyLevelDb,
                  eightyLevelDb, hundredLevelDb]

    X = np.zeros((numExamples, numFeatures))
    W = np.zeros((numFeatures,))
    Y = np.zeros((numExamples,))

    X[:, 0] = 1
    for y in range(len(yVals)):
        startYIdx = y * numExamplesPerY
        endYIdx = (y + 1) * numExamplesPerY
        Y[startYIdx : endYIdx] = yVals[y]
        X[startYIdx : endYIdx, 1] = dbInLevels[y]
        X[startYIdx : endYIdx, 2] = freqRange
        X[startYIdx : endYIdx, 3] = np.log10(freqRange)

    xIdx = 4
    for i in range(2, featureExponents + 2):
        X[:, xIdx] = X[:, 1] ** i
        xIdx += 1
        X[:, xIdx] = np.log10(X[:, 2]) ** i
        xIdx += 1

    W[:] = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)


freqRange = np.array( \
[
1, 20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,
4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000
])

logFreqRange = np.array( \
[
0, 1.30103, 1.47712125, 1.60205999, 1.69897, 1.77815125, 1.84509804, 1.90308999,
1.95424251, 2., 2.30103, 2.47712125, 2.60205999, 2.69897, 2.77815125,
2.84509804, 2.90308999, 2.95424251, 3., 3.30103, 3.47712125, 3.60205999,
3.69897, 3.77815125, 3.84509804, 3.90308999, 3.95424251, 4., 4.04139269,
4.07918125, 4.11394335, 4.14612804, 4.17609126, 4.20411998, 4.23044892
])
equalLoudnessCorrectionPerFreq = np.array( \
[
-150., -57., -45.25, -39.25, -35., -31.25, -28., -26., -24.5, -23., -11.25,
-6.5,-4., -2.5, -1.5, -0.75, -0.25, 0., 0., 1.5, 3.5, 1.5, -2.25, -7.75, -11.25,
-12.25, -13., -13., -11.75, -10.75, -9.5, -8.5, -8., -8., -8.
])

zeroLevelDb = np.array( \
[
112,80,64,53,48,41,36,32,30,28,15,10,7,4,3,2,1,0,0,-2,-7,-3,1,8,12,14,14,13,12,
11,10,9,8,8,8
])

twentyLevelDb = np.array( \
[
170,92,78,71,66,62,57,55,53,51,36,29,26,24,23,22,21,20,20,18,15,17,21,27,32,33,
34,35,34,33,32,31,30,30,30
])

fortyLevelDb = np.array( \
[
190,103,90,83,79,75,72,69,68,67,54,48,45,43,42,41,40,40,40,39,36,38,42,48,51,52,
53,52,51,50,49,48,48,48,48
])

sixtyLevelDb = np.array( \
[
210,111,101,96,92,88,86,84,82,80,70,66,63,62,61,60,60,60,60,58,57,59,63,68,71,
72,73,74,72,71,69,68,67,67,67
])

eightyLevelDb = np.array( \
[
230,122,112,107,103,100,97,96,95,94,85,83,82,81,80,80,80,80,80,79,78,80,83,88,
91,92,92,91,90,89,88,87,87,87,87
])

hundredLevelDb = np.array( \
[
250,130,121,118,114,112,110,109,108,107,102,101,100,100,100,100,100,100,100,98,
97,100,104,108,110,111,112,110,109,108,107,106,107,107,107
])
