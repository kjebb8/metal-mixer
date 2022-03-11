import numpy as np
import matplotlib.pyplot as plt
import spectrogramUtils as specUtils
import constants as const
import equalLoudness as eql
from frequencyAnalyzer import FreqAnalyzer
from spectrogramParams import SpectrogramParams
from spectrogram import Spectrogram
from spectrogramInputs import SpectrogramInputs

'''
File: spectrogramTest.py
Brief: Test the Spectrogram Class functions.
'''
def testFunction1(time, freq):
    timeHalfLength = int(len(time) / 2)
    timeFirstHalf = time[:timeHalfLength]
    testFuncFirstHalf = np.cos(2 * np.pi * freq * timeFirstHalf) * \
                        np.sin(2 * np.pi * freq / 2 * timeFirstHalf) / 2

    timeSecondHalf = time[timeHalfLength:]
    testFuncSecondHalf = np.cos(2 * np.pi * freq / 2 * timeSecondHalf) * \
                         np.sin(2 * np.pi * freq / 4 * timeSecondHalf)
    return np.append(testFuncFirstHalf, testFuncSecondHalf)


def testFunction2(time, freq):
    timeHalfLength = int(len(time) / 2)
    timeFirstHalf = time[:timeHalfLength]
    timeSecondHalf = time[timeHalfLength:]
    testFuncFirstHalf = np.zeros((timeFirstHalf.shape))
    testFuncSecondHalf = np.zeros((timeSecondHalf.shape))
    twoPi = 2 * np.pi
    for fq in range(30):
        testFuncFirstHalf += np.sin(twoPi * fq * freq * timeFirstHalf) * \
            5 * (30 - fq) * np.cos(twoPi * fq * freq / 4 * timeFirstHalf)

        testFuncSecondHalf += np.cos(twoPi * fq * freq * timeSecondHalf) * \
            6 * fq * np.sin(twoPi * fq * freq / 3 * timeSecondHalf)
    return np.append(testFuncFirstHalf, testFuncSecondHalf)


def testFunction3(time, freq):
    testFunc = np.zeros((time.shape))
    twoPi = 2 * np.pi
    for fq in range(30):
        testFunc += np.sin(twoPi * fq * freq * time) * \
            5 * (30 - fq) * np.cos(twoPi * fq * freq / 4 * time)

        testFunc += np.cos(twoPi * fq * freq * time) * \
            6 * fq * np.sin(twoPi * fq * freq / 3 * time)

        testFunc += np.cos(twoPi * fq * time) * \
            6 * fq * np.sin(twoPi * fq / 3 * time)
    return testFunc * 100


def testFunction4(time, freq):
    testFunc = np.zeros((time.shape))
    twoPi = 2 * np.pi
    for fq in range(9):
        testFunc += np.sin(twoPi * 2**fq * freq * time)
    mask = np.zeros((time.shape))
    for i in range(len(time)/3000):
        mask[i * 3000 : i * 3000 + 500] = 1
    testFunc *= mask
    return testFunc * 150


def testFunction5(time):
    testFunc = np.zeros((time.shape))
    maxFreq = 16000
    numFreq = 80
    twoPi = 2 * np.pi

    for fq in range(1, numFreq):
        freq = 10 ** (fq * np.log10(maxFreq) / numFreq)
        testFunc += np.sin(twoPi * freq * time)
    return testFunc * 1000


def testFunction6(time, freq, ampRatios):
    testFunc = np.zeros((time.shape))
    twoPi = 2 * np.pi

    for ratio in ampRatios:
        testFunc += ratio * np.sin(twoPi * freq * time)
        freq *= 2
    return testFunc


def testFunctionTimeRes(time, freq, res, sampleRate):
    testFunc = np.zeros((time.shape))
    twoPi = 2 * np.pi
    for fq in range(9):
        testFunc += np.sin(twoPi * 2**fq * freq * time)

    resSamples = specUtils.getSamplesfromTime(sampleRate, res)
    intervalSamples = resSamples * 2
    numIntervals = len(time) / intervalSamples
    mask = np.zeros((time.shape))
    for i in range(numIntervals):
        mask[i * intervalSamples : i * intervalSamples + resSamples] = 1
    testFunc *= mask
    return testFunc


def testFunctionFreqRes(time, freq, res):
    timeHalfLength = int(len(time) / 2)
    timeFirstHalf = time[:timeHalfLength]
    timeSecondHalf = time[timeHalfLength:]
    testFuncFirstHalf = np.zeros((timeFirstHalf.shape))
    testFuncSecondHalf = np.zeros((timeSecondHalf.shape))
    twoPi = 2 * np.pi
    testFuncFirstHalf += np.cos(twoPi * freq * timeFirstHalf)

    testFuncSecondHalf += np.cos(twoPi * (freq + res) * timeSecondHalf)

    return np.append(testFuncFirstHalf, testFuncSecondHalf)


# Check the spectrogram parameters
def spectrogramTest1():
    print("\nSpectrogram Test 1 Start\n")

    testFreq = 12000
    sampleRate = 44100.
    numWindowSamples = 2 ** 9  # 512
    overlap = 0.7

    targetSpectrogramTimeSec = 0.1
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction1(timeVec, testFreq)

    # Visualize the time signal in Freq domain
    freqVec = specUtils.getFreqVectorFromSamples(sampleRate,
                                                 numInputSamples)
    freqSignal = FreqAnalyzer.fft(timeSignal)
    positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)

    assert(freqSignalNorm.shape == positiveFreqVec.shape)
    # plt.plot(positiveFreqVec, freqSignalNorm)
    # plt.show()

    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.numWindowSamples = numWindowSamples
    spectrogramInputs.overlap = overlap

    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    # spectrogram.printSpectrogramParams()

    # plt.plot(spectrogram.freqLabelsList[0],
    #          spectrogram.powerDbComponentsList[0][:, 0])
    # plt.show()
    # plt.plot(spectrogram.freqLabelsList[0],
    #          spectrogram.powerDbComponentsList[0][:, -1])
    # plt.show()
    # spectrogram.plotSpectrogram()

    assert(spectrogram.params.hopSize == 153)
    assert(spectrogram.params.numSpectrogramSamples == 4337)
    assert(round(spectrogram.params.freqResolution, 2) == \
        round(sampleRate / numWindowSamples, 2))
    assert(spectrogram.powerDbComponentsList[0].shape == (257, 26))

    assert(np.array_equal(np.around(spectrogram.powerDbComponentsList[0][70, :],
                          decimals=4),
                          np.around(test1PowerRef, decimals=4)))

    print("\nSpectrogram Test 1 Finish\n")


# Check the regeneration of the time signal from the spectrogram
def spectrogramTest2():
    print("\nSpectrogram Test 2 Start\n")

    testFreq = 500
    sampleRate = 44100.
    numWindowSamples = 2 ** 9  # 512

    targetSpectrogramTimeSec = 0.3
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction2(timeVec, testFreq)

    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.numWindowSamples = numWindowSamples

    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    regeneratedTimeSignal = spectrogram.getTimeRepresentation()

    # Due to the Hanning window, the first and last half windows are not fully
    # reconstructed to the original magnitude, but it is corrected to the
    # last few samples
    numSamples = spectrogram.params.numSpectrogramSamples

    # Check the percent error
    averageError = \
        np.sum(np.abs(regeneratedTimeSignal - timeSignal[:numSamples])) / \
                      numSamples
    averageValue = np.sum(np.abs(timeSignal[:numSamples])) / numSamples
    # print(averageError / averageValue)
    assert(averageError < 0.012 * averageValue)

    print("\nSpectrogram Test 2 Finish\n")


# Check compute the composite spectrogram and check time representation
def spectrogramTest3():
    print("\nSpectrogram Test 3 Start\n")

    testFreq = 500
    sampleRate = 30000.
    numWindowSamples = 2 ** np.array([7, 8, 11, 13, 5])

    targetSpectrogramTimeSec = 1.0
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction3(timeVec, testFreq)

    # Visualize the time signal in Freq domain
    freqVec = specUtils.getFreqVectorFromSamples(sampleRate,
                                                 numInputSamples)
    freqSignal = FreqAnalyzer.fft(timeSignal)
    positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)

    # plt.plot(positiveFreqVec, freqSignalNorm)
    # plt.show()

    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.numWindowSamples = numWindowSamples

    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    # spectrogram.printSpectrogramParams()

    # Check the params for the composite spectrogram
    pars = spectrogram.params

    assert(pars.numSegments == len(numWindowSamples))
    assert(np.all(pars.hopSize == numWindowSamples / 2))
    assert(pars.numInputSamples - pars.numSpectrogramSamples < pars.hopSize[0])
    assert(np.all(pars.numWindowSamples + (pars.numFftWindows - 1) * \
           pars.hopSize == pars.numSpectrogramSamples))
    assert(np.all(pars.segmentFreqIdx == \
                  [[0, 40], [10, 80], [10, 20], [10, 40], [10, 17]]))
    assert(np.all(pars.numSegmentFreq == [40, 70, 10, 30, 7]))
    assert(pars.freqResolution[-1] * (pars.numPositiveFreq[-1] - 1) == \
           pars.segmentStartFreq[-1])
    assert(pars.segmentFreqIdx[-1][-1] == pars.numPositiveFreq[-1])


    # spectrogram.plotSpectrogram()

    regeneratedTimeSignal = spectrogram.getTimeRepresentation()

    # Hanning window correction less effective. Don't use 50 samples on the ends
    sampleCutoff = 50
    numSamples = spectrogram.params.numSpectrogramSamples - 2 * sampleCutoff
    regeneratedTimeSignal = regeneratedTimeSignal[sampleCutoff : -sampleCutoff]
    timeSignal = timeSignal[sampleCutoff : numSamples + sampleCutoff]

    # Check the percent error
    averageError = \
        np.sum(np.abs(regeneratedTimeSignal - timeSignal)) / \
                      numSamples
    averageValue = np.sum(np.abs(timeSignal)) / numSamples
    # print(averageError/averageValue)
    assert(averageError < 0.19 * averageValue)

    print("\nSpectrogram Test 3 Finish\n")


# Check compute the composite spectrogram with "optimal" segments and check the
# time representation
def spectrogramTest4():
    print("\nSpectrogram Test 4 Start\n")

    testFreq = 400
    sampleRate = 2 ** 15. # 32768.
    numWindowSamples = const.optWindowSamples32kHz

    targetSpectrogramTimeSec = 1.0
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction4(timeVec, testFreq)

    # Visualize the time signal in Freq domain
    freqVec = specUtils.getFreqVectorFromSamples(sampleRate,
                                                 numInputSamples)
    freqSignal = FreqAnalyzer.fft(timeSignal)
    positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)

    # plt.plot(positiveFreqVec, freqSignalNorm)
    # plt.show()

    spectrogramInputs = SpectrogramInputs(True)

    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    # spectrogram.printSpectrogramParams()

    # Check the params for the composite spectrogram
    pars = spectrogram.params

    assert(pars.numSegments == len(numWindowSamples))
    assert(np.all(pars.hopSize == numWindowSamples / 2))
    assert(pars.numInputSamples - pars.numSpectrogramSamples < pars.hopSize[0])
    assert(np.all(pars.numWindowSamples + (pars.numFftWindows - 1) * \
           pars.hopSize == pars.numSpectrogramSamples))
    assert(np.all(pars.segmentFreqIdx == \
                  [[0, 64], [32, 96], [48, 126], [63, 126], [63, 96],
                   [48, 65]]))
    assert(np.all(pars.numSegmentFreq == [64, 64, 78, 63, 33, 17]))
    assert(pars.freqResolution[-1] * (pars.numPositiveFreq[-1] - 1) == \
           pars.segmentStartFreq[-1])
    assert(pars.segmentFreqIdx[-1][-1] == pars.numPositiveFreq[-1])

    # spectrogram.plotSpectrogram()

    regeneratedTimeSignal = spectrogram.getTimeRepresentation()

    # Hanning window correction less effective. Don't use 50 samples on the ends
    sampleCutoff = 50
    numSamples = spectrogram.params.numSpectrogramSamples - 2 * sampleCutoff
    regeneratedTimeSignal = regeneratedTimeSignal[sampleCutoff : -sampleCutoff]
    timeSignal = timeSignal[sampleCutoff : numSamples + sampleCutoff]

    # Check the percent error
    averageError = \
        np.sum(np.abs(regeneratedTimeSignal - timeSignal)) / \
                      numSamples
    averageValue = np.sum(np.abs(timeSignal)) / numSamples
    # print(averageError/averageValue)
    assert(averageError < 0.17 * averageValue)

    print("\nSpectrogram Test 4 Finish\n")


# Check music correction including equal loundness and clipping
def spectrogramTest5():
    print("\nSpectrogram Test 5 Start\n")

    sampleRate = 2 ** 15. # 32768.
    numWindowSamples = const.optWindowSamples32kHz

    targetSpectrogramTimeSec = 1.0
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction5(timeVec)

    # Visualize the time signal in Freq domain
    # freqVec = specUtils.getFreqVectorFromSamples(sampleRate,
    #                                              numInputSamples)
    # freqSignal = FreqAnalyzer.fft(timeSignal)
    # positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    # positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    # freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)
    #
    # plt.plot(positiveFreqVec, freqSignalNorm)
    # plt.show()

    spectrogramInputs = SpectrogramInputs(True)
    spectrogramInputs.correctPowerForMusic = True
    spectrogram = Spectrogram(spectrogramInputs, timeSignal)
    spectrogram.correctPowerForMusic() # Make sure correction isn't done twice
    # spectrogram.printSpectrogramParams()
    # spectrogram.plotSpectrogram()

    crossSectionPowerDb = np.array([])
    logFreqCross = np.array([])
    for segIdx in range(spectrogram.params.numSegments):
        crossSectionPowerDb = \
            np.append(crossSectionPowerDb,
                      spectrogram.powerDbComponentsList[segIdx][:, 7])
        logFreqCross = np.append(logFreqCross,
                                 np.log10(spectrogram.freqLabelsList[segIdx]))

    # plt.plot(logFreqCross, crossSectionPowerDb)
    # plt.show()

    assert(np.array_equal(np.around(crossSectionPowerDb), test5PowerRef))

    print("\nSpectrogram Test 5 Finish\n")


# Check the functions for applying a power mask to a non-composite spectrogram
def spectrogramTest6():
    print("\nSpectrogram Test 6 Start\n")

    testFreq = 500
    sampleRate = 2 ** 15. # 32768.
    maskValue = 0.5

    targetSpectrogramTimeSec = 0.5
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction2(timeVec, testFreq)

    spectrogramInputs = SpectrogramInputs()
    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    powerDbComponentsListCpy = []
    for segIdx in range(spectrogram.params.numSegments):
        powerDbComponentsListCpy.append( \
            np.copy(spectrogram.powerDbComponentsList[segIdx]))

    assert(len(powerDbComponentsListCpy) == 1)

    mask = np.full((spectrogram.params.numSpectrogramValues,), maskValue)
    spectrogram.applyPowerDbMask(mask)
    powerDbComponentsListMask = spectrogram.powerDbComponentsList

    for segIdx in range(spectrogram.params.numSegments):
        # print(powerDbComponentsListCpy[segIdx])
        # print(powerDbComponentsListMask[segIdx])
        assert(np.array_equal(powerDbComponentsListCpy[segIdx] * maskValue,
                              powerDbComponentsListMask[segIdx]))

    print("\nSpectrogram Test 6 Finish\n")


# Check the functions for applying a power mask to a composite spectrogram
def spectrogramTest7():
    print("\nSpectrogram Test 7 Start\n")

    testFreq = 500
    sampleRate = 2 ** 15. # 32768.
    maskValue = 0.5

    targetSpectrogramTimeSec = 0.5
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction2(timeVec, testFreq)

    spectrogramInputs = SpectrogramInputs(True)
    spectrogram = Spectrogram(spectrogramInputs, timeSignal)
    # spectrogram.plotSpectrogram()

    powerDbComponentsListCpy = []
    for segIdx in range(spectrogram.params.numSegments):
        powerDbComponentsListCpy.append( \
            np.copy(spectrogram.powerDbComponentsList[segIdx]))

    assert(len(powerDbComponentsListCpy) == 6)

    mask = np.full((spectrogram.params.numSpectrogramValues,), maskValue)
    # mask[:spectrogram.params.numSpectrogramValues / 2] = maskValue / 10
    spectrogram.applyPowerDbMask(mask)
    # spectrogram.plotSpectrogram()
    powerDbComponentsListMask = spectrogram.powerDbComponentsList

    for segIdx in range(spectrogram.params.numSegments):
        # print(powerDbComponentsListCpy[segIdx])
        # print(powerDbComponentsListMask[segIdx])
        assert(np.array_equal(powerDbComponentsListCpy[segIdx] * maskValue,
                              powerDbComponentsListMask[segIdx]))

    print("\nSpectrogram Test 7 Finish\n")


# Check unity scaling function
def spectrogramTest8():
    print("\nSpectrogram Test 8 Start\n")

    testFreq = 2 ** 7
    sampleRate = 2 ** 13. # 8192.
    maxAmp = np.iinfo(np.int16).max # For 16 bit signal
    ampRatios = [maxAmp]

    targetSpectrogramTimeSec = 1.0
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction6(timeVec, testFreq, ampRatios)

    # Test a full amplitude function for unity power
    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    numFreq = spectrogram.params.numSegmentFreq[0]
    freqSpacing = sampleRate / spectrogram.params.numWindowSamples
    testFreqIdx = int(testFreq / freqSpacing)
    powerDbVec = spectrogram.getPowerDbVecUnityScale().reshape(numFreq, -1)
    powerDbVec = np.around(powerDbVec, decimals=3)
    assert(np.all(powerDbVec[testFreqIdx, :] == 1.0))
    assert(np.min(powerDbVec) == 0)

    # Test half the unity power
    ampRatios = [((maxAmp / 2) ** (1. / 2)) * 2, # Amp for 1/2 power
                 ((maxAmp / 2) ** (1. / 10)) * 2] # Amp for 1/10 power
    timeSignal = testFunction6(timeVec, testFreq, ampRatios)

    spectrogram = Spectrogram(spectrogramInputs, timeSignal)
    numFreq = spectrogram.params.numSegmentFreq[0]
    freqSpacing = sampleRate / spectrogram.params.numWindowSamples
    testFreqIdx1 = int(testFreq / freqSpacing)
    testFreqIdx2 = int(testFreq * 2 / freqSpacing)
    powerDbVec = spectrogram.getPowerDbVecUnityScale().reshape(numFreq, -1)
    powerDbVec = np.around(powerDbVec, decimals=3)
    assert(np.all(powerDbVec[testFreqIdx1, :] == 0.5))
    assert(np.all(powerDbVec[testFreqIdx2, :] == 0.1))
    assert(np.min(powerDbVec) == 0)

    # Test the equal loudness correction effect
    # 1. Test when equal loundness is applied after spectrogram is created
    ampRatios = [2 ** 8] # 1/2 power for 2 ** 15 sample rate
    timeSignal = testFunction6(timeVec, testFreq, ampRatios)
    spectrogram = Spectrogram(spectrogramInputs, timeSignal)
    numFreq = spectrogram.params.numSegmentFreq[0]
    freqSpacing = sampleRate / spectrogram.params.numWindowSamples
    testFreqIdx = int(np.around(testFreq / freqSpacing))

    maxPower = 20 * np.log10(maxAmp / 2)
    maxPowerSpec = np.max(spectrogram.getPowerDbVec())
    powerCorrection = eql.getEqualLoudnessCorrection(\
        spectrogram.freqLabelsList[0])[testFreqIdx]
    maxPowerCorrection = np.max(eql.equalLoudnessCorrectionPerFreq)

    spectrogram.applyEqualLoudness()
    powerDbVec = spectrogram.getPowerDbVecUnityScale().reshape(numFreq, -1)
    testVal = (maxPowerSpec + powerCorrection) / (maxPower + maxPowerCorrection)
    powerDbVec = np.around(powerDbVec, decimals=3)
    testVal = np.around(testVal, decimals=3)
    assert(np.all(powerDbVec[testFreqIdx, :] == testVal))

    # 2. Test when spectrogram inputs set correctPowerForMusic
    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.correctPowerForMusic = True
    spectrogram = Spectrogram(spectrogramInputs, timeSignal)
    powerDbVec = spectrogram.getPowerDbVecUnityScale().reshape(numFreq, -1)
    powerDbVec = np.around(powerDbVec, decimals=3)
    assert(np.all(powerDbVec[testFreqIdx, :] == testVal))

    print("\nSpectrogram Test 8 Finish\n")


# Check log resampling function
def spectrogramTest9():
    print("\nSpectrogram Test 9 Start\n")

    testFreq = 500
    sampleRate = 2 ** 15
    logFreqFactorDs = 32

    targetSpectrogramTimeSec = 0.1
    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    timeSignal = testFunction3(timeVec, testFreq)

    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.logFreqFactorDs = logFreqFactorDs

    spectrogram = Spectrogram(spectrogramInputs, timeSignal)

    assert(spectrogram.params.numPositiveFreqDs == 16)
    assert(spectrogram.params.numSpectrogramValues == 80)

    assert(np.array_equal(spectrogram.resampleMap, resampleMapRef))

    upsampleFreq = spectrogram.getUpsampleFreqLog()
    # np.save('./specTest9_upsampleFreqRef.npy', upsampleFreq)
    upsampleFreqRef = np.load('./specTest9_upsampleFreqRef.npy')
    assert(np.array_equal(upsampleFreq, upsampleFreqRef))

    upsampleTimeSignal = spectrogram.getTimeRepresentation()
    # np.save('./specTest9_upsampleTimeRef.npy', upsampleTimeSignal)
    upsampleTimeSignalRef = np.load('./specTest9_upsampleTimeRef.npy')
    assert(np.array_equal(upsampleTimeSignal, upsampleTimeSignalRef))

    print("\nSpectrogram Test 9 Finish\n")


# Test for checking time and frequency resolution
def spectrogramTestRes():
    print("\nSpectrogram Test Res Start\n")

    testFreq = 500
    sampleRate = 2 ** 15.

    targetSpectrogramTimeSec = 1.
    timeRes = [5 / 1000., 10 / 1000., 20 / 1000., 50 / 1000., 100/1000.]
    freqRes = [1, 3, 10, 100]

    numInputSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromSamples(sampleRate,
                                                 numInputSamples)
    spectrogramInputs = SpectrogramInputs()

    for j in range(len(timeRes)):
        print("\nTime Res for: " + str(timeRes[j]))
        for i in range(6, 14):
            numWindowSamples = 2 ** i

            timeSignal = \
                testFunctionTimeRes(timeVec, testFreq, timeRes[j], sampleRate)
            spectrogramInputs.numWindowSamples = numWindowSamples
            spectrogram = Spectrogram(spectrogramInputs, timeSignal)

            print("Window Size  = " + str(i) + "\t" + str(numWindowSamples))
            # spectrogram.printSpectrogramParams()
            spectrogram.plotSpectrogram()

    for j in range(len(freqRes)):
        print("\nFreq Res for: " + str(freqRes[j]))
        for i in range(6, 14):
            numWindowSamples = 2 ** i

            timeSignal = testFunctionFreqRes(timeVec, testFreq, freqRes[j])
            spectrogramInputs.numWindowSamples = numWindowSamples
            spectrogram = Spectrogram(spectrogramInputs, timeSignal)

            print("Window Size  = " + str(i) + "\t" + str(numWindowSamples))
            # spectrogram.printSpectrogramParams()
            spectrogram.plotSpectrogram()


    print("\nSpectrogram Test Res Finish\n")


test1PowerRef = [-18.7292, -18.7292, -18.7292, -18.7292, -18.7292, -18.7292,
-18.7292, -18.7292, -18.7292, -18.7292, -18.7292, -18.7292, -19.7664, -27.5488,
-56.0833, -111.9609, -113.1973, -116.1786, -117.0741, -114.7271, -112.2721,
-112.3221, -114.8462, -117.1022, -116.0814, -113.106]

test5PowerRef = [  \
0,   0,   0,   4,  15,  15,  23,  17,  26,  23,  30,  31,  28,  26,  32,
33,  35,  35,  37,  34,  39,  35,  37,  41,  31,  40,  42,  32,  38,  45,
38,  24,  43,  45,  35,  15,  42,  47,  40,  19,  29,  46,  48,  36,  17,
4,  44,  49,  43,   1,   0,   3,  44,  50,  45,   4,   1,   8,  21,  47,
51,  43,  17,   6,  28,  48,  52,  43,  17,  45,  52,  48,  20,   4,  47,
53,  47,  14,  21,  37,  52,  53,  40,  21,   0,  27,  50,  54,  45,  21,
12,  13,  25,  50,  54,  46,  19,   7,   5,  14,  27,  50,  54,  45,  21,
11,   7,   9,  16,  29,  51,  54,  45,  22,  11,   5,   1,   2,   8,  19,
50,  55,  48,  16,   5,   0,   0,   1,  19,  46,  55,  51,  27,  14,  20,
36,  53,  54,  43,  24,  15,  12,  21,  51,  56,  48,  17,   7,   4,  10,
21,  48,  56,  52,  26,  14,   9,  10,  16,  25,  43,  56,  55,  39,  23,
14,   6,   0,   1,  11,  25,  53,  57,  50,  21,  10,   3,   0,   1,   5,
12,  20,  35,  54,  56,  46,  25,  14,   7,   2,   0,   0,   0,   0,   1,
13,  49,  56,  50,  14,   2,   0,   0,   0,   0,   0,   3,  10,  48,  55,
49,  13,   0,   1,  10,  20,  36,  51,  51,  38,  20,  11,   4,   2,   5,
15,  42,  49,  45,  18,   5,   0,   0,   0,   0,   0,   5,  41,  46,  39,
1,   0,   0,   0,   0,   0,   0,   0,  10,  35,  43,  39,  15,   1,   0,
0,   0,   0,   0,   0,   0,   0,   1,  37,  42,  35,   0,   0,   0,   9,
0,   0,   9,  31,  41,  38,  18,   4,   0,   0,   0,   0,  34,  41,  35,
0,   0,   0,   0,   0,   0,   3,  37,  42,  36,   0,   0,   0,   0,   0,
0,   0,  37,  44,  39,   8,   4,  14,  33,  45,  43,  27,  12,   3,   0,
0,   0,   0,   0,]

resampleMapRef = \
[[  0,  1],
 [  1,  2],
 [  2,  3],
 [  3,  4],
 [  4,  6],
 [  6,  9],
 [  9, 14],
 [ 14, 21],
 [ 21, 32],
 [ 32, 48],
 [ 48, 71],
 [ 71,106],
 [106,158],
 [158,234],
 [234,346],
 [346,513]]


def main():
    print("\n\n----------Spectrogram Test Starting----------\n\n")

    spectrogramTest1()
    spectrogramTest2()
    spectrogramTest3()
    spectrogramTest4()
    spectrogramTest5()
    spectrogramTest6()
    spectrogramTest7()
    spectrogramTest8()
    spectrogramTest9()
    # spectrogramTestRes()

    print("\n\n----------Spectrogram Test Finished----------\n\n")

if __name__ == "__main__": main()
