import copy
import numpy as np
import matplotlib.pyplot as plt
import spectrogramUtils as specUtils
from audioSpectrograms import AudioSpectrograms
from frequencyAnalyzer import FreqAnalyzer
from spectrogramInputs import SpectrogramInputs

'''
File: audioSpectrogramsTest.py
Brief: Test the AudioSpectrograms Class functions.
'''
def testFunction1(time, freq):
    timeHalfLength = int(len(time) / 2)
    timeFirstHalf = time[:timeHalfLength]
    testFuncFirstHalf = np.cos(2 * np.pi * freq * timeFirstHalf) * \
                        np.sin(2 * np.pi * freq / 2 * timeFirstHalf) / 2

    timeSecondHalf = time[timeHalfLength:]
    testFuncSecondHalf = np.cos(2 * np.pi * freq / 2 * timeSecondHalf) * \
                         np.sin(2 * np.pi * freq / 4 * timeSecondHalf)
    timeFunc = np.append(testFuncFirstHalf, testFuncSecondHalf) * 10
    return timeFunc


def testFunction2(time, freq):
    timeHalfLength = int(len(time) / 2)
    timeFirstHalf = time[:timeHalfLength]
    timeSecondHalf = time[timeHalfLength:]
    testFuncFirstHalf = np.zeros((timeFirstHalf.shape))
    testFuncSecondHalf = np.zeros((timeSecondHalf.shape))
    twoPi = 2 * np.pi
    for fq in range(10):
        testFuncFirstHalf += np.sin(twoPi * fq * freq * 2 * timeFirstHalf) * \
            5 * (30 - fq) * np.cos(twoPi * fq * freq * 4 * timeFirstHalf)

        testFuncSecondHalf += np.cos(twoPi * fq * freq * timeSecondHalf) * \
            6 * fq * np.sin(twoPi * fq * freq / 3 * timeSecondHalf)
    return np.append(testFuncFirstHalf, testFuncSecondHalf)


def testFunction3(time, freq):
    timeHalfLength = int(len(time) / 2)
    timeFirstHalf = time[:timeHalfLength]
    testFuncFirstHalf = np.cos(2 * np.pi * freq * timeFirstHalf) * \
                        np.sin(2 * np.pi * freq / 2 * timeFirstHalf) / 2

    timeSecondHalf = time[timeHalfLength:]
    testFuncSecondHalf = np.cos(2 * np.pi * freq / 2 * timeSecondHalf) * \
                         np.sin(2 * np.pi * freq / 4 * timeSecondHalf)
    timeFunc = np.append(testFuncFirstHalf, testFuncSecondHalf) * 2 ** 12
    return timeFunc


# Test a non-composite example and verify numSpectrograms is correct
def audioSpectrogramsTest1():
    print("\nAudio Spectrogram Converter Test 1 Start\n")

    testFreq = 12000
    sampleRate = 44100.
    numWindowSamples = 2 ** 9  # 512
    targetTotalTime = 0.09
    targetSpectrogramTimeSec = targetTotalTime / 4

    targetSpectrogramSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromTime(sampleRate,
                                              targetTotalTime)
    timeSignal = testFunction1(timeVec, testFreq)

    # Visualize the time signal in Freq domain
    freqVec = specUtils.getFreqVectorFromTime(sampleRate,
                                              targetTotalTime)
    freqSignal = FreqAnalyzer.fft(timeSignal)
    positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)
    plt.plot(positiveFreqVec, freqSignalNorm)
    plt.show()

    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.numWindowSamples = numWindowSamples

    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          timeSignal)
    for spec in audioSpectrograms.spectrogramList:
        spec.plotSpectrogram()

    # Note the targetSpectrogramTimeSec is 1/4 of the targetTotalTime but since
    # there are many unused samples for this spectrogram size, we get an extra
    # spectrogram. This is correct behaviour. We want to use as much of the
    # time signal as possible
    assert(audioSpectrograms.numSpectrograms == 5)

    print("\nAudio Spectrogram Converter Test 1 Finish\n")


# Test composite spectrogram
def audioSpectrogramsTest2():
    print("\nAudio Spectrogram Converter Test 2 Start\n")

    testFreq = 500
    sampleRate = 2 ** 15
    targetTotalTime = 0.9
    targetSpectrogramTimeSec = targetTotalTime / 2

    targetSpectrogramSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromTime(sampleRate,
                                              targetTotalTime)
    timeSignal = testFunction2(timeVec, testFreq)

    # Visualize the time signal in Freq domain
    freqVec = specUtils.getFreqVectorFromTime(sampleRate,
                                              targetTotalTime)
    freqSignal = FreqAnalyzer.fft(timeSignal)
    positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)
    plt.plot(positiveFreqVec, freqSignalNorm)
    plt.show()

    spectrogramInputs = SpectrogramInputs(True)

    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          timeSignal)
    for spec in audioSpectrograms.spectrogramList:
        spec.plotSpectrogram()

    assert(audioSpectrograms.numSpectrograms == 2)
    assert(audioSpectrograms.spectrogramList[0].params.numSpectrogramSamples ==\
           audioSpectrograms.numSpectrogramSamples)
    assert(audioSpectrograms.spectrogramList[0].params.numSpectrogramSamples ==\
           audioSpectrograms.spectrogramList[1].params.numSpectrogramSamples)

    print("\nAudio Spectrogram Converter Test 2 Finish\n")


# Test applyPowerDbMasks
def audioSpectrogramsTest3():
    print("\nAudio Spectrogram Converter Test 3 Start\n")

    testFreq = 300
    sampleRate = 2 ** 15
    targetTotalTime = 4.5
    targetSpectrogramTimeSec = targetTotalTime / 5

    targetSpectrogramSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromTime(sampleRate,
                                              targetTotalTime)
    timeSignal = testFunction1(timeVec, testFreq)

    # Visualize the time signal in Freq domain
    # freqVec = specUtils.getFreqVectorFromTime(sampleRate,
    #                                           targetTotalTime)
    # freqSignal = FreqAnalyzer.fft(timeSignal)
    # positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
    # positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
    # freqSignalNorm = FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)
    # plt.plot(positiveFreqVec, freqSignalNorm)
    # plt.show()

    spectrogramInputs = SpectrogramInputs()

    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          timeSignal)
    audioSpectrogramsRef = copy.deepcopy(audioSpectrograms)
    # for spec in audioSpectrograms.spectrogramList:
    #     spec.plotSpectrogram()

    # Create a fake random mask
    numSpectrogramValues = \
            audioSpectrograms.spectrogramList[0].params.numSpectrogramValues
    numSpectrograms = audioSpectrograms.numSpectrograms
    numMaskValues = numSpectrogramValues * numSpectrograms

    mask = np.zeros((numMaskValues,))
    mask[:numMaskValues / 3] = 1
    np.random.shuffle(mask)
    mask = mask.reshape(numSpectrogramValues, numSpectrograms)

    # Test the applyPowerDbMasks function
    audioSpectrograms.applyPowerDbMasks(mask)

    for specIdx in range(numSpectrograms):

        zerosIdx = np.where(mask[:, specIdx] == 0)[0]
        onesIdx = np.where(mask[:, specIdx] == 1)[0]
        assert(len(zerosIdx) + len(onesIdx) == numSpectrogramValues)

        powerVec = audioSpectrograms.spectrogramList[specIdx].getPowerDbVec()
        powerVecRef = \
                audioSpectrogramsRef.spectrogramList[specIdx].getPowerDbVec()
        assert(np.all(powerVec[zerosIdx] == 0))
        assert(np.all(powerVec[onesIdx] == powerVecRef[onesIdx]))

    print("\nAudio Spectrogram Converter Test 3 Finish\n")


# Test getTimeRepresentation
def audioSpectrogramsTest4():
    print("\nAudio Spectrogram Converter Test 4 Start\n")

    testFreq = 300
    sampleRate = 2 ** 15
    targetTotalTime = 4.5
    targetSpectrogramTimeSec = targetTotalTime / 5

    targetSpectrogramSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromTime(sampleRate,
                                              targetTotalTime)
    timeSignal = testFunction1(timeVec, testFreq)

    spectrogramInputs = SpectrogramInputs()
    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          timeSignal)

    # Test the getTimeRepresentation function
    regeneratedTimeSignal = audioSpectrograms.getTimeRepresentation()
    numSamples = len(regeneratedTimeSignal)
    assert(float(numSamples) / len(timeSignal) > 0.98)
    timeSignalRef = timeSignal[:numSamples]

    averageError = \
        np.sum(np.abs(regeneratedTimeSignal - timeSignalRef)) / numSamples
    averageValue = np.sum(np.abs(timeSignalRef)) / numSamples
    # print(averageError)
    # print(averageValue)
    # print(averageError / averageValue)
    assert(averageError < 0.01 * averageValue)

    print("\nAudio Spectrogram Converter Test 4 Finish\n")


# Test getPowerDbMatrixUnityScale
def audioSpectrogramsTest5():
    print("\nAudio Spectrogram Converter Test 5 Start\n")

    testFreq = 300
    sampleRate = 2 ** 15
    targetTotalTime = 4.5
    targetSpectrogramTimeSec = targetTotalTime / 5
    refDataPath = "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/" + \
                  "Code/Spectrogram/ast5Ref.npy"

    targetSpectrogramSamples = \
        specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    timeVec = specUtils.getTimeVectorFromTime(sampleRate,
                                              targetTotalTime)
    timeSignal = testFunction3(timeVec, testFreq)

    spectrogramInputs = SpectrogramInputs()
    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          timeSignal)

    numSpectrogramValues = \
        audioSpectrograms.spectrogramList[0].params.numSpectrogramValues
    numSpectrograms = audioSpectrograms.numSpectrograms

    # Get the reference value
    powerDbMatrixRef = np.load(refDataPath)
    # powerDbMatrixRef = np.zeros((numSpectrogramValues, numSpectrograms))
    # for specIdx in range(audioSpectrograms.numSpectrograms):
    #     powerDbMatrixRef[:,specIdx] = \
    #         audioSpectrograms.spectrogramList[specIdx].getPowerDbVecUnityScale()
    # np.save(refDataPath, powerDbMatrixRef)

    # Test the getPowerDbMatrixUnityScale function
    powerDbMatrix = audioSpectrograms.getPowerDbMatrixUnityScale()
    assert(powerDbMatrix.shape == (numSpectrogramValues, numSpectrograms))
    assert(np.min(powerDbMatrix) >= 0)
    assert(np.max(powerDbMatrix) <= 1)
    assert(np.array_equal(powerDbMatrix, powerDbMatrixRef))

    print("\nAudio Spectrogram Converter Test 5 Finish\n")


def main():
    print("\n\n-----Audio Spectrogram Converter Test Starting-----\n\n")

    audioSpectrogramsTest1()
    audioSpectrogramsTest2()
    audioSpectrogramsTest3()
    audioSpectrogramsTest4()
    audioSpectrogramsTest5()

    print("\n\n-----Audio Spectrogram Converter Test Finished-----\n\n")

if __name__ == "__main__": main()
