import numpy as np
import matplotlib.pyplot as plt
import time
import spectrogramUtils as specUtils
from frequencyAnalyzer import FreqAnalyzer

'''
File: frequencyAnalyzerTest.py
Brief: Test the FreqAnalyzer Class functions.
'''
def testFunction1(time, freq):
    return (np.cos(2 * np.pi * freq * time) + \
            np.sin(2 * np.pi * freq / 2 * time) / 2)


def testFunction2(time):
    testFunction = np.zeros((time.shape))
    for i in range(7):
        testFunction += i * np.cos(2 * np.pi * i * time)
    return testFunction


def testFunction3(time):
    testFunction = np.zeros((time.shape))
    for fq in range(7):
        testFunction += fq * np.cos(2 * np.pi * fq * time)
        testFunction += fq / 2 * np.sin(2 * np.pi * fq * time)
    return testFunction


def testFunction4(time):
    testFunction = np.zeros((time.shape))
    for fq in range(7):
        testFunction += fq * np.cos(2 * np.pi * fq * 1000 * time)
        testFunction *= fq / 2 * np.sin(2 * np.pi * fq * 1000 / 2 * time)
    return testFunction


def testFunction5(time):
    testFunction = np.zeros((time.shape))
    for fq in range(11):
        testFunction += fq * np.sin(2 * np.pi * fq * 1000 * time + 2 * fq)
        testFunction *= fq / 2 * np.cos(2 * np.pi * fq * 1000 / 2 * time - fq)
    return testFunction


def testFunction6(time):
    testFunction = np.zeros((time.shape))
    for fq in range(30):
        testFunction += fq * np.sin(2 * np.pi * fq * 1274 / 3.42 * time + fq)
        testFunction *= np.cos(2 * np.pi * fq * 1846 * time + 2.34 * fq)
    return testFunction


# Check the FFT correctly extracts the signal freqency and power
def fftTest1():
    print("\nFFT Test 1 Start")

    testFreq = 2
    # Even and odd # of samples
    testSampleRates = [10., 10.2]
    durationSec = 5.
    freqAnalyzer = FreqAnalyzer()

    for sampleRate in testSampleRates:
        numSamples = specUtils.getSamplesfromTime(sampleRate, durationSec)

        # Time and frequency vectors (independent variables)
        timeVec = specUtils.getTimeVectorFromTime(sampleRate, durationSec)
        freqVec = specUtils.getFreqVectorFromTime(sampleRate, durationSec)

        # Get the test function in time and frequency using FFT
        timeSignal = testFunction1(timeVec, testFreq)
        freqSignal = freqAnalyzer.fft(timeSignal)

        # Get the positive part of the signal
        positiveFreqVec = freqAnalyzer.getPositiveFreqPart(freqVec)

        # Verify the highest frequency of fft occurs at test frequency
        freqSignalPowerDb = freqAnalyzer.getFreqPowerDb(freqSignal)
        testFreqLocation = (np.where(positiveFreqVec == testFreq))[0]
        # Only works if one of the freq values matches the test frequency
        if testFreqLocation.size > 0:
            freqMaxValueLocation = np.argmax(freqSignalPowerDb)
            assert(testFreqLocation == freqMaxValueLocation)
        else:
            print("No frequency matches the test frequency. " +
                  "Check cannot be completed.")

        # Check the energy of the freq signal matches the time signal
        # Use Parseval theorem:
        timeEnergy = np.sum(timeSignal ** 2)
        # Reverse the dB scaling
        positiveFreqSignalNorm = \
            10 ** (freqSignalPowerDb / 20.) / 2 * numSamples
        freqEnergy = np.sum(positiveFreqSignalNorm ** 2) * 2 / numSamples
        # Error is introduced from the clipping of the low power signals
        assert((timeEnergy - freqEnergy) <= (0.01 * timeEnergy))

    print("FFT Test 1 Finish\n")


# Plot the frequency signal and the positive part only for visualization
# Check even and odd number of samples
def fftTest2():
    print("\nFFT Test 2 Start")

    # Even and odd # of samples
    testSampleRates = [10., 10.2]
    durationSec = 5.
    freqAnalyzer = FreqAnalyzer()

    for sampleRate in testSampleRates:
        numSamples = specUtils.getSamplesfromTime(sampleRate, durationSec)

        # Time and frequency vectors (independent variables)
        timeVec = specUtils.getTimeVectorFromTime(sampleRate, durationSec)
        freqVec = specUtils.getFreqVectorFromTime(sampleRate, durationSec)

        # Get the test function in time and frequency using FFT
        timeSignal = testFunction2(timeVec)
        freqSignal = freqAnalyzer.fft(timeSignal)

        # Get the positive part of the signal
        positiveFreqVec = freqAnalyzer.getPositiveFreqPart(freqVec)
        positiveFreqSignal = freqAnalyzer.getPositiveFreqPart(freqSignal)

        # Plot the original signal
        plt.plot(freqVec, np.real(freqSignal))
        plt.plot(freqVec, np.imag(freqSignal))
        plt.show()

        # Plot the positive signal
        plt.plot(positiveFreqVec, np.real(positiveFreqSignal))
        plt.plot(positiveFreqVec, np.imag(positiveFreqSignal))
        plt.show()

    print("FFT Test 2 Finish\n")


# Check the correctness of the FFT values, extracting the positive values,
# the norm function and frequency vector
def fftTest3():
    print("\nFFT Test 3 Start")

    sampleRate = 12.548
    # Even and odd # of samples
    testDurationSec = [4.241, 4.318]
    freqAnalyzer = FreqAnalyzer()
    testNumber = 0

    for durationSec in testDurationSec:
        numSamples = specUtils.getSamplesfromTime(sampleRate, durationSec)

        # Time and frequency vectors (independent variables)
        timeVec = specUtils.getTimeVectorFromTime(sampleRate, durationSec)
        freqVec = specUtils.getFreqVectorFromTime(sampleRate, durationSec)

        # Get the test function in time and frequency using FFT
        timeSignal = testFunction3(timeVec)
        freqSignal = freqAnalyzer.fft(timeSignal)
        positiveFreqVec = freqAnalyzer.getPositiveFreqPart(freqVec)
        positiveFreqSignal = freqAnalyzer.getPositiveFreqPart(freqSignal)
        positiveFreqSignalNorm = \
            freqAnalyzer.getFreqSignalNorm(positiveFreqSignal)

        # Check values
        assert(freqVec[1] == freqVec[-1] * -1)
        assert(positiveFreqVec[0] == 0)
        assert(round(positiveFreqVec[1], 2) == \
            round(sampleRate / numSamples, 2))

        oddCorrectFactor = (numSamples % 2 * float(sampleRate) / numSamples) / 2
        assert(round(positiveFreqVec[-1], 2) == \
            round((sampleRate / 2 - oddCorrectFactor), 2))

        assert(np.array_equal(positiveFreqSignalNorm.astype(int),
                              fftTest3FreqSignalRef[testNumber]))
        testNumber += 1

    print("FFT Test 3 Finish\n")


# Check the regeneration of full frequency spectrum from the positive only
# spectrum (complex and normailized)
def fftTest4():
    print("\nFFT Test 4 Start")

    sampleRate = 4410.54
    # Even and odd # of samples
    numSamplesTest = [15700, 15701]
    freqAnalyzer = FreqAnalyzer()

    for numSamples in numSamplesTest:
        # Time and frequency vectors (independent variables)
        timeVec = specUtils.getTimeVectorFromSamples(sampleRate, numSamples)
        freqVec = specUtils.getFreqVectorFromSamples(sampleRate, numSamples)

        # Get the test function in time and frequency using FFT
        timeSignal = testFunction4(timeVec)
        freqSignal = freqAnalyzer.fft(timeSignal)

        # Test the complex signal regeneration
        positiveFreqSignal = freqAnalyzer.getPositiveFreqPart(freqSignal)

        regeneratedFreqSignal = \
            freqAnalyzer.regenerateNegativeFreqSignal(positiveFreqSignal,
                                                      numSamples)

        assert(np.array_equal(np.around(regeneratedFreqSignal, decimals=2),
                              np.around(freqSignal, decimals=2)))

        # Test the normalized signal regeneration
        freqSignalNorm = freqAnalyzer.getFreqSignalNorm(freqSignal)
        positiveFreqSignalNorm = \
            freqAnalyzer.getPositiveFreqPart(freqSignalNorm)

        regeneratedFreqSignalNorm = \
            freqAnalyzer.regenerateNegativeFreqSignal(positiveFreqSignalNorm,
                                                      numSamples)

        assert(np.array_equal(np.around(regeneratedFreqSignalNorm, decimals=2),
                              np.around(freqSignalNorm, decimals=2)))

    print("FFT Test 4 Finish\n")


# Check the regeneration of the complex spectrum from norm and phase components
def fftTest5():
    print("\nFFT Test 5 Start")

    sampleRate = 4410.54
    # Even and odd # of samples
    # numSamplesTest = [11260, 11261]
    numSamplesTest = [11260, 11261]
    freqAnalyzer = FreqAnalyzer()

    for numSamples in numSamplesTest:
        # Time and frequency vectors (independent variables)
        timeVec = specUtils.getTimeVectorFromSamples(sampleRate, numSamples)
        freqVec = specUtils.getFreqVectorFromSamples(sampleRate, numSamples)

        # Get the test function in time and frequency using FFT
        timeSignal = testFunction5(timeVec)
        freqSignal = freqAnalyzer.fft(timeSignal)

        # Norm -> Positive -> Full -> Phase
        freqSignalPhase = freqAnalyzer.getFreqSignalPhase(freqSignal)
        freqSignalNorm = freqAnalyzer.getFreqSignalNorm(freqSignal)
        positiveFreqSignalNorm = \
            freqAnalyzer.getPositiveFreqPart(freqSignalNorm)
        regeneratedFreqSignalNorm = \
            freqAnalyzer.regenerateNegativeFreqSignal(positiveFreqSignalNorm,
                                                      numSamples)
        regeneratedFreqSignal = \
            freqAnalyzer.applyPhaseToNorm(regeneratedFreqSignalNorm,
                                          freqSignalPhase)

        assert(np.array_equal(np.around(regeneratedFreqSignal, decimals=2),
                              np.around(freqSignal, decimals=2)))

        # Positive -> Norm -> Phase -> Full
        positiveFreqSignal = freqAnalyzer.getPositiveFreqPart(freqSignal)
        positiveFreqSignalPhase = \
            freqAnalyzer.getFreqSignalPhase(positiveFreqSignal)
        positiveFreqSignalNorm = \
            freqAnalyzer.getFreqSignalNorm(positiveFreqSignal)
        regeneratedPositiveFreqSignal = \
            freqAnalyzer.applyPhaseToNorm(positiveFreqSignalNorm,
                                          positiveFreqSignalPhase)
        regeneratedFreqSignal = \
            freqAnalyzer.regenerateNegativeFreqSignal( \
                regeneratedPositiveFreqSignal,
                numSamples)

        assert(np.array_equal(np.around(regeneratedFreqSignal, decimals=2),
                              np.around(freqSignal, decimals=2)))

    print("FFT Test 5 Finish\n")


# Check the full regeneration of the time signal from the power spectrum
def fftTest6():
    print("\nFFT Test 6 Start")

    sampleRate = 44100.54
    # Even and odd # of samples
    numSamplesTest = [28926, 28927]
    freqAnalyzer = FreqAnalyzer()

    for numSamples in numSamplesTest:
        # Time and frequency vectors (independent variables)
        timeVec = specUtils.getTimeVectorFromSamples(sampleRate, numSamples)
        freqVec = specUtils.getFreqVectorFromSamples(sampleRate, numSamples)

        # Get the test function in time and frequency using FFT
        timeSignal = testFunction6(timeVec)
        freqSignal = freqAnalyzer.fft(timeSignal)

        # Get the Phase and Power spectrum
        freqSignalPhase = freqAnalyzer.getFreqSignalPhase(freqSignal)
        freqSignalPowerDb = freqAnalyzer.getFreqPowerDb(freqSignal)

        # Regenerate the original complex frequency spectrum
        regeneratedFreqSignal = \
            freqAnalyzer.getFreqSignalFromPowerDb(freqSignalPowerDb,
                                                  freqSignalPhase)

        # Error from clipping so set a threshold
        averageError = np.sum( \
            np.abs(regeneratedFreqSignal - freqSignal)) / numSamples
        averageValue = np.sum(np.abs(freqSignal)) / numSamples
        assert(averageError < 1e-3 * averageValue)

        # Convert to time domain with IFFT
        regeneratedTimeSignal = freqAnalyzer.ifft(regeneratedFreqSignal)

        # Error from clipping so set a threshold
        averageError = np.sum( \
                np.abs(regeneratedTimeSignal - timeSignal)) / numSamples
        averageValue = np.sum(np.abs(timeSignal)) / numSamples
        # print(averageError/averageValue)
        assert(averageError < 1e-10 * averageValue)

    print("FFT Test 6 Finish\n")


def fftTestNpVsKj():
    testFreq = 2
    sampleRate = 1000.
    durationSec = 5.14

    freqAnalyzer = FreqAnalyzer()
    timeVec = specUtils.getTimeVectorFromTime(sampleRate, durationSec)

    timeSignal = testFunction(timeVec, testFreq)

    print("Numpy FFT Start")
    tsNp1 = time.time()
    freqSignalNp = freqAnalyzer.fft(timeSignal)
    tsNp2 = time.time()
    print("Numpy FFT End")
    print("Numpy Time = " + str(tsNp2 - tsNp1))

    print("KJ FFT Start")
    tsKj1 = time.time()
    freqSignalKj = freqAnalyzer.fftKj(timeSignal)
    tsKj2 = time.time()
    print("KJ FFT End")
    print("KJ Time = " + str(tsKj2 - tsKj1))


fftTest3FreqSignalRef = \
    np.array([[13, 13, 14, 16, 37, 7, 13, 20, 50, 23, 0, 10, 32, 69, 16, 8,
               6, 127, 23, 21, 28, 152, 10, 18, 35, 142, 86],
              [15, 15, 16, 19, 36, 8, 13, 18, 36, 41, 13, 11, 11, 96, 25, 27,
               36, 129, 18, 21, 37, 99, 82, 14, 8, 34, 170, 14]])


def main():
    print("\n\n----------Frequency Analyzer Test Starting----------\n\n")

    fftTest1()
    fftTest2()
    fftTest3()
    fftTest4()
    fftTest5()
    fftTest6()
    # fftTestNpVsKj() # Numpy is 10x faster and doesn't crash the RAM...

    print("\n\n----------Frequency Analyzer Test Finished----------\n\n")
if __name__ == "__main__": main()
