import numpy as np
import spectrogramUtils as specUtils
import constants as const

'''
File: frequencyAnalyzer.py
Brief: Set of static methods used to do frequency/time related calculations.
       Essential functions for creating spectrograms for audio including
       short-time FFT/IFFT and converting frequency domain amplitude to power.
'''
class FreqAnalyzer:

    @staticmethod
    def fft(timeSignal):
        freqSignal = np.fft.fft(timeSignal)
        FreqAnalyzer.postProcessFft(freqSignal)
        return freqSignal


    @staticmethod
    def postProcessFft(freqSignal):
        # Correct superimposed positive and negative max frequencies
        numSamples = len(freqSignal)
        if numSamples % 2. == 0:
             maxFreqIdx = specUtils.getNumPositiveFreq(numSamples) - 1
             freqSignal[maxFreqIdx] /= 2


    @staticmethod
    def ifft(freqSignal):
        FreqAnalyzer.preProcessIfft(freqSignal)
        timeSignal = np.fft.ifft(freqSignal)
        # Make sure the imaginary part is insignificant (none higher than 1e-9)
        averageValue = np.average(np.abs(np.real(timeSignal)))
        # print(averageValue * 1e-9)
        # print(np.max(np.abs(np.imag(timeSignal))))
        assert(np.any(np.abs(np.imag(timeSignal)) >=  \
               1e-9 * averageValue) == 0)
        return np.real(timeSignal)


    @staticmethod
    def preProcessIfft(freqSignal):
        # Correct superimposed positive and negative max frequencies
        numSamples = len(freqSignal)
        if numSamples % 2. == 0:
             maxFreqIdx = specUtils.getNumPositiveFreq(numSamples) - 1
             freqSignal[maxFreqIdx] *= 2


    @staticmethod
    def fftKj(timeSignal):
        numSamples = len(timeSignal)
        indexArray = np.arange(numSamples)
        freqSignal = np.zeros((numSamples), dtype=np.complex_)
        for k in indexArray:
            fftVec = k * indexArray
            fftVec = np.exp(fftVec * -2 * np.pi * (0+1j) / numSamples)
            freqSignal[k] = np.dot(fftVec, timeSignal)
        assert(len(freqSignal) == (numSamples))
        FreqAnalyzer.postProcessFft(freqSignal)
        return freqSignal


    @staticmethod
    def getPositiveFreqPart(freqVector):
        numPositiveFreq = specUtils.getNumPositiveFreq(len(freqVector))
        positiveFreqVector = np.copy(freqVector[:numPositiveFreq])
        return positiveFreqVector


    @staticmethod
    def regenerateNegativeFreqSignal(positiveFreqSignal, numSamples):
        numNegativeFreq = specUtils.getNumNegativeFreq(numSamples)
        negativeFreqSignal = \
            np.flipud(positiveFreqSignal[1:numNegativeFreq + 1])
        if positiveFreqSignal.dtype == "cfloat":
            negativeFreqSignal = np.conj(negativeFreqSignal)
        freqSignal = np.append(positiveFreqSignal, negativeFreqSignal)
        return freqSignal


    @staticmethod
    def getFreqSignalPhase(freqSignal):
        freqSignalPhase = np.angle(freqSignal)
        return freqSignalPhase


    @staticmethod
    def applyPhaseToNorm(freqSignalNorm, freqSignalPhase):
        assert(freqSignalNorm.shape == freqSignalPhase.shape)
        freqSignal = \
            freqSignalNorm * \
            (np.cos(freqSignalPhase) + 1j * np.sin(freqSignalPhase))
        return freqSignal


    # Short Time FFT Algorithm
    @staticmethod
    def shortTimeFft(numFftWindows, numWindowSamples, hopSize, timeSignal):

        numPositiveFreq = specUtils.getNumPositiveFreq(numWindowSamples)
        phaseComponents = np.zeros((numWindowSamples, numFftWindows))
        powerDbComponents = np.zeros((numPositiveFreq, numFftWindows))

        hanningWindow = np.hanning(numWindowSamples)
        startSample = 0

        for windowIdx in range(numFftWindows):

            endSample = startSample + numWindowSamples

            windowTimeSignal = timeSignal[startSample:endSample] * hanningWindow

            windowFreqSignal = FreqAnalyzer.fft(windowTimeSignal)

            phaseComponents[:, windowIdx] = \
                FreqAnalyzer.getFreqSignalPhase(windowFreqSignal)

            powerDbComponents[:, windowIdx] = \
                FreqAnalyzer.getFreqPowerDb(windowFreqSignal)

            startSample += hopSize

        return (phaseComponents, powerDbComponents)


    # Short Time IFFT Algorithm
    @staticmethod
    def shortTimeIfft(numSpectrogramSamples, numFftWindows, numWindowSamples,
                     hopSize, phaseComponents, powerDbComponents):

        timeSignal = np.zeros((numSpectrogramSamples,))
        startSample = 0

        for windowIdx in range(numFftWindows):

            endSample = startSample + numWindowSamples

            windowPowerDb = powerDbComponents[:, windowIdx]

            windowPhase = phaseComponents[:, windowIdx]

            windowFreqSignal = \
                FreqAnalyzer.getFreqSignalFromPowerDb(windowPowerDb,
                                                      windowPhase)
            assert(len(windowFreqSignal) == numWindowSamples)

            windowTimeSignal = FreqAnalyzer.ifft(windowFreqSignal)
            timeSignal[startSample:endSample] += windowTimeSignal

            startSample += hopSize

        return timeSignal


    @staticmethod
    def correctTimeSignalEnds(numWindowSamples, overlap, timeSignal):
        # Only do the correction if overlap is 0.5. If not, there needs to be a
        # more intensive correction to the entire signal from the Hanning window
        if overlap != 0.5:
            print("Cannot correct hanning window if overlap is not 0.5")
            return

        # The first/last half windows need to be corrected by dividing by half
        # the Hanning window. The cutoff sets the smallest value to divide by.
        hanningWindow = np.hanning(numWindowSamples)
        hanningWindow = np.maximum(const.windowCorrectionCutoff, hanningWindow)
        halfWindowIdx = int(np.floor(numWindowSamples * (1 - overlap)))
        firstHalfHanning = hanningWindow[:halfWindowIdx]
        secondHalfHanning = hanningWindow[(-halfWindowIdx + 1):]

        # Apply the haf window correction
        timeSignal[:halfWindowIdx] /= firstHalfHanning
        timeSignal[(-halfWindowIdx + 1):] /= secondHalfHanning


    # Get the length of each complex number in the complex plane
    @staticmethod
    def getFreqSignalNorm(freqSignal):
        freqSignalNorm = np.abs(freqSignal)
        return freqSignalNorm


    @staticmethod
    def getBandpassFilteredTime(sampleRate, lowerFreq, upperFreq, timeSignal):
        numSamples = len(timeSignal)
        freqVec = specUtils.getFreqVectorFromSamples(sampleRate, numSamples)
        positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)

        freqSignal = FreqAnalyzer.fft(timeSignal)
        positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)

        lowerIdx = (np.abs(positiveFreqVec - lowerFreq)).argmin()
        upperIdx = (np.abs(positiveFreqVec - upperFreq)).argmin() - 1

        positiveFreqSignalFiltered = \
            np.zeros((positiveFreqSignal.shape), dtype=complex)
        positiveFreqSignalFiltered[lowerIdx:upperIdx] = \
            positiveFreqSignal[lowerIdx:upperIdx]

        freqSignalFiltered = \
            FreqAnalyzer.regenerateNegativeFreqSignal( \
                positiveFreqSignalFiltered,
                numSamples)

        timeSignalFiltered = FreqAnalyzer.ifft(freqSignalFiltered)
        return timeSignalFiltered


    # Get the power of each frequency in dB
    @staticmethod
    def getFreqPowerDb(freqSignal):
        numSamples = len(freqSignal)
        # 1. Get the positive frequencies of the signal
        positiveFreqSignal = FreqAnalyzer.getPositiveFreqPart(freqSignal)
        # 2. Calculate the magnitude of the spectrum
        positiveFreqSignalNorm = \
            FreqAnalyzer.getFreqSignalNorm(positiveFreqSignal)
        # 3. Scale the values to equal magnitude as the time domain amplitude
        positiveFreqSignalNorm *= 2. / numSamples
        # 4. Make sure there are no zero values before taking log
        positiveFreqSignalNorm[positiveFreqSignalNorm == 0] = \
            const.minMagnitudeValue
        # 5. Convert the power to dB
        freqSignalPowerDb = 20. * np.log10(positiveFreqSignalNorm)

        return freqSignalPowerDb


    # Get the original spectrum from the power/energy spectrum
    @staticmethod
    def getFreqSignalFromPowerDb(freqSignalPowerDb, freqSignalPhase):
        assert((len(freqSignalPowerDb) * 2 == len(freqSignalPhase) + 1) or \
               (len(freqSignalPowerDb) * 2 == len(freqSignalPhase) + 2))
        numSamples = len(freqSignalPhase)
        # 1. Convert the dB back to magnitude
        positiveFreqSignalNorm = 10. ** (freqSignalPowerDb / 20.)
        # 2. Rescale back to the FFT magnitude
        positiveFreqSignalNorm *= numSamples / 2.
        # 3. Add back the negative frequencies to the end of the signal
        freqSignalNorm = \
            FreqAnalyzer.regenerateNegativeFreqSignal( \
                positiveFreqSignalNorm,
                numSamples)
        # 4. Apply the phase component to each frequency
        freqSignal = \
            FreqAnalyzer.applyPhaseToNorm(freqSignalNorm, freqSignalPhase)

        return freqSignal
