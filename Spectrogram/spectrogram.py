import numpy as np
import matplotlib.pyplot as plt
import spectrogramUtils as specUtils
import constants as const
import equalLoudness as eql
from spectrogramParams import SpectrogramParams
from frequencyAnalyzer import FreqAnalyzer

'''
File: spectrogram.py
Brief: Class representing a spectrogram. Takes SpectrogramInputs and a time
       signal (i.e. audio) and creates a power spectrogram of the data. Can
       also apply a mask to the spectrogram and output the time domain data.
       Contains the functions for plotting the spectrogram for visualization.
'''
class Spectrogram:

    def __init__(self, spectrogramInputs, timeSignal):

        self.params = SpectrogramParams(spectrogramInputs, len(timeSignal))
        self.timeLabels = None
        self.equalLoudnessApplied = False
        self.freqLabelsList = []
        self.phaseComponentsList = []
        self.powerDbComponentsList = []
        self.initializeComponentLists()
        self.generateSpectrogram(spectrogramInputs, timeSignal)

        self.resampleMap = None
        if self.params.logFreqFactorDs is not None:
            self.downsampleFreqLog()


    def initializeComponentLists(self):
        for segIdx in range(self.params.numSegments):
            segPhaseComponents = np.zeros((self.params.numWindowSamples[segIdx],
                                           self.params.numFftWindows[segIdx]))
            segPowerDbComponents = np.zeros((self.params.numSegmentFreq[segIdx],
                                             self.params.numFftWindows[segIdx]))

            self.phaseComponentsList.append(segPhaseComponents)
            self.powerDbComponentsList.append(segPowerDbComponents)


    def generateSpectrogram(self, spectrogramInputs, timeSignal):
        self.generateTimeLabels()
        self.generateFreqLabelsList()
        self.generateSpectrogramFreqData(timeSignal)

        if spectrogramInputs.correctPowerForMusic:
            self.correctPowerForMusic()
        else:
            if spectrogramInputs.correctEqualLoudness:
                self.applyEqualLoudness()
            if spectrogramInputs.correctLowerPowerLimit:
                self.applyLowerPowerLimit()


    def generateTimeLabels(self):
        halfWindowTime = np.min(self.params.windowTimeSec) / 2
        startTime = halfWindowTime
        endTime = self.params.spectrogramTimeSec - halfWindowTime
        self.timeLabels = np.linspace(startTime,
                                      endTime,
                                      num=np.max(self.params.numFftWindows))


    def generateFreqLabelsList(self):
        for segIdx in range(self.params.numSegments):
            freqVec = \
                specUtils.getFreqVectorFromSamples( \
                    self.params.sampleRate,
                    self.params.numWindowSamples[segIdx])
            positiveFreqVec = FreqAnalyzer.getPositiveFreqPart(freqVec)
            startFreqIdx = self.params.segmentFreqIdx[segIdx][0]
            endFreqIdx = self.params.segmentFreqIdx[segIdx][1]
            self.freqLabelsList.append(positiveFreqVec[startFreqIdx:endFreqIdx])


    def generateSpectrogramFreqData(self, timeSignal):
        for segIdx in range(self.params.numSegments):

            phaseComponents, powerDbComponents = \
                FreqAnalyzer.shortTimeFft(self.params.numFftWindows[segIdx],
                                          self.params.numWindowSamples[segIdx],
                                          self.params.hopSize[segIdx],
                                          timeSignal)

            startFreqIdx = self.params.segmentFreqIdx[segIdx][0]
            endFreqIdx = self.params.segmentFreqIdx[segIdx][1]

            self.phaseComponentsList[segIdx] = phaseComponents
            self.powerDbComponentsList[segIdx] = \
                powerDbComponents[startFreqIdx:endFreqIdx, :]


    def getTimeRepresentation(self):
        timeSignal = np.zeros((self.params.numSpectrogramSamples,))
        for segIdx in range(self.params.numSegments):

            powerDbComponents = np.zeros((self.params.numPositiveFreq[segIdx],
                                         self.params.numFftWindows[segIdx]))

            startFreqIdx = self.params.segmentFreqIdx[segIdx][0]
            endFreqIdx = self.params.segmentFreqIdx[segIdx][1]

            if self.resampleMap is not None:
                powerDb = self.getUpsampleFreqLog()
            else:
                powerDb = self.powerDbComponentsList[segIdx]

            powerDbComponents[startFreqIdx:endFreqIdx, :] = powerDb

            timeSignalSeg = \
                FreqAnalyzer.shortTimeIfft(self.params.numSpectrogramSamples,
                                           self.params.numFftWindows[segIdx],
                                           self.params.numWindowSamples[segIdx],
                                           self.params.hopSize[segIdx],
                                           self.phaseComponentsList[segIdx],
                                           powerDbComponents)

            FreqAnalyzer.correctTimeSignalEnds( \
                self.params.numWindowSamples[segIdx],
                self.params.overlap,
                timeSignalSeg)

            # Refilter the time domain signal to eliminate frequencies outside
            # the segment.
            if self.params.numSegments > 1:
                lowerFreq = self.params.segmentStartFreq[segIdx]
                upperFreq = self.params.segmentStartFreq[segIdx+1]

                timeSignalSegFiltered = \
                    FreqAnalyzer.getBandpassFilteredTime(self.params.sampleRate,
                                                         lowerFreq,
                                                         upperFreq,
                                                         timeSignalSeg)
                timeSignal += timeSignalSegFiltered
                # timeSignal += timeSignalSeg
            else:
                timeSignal += timeSignalSeg

        return timeSignal


    def correctPowerForMusic(self):
        self.applyEqualLoudness()
        self.applyLowerPowerLimit()


    def applyEqualLoudness(self):
        if not self.equalLoudnessApplied:
            for segIdx in range(self.params.numSegments):
                # Equal Loudness Correction
                segCorrection = \
                    eql.getEqualLoudnessCorrection(self.freqLabelsList[segIdx])
                self.powerDbComponentsList[segIdx] += segCorrection
            self.equalLoudnessApplied = True


    def applyLowerPowerLimit(self):
        for segIdx in range(self.params.numSegments):
            # Clip lower power limit (i.e. to 0 dB)
            self.powerDbComponentsList[segIdx] = \
                np.maximum(const.minPowerDbValue,
                           self.powerDbComponentsList[segIdx])


    def getPowerDbVec(self):
        powerDbVec = np.array([])
        for segIdx in range(self.params.numSegments):
            powerDbSeg = self.powerDbComponentsList[segIdx]
            powerDbVec = np.append(powerDbVec, powerDbSeg)

        assert(len(powerDbVec) == self.params.numSpectrogramValues)

        return powerDbVec


    def getPowerDbVecUnityScale(self):
        powerDbVec = self.getPowerDbVec()
        powerDbVec = np.maximum(0, powerDbVec)
        powerDbVec /= self.getMaxPower()

        assert(np.max(powerDbVec) <= 1)
        assert(np.min(powerDbVec) >= 0)

        return powerDbVec


    def setPowerDbFromVec(self, powerDbVec):
        startSegIdx = 0
        for segIdx in range(self.params.numSegments):
            numFftWindows = self.params.numFftWindows[segIdx]

            if self.resampleMap is not None:
                numSegmentFreq = self.params.numPositiveFreqDs
            else:
                numSegmentFreq = self.params.numSegmentFreq[segIdx]

            numSegValues = numFftWindows * numSegmentFreq
            endSegIdx = startSegIdx + numSegValues

            powerDbSeg = powerDbVec[startSegIdx:endSegIdx]
            self.powerDbComponentsList[segIdx] = \
                powerDbSeg.reshape(numSegmentFreq, numFftWindows)

            startSegIdx = endSegIdx


    def applyPowerDbMask(self, powerDbMask):
        powerDbVec = self.getPowerDbVec()
        assert(len(powerDbVec) == len(powerDbMask))
        powerDbVec *= powerDbMask
        self.setPowerDbFromVec(powerDbVec)


    def getMaxPower(self):
        maxFreqAmp = np.iinfo(np.int16).max # For 16 bit audio wav file
        # Multiply by ~0.5 because of the hanning window
        maxFreqAmp *= np.mean(np.hanning(self.params.numWindowSamples[0]))
        maxPower = 20 * np.log10(maxFreqAmp)
        if self.equalLoudnessApplied:
            maxPower += np.max(eql.equalLoudnessCorrectionPerFreq)

        return maxPower


    def downsampleFreqLog(self):
        self.createResampleMap()

        numFreqResample = self.resampleMap.shape[0]
        resamplePowerDb = \
            np.zeros((numFreqResample,self.params.numFftWindows[0]))
        powerDb = self.powerDbComponentsList[0]
        for idx in range(numFreqResample):
            mapIdx = self.resampleMap[idx]
            powerDbToAverage = powerDb[mapIdx[0] : mapIdx[1], :]
            averagePower = np.mean(powerDbToAverage, axis=0)
            resamplePowerDb[idx] = averagePower

        self.powerDbComponentsList[0] = resamplePowerDb


    def createResampleMap(self):
        # Calculate the log base to use to fit the full frequency spectrum in
        # the requested number/fraction of values
        numFreq = self.params.numPositiveFreq[0]
        targetNumFreq = self.params.numPositiveFreqDs
        logBase = np.exp(np.log(numFreq) / targetNumFreq)

        # The initial section of the log sampling will be smaller than the
        # possible indices of 0, 1, 2, 3, etc. so repace the values with the
        # linear alternative until the log indices overtake the linear
        linearIdx = np.arange(targetNumFreq)
        logIdx = (logBase ** np.arange(targetNumFreq) - 1).astype(np.int64)
        useLinearIdx = linearIdx >= logIdx
        logIdx[useLinearIdx] = linearIdx[useLinearIdx]

        # Create a resample map so the operation can be reversed later when
        # going back to time domain. Store which indices from the original
        # map to the new log indices.
        self.resampleMap = np.zeros((targetNumFreq, 2))
        startIdx = 0
        endIdx = 0
        for idx in range(targetNumFreq - 1):
            endIdx = startIdx + (logIdx[idx + 1] - logIdx[idx])
            self.resampleMap[idx] = np.array([startIdx, endIdx])
            startIdx = endIdx
        self.resampleMap[-1] = np.array([startIdx, numFreq])

        self.resampleMap = self.resampleMap.astype(np.int64)


    def getUpsampleFreqLog(self):
        assert(self.resampleMap is not None)
        numFreqResample = self.resampleMap.shape[0]
        resamplePowerDb = np.zeros((self.params.numPositiveFreq[0],
                                    self.params.numFftWindows[0]))
        powerDb = self.powerDbComponentsList[0]
        for idx in range(numFreqResample):
            mapIdx = self.resampleMap[idx]
            averagePower = powerDb[idx]
            resamplePowerDb[mapIdx[0] : mapIdx[1], :] = averagePower

        return resamplePowerDb


    def plotSpectrogram(self):
        if self.params.numSegments > 1:
            self.plotCompositeSpectrogram()
        else:
            self.plotNonCompositeSpectrogram()

    def plotNonCompositeSpectrogram(self):
        if self.resampleMap is not None:
            powerDb = self.getUpsampleFreqLog()
        else:
            powerDb = self.powerDbComponentsList[0]
        plt.imshow(powerDb,
                    origin='lower',
                    cmap='jet',
                    interpolation='nearest',
                    aspect='auto',
                    extent=[self.timeLabels[0],
                            self.timeLabels[-1],
                            self.freqLabelsList[0][0],
                            self.freqLabelsList[-1][-1]])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()


    def plotCompositeSpectrogram(self):
        # 1. Expand the segments in time
        maxFftWindows = np.max(self.params.numFftWindows)
        specPlotTimeExpanded = np.zeros((np.sum(self.params.numSegmentFreq),\
                                         maxFftWindows))

        repeatsPerWindow = \
            np.floor(maxFftWindows / self.params.numFftWindows).astype(int)

        extraRepeats = maxFftWindows - \
                       repeatsPerWindow * self.params.numFftWindows

        freqStart = 0
        for segIdx in range(self.params.numSegments - 1):
            freqEnd = freqStart + self.params.numSegmentFreq[segIdx]

            for window in range(self.params.numFftWindows[segIdx]):
                windowToRepeat = self.powerDbComponentsList[segIdx][:, window]
                numRepeats = repeatsPerWindow[segIdx]
                if window == self.params.numFftWindows[segIdx] - 1:
                    numRepeats += extraRepeats[segIdx]

                for iter in range(numRepeats):
                    col = (window * repeatsPerWindow[segIdx]) + iter
                    specPlotTimeExpanded[freqStart:freqEnd, col] = \
                        windowToRepeat

            freqStart = freqEnd

        # Add the last segment that matches the dimension of the plot matrix
        specPlotTimeExpanded[freqStart:, :] = self.powerDbComponentsList[-1]


        # 2. Expand the segments in frequency
        maxNumFreq = np.max(self.params.numPositiveFreq)
        specPlotFullLinear = np.zeros((maxNumFreq, \
                                       np.max(self.params.numFftWindows)))

        repeatsPerFreq = np.around(self.params.freqResolution / \
                                   self.params.freqResolution[0]).astype(int)

        # Fill the first segment that matches the dimension of the plot matrix
        specPlotFullLinear[:self.params.numSegmentFreq[0], :] = \
            specPlotTimeExpanded[:self.params.numSegmentFreq[0], :]

        spectrogramRow = self.params.numSegmentFreq[0]
        rowToRepeat = self.params.numSegmentFreq[0]

        for segIdx in range(1, self.params.numSegments):

            for freqRow in range(self.params.numSegmentFreq[segIdx]):
                freqToRepeat = specPlotTimeExpanded[rowToRepeat, :]
                numRepeats = repeatsPerFreq[segIdx]
                if segIdx == self.params.numSegments - 1 and \
                   freqRow == self.params.numSegmentFreq[segIdx] - 1:
                   numRepeats = 1

                for iter in range(numRepeats):
                    specPlotFullLinear[spectrogramRow , :] = freqToRepeat
                    spectrogramRow += 1

                rowToRepeat += 1


        plt.imshow(specPlotTimeExpanded,
                   origin='lower',
                   cmap='jet',
                   interpolation='nearest',
                   aspect='auto',
                   extent=[self.timeLabels[0],
                           self.timeLabels[-1],
                           self.freqLabelsList[0][0],
                           self.freqLabelsList[-1][-1]])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        plt.imshow(specPlotFullLinear,
                   origin='lower',
                   cmap='jet',
                   interpolation='nearest',
                   aspect='auto',
                   extent=[self.timeLabels[0],
                           self.timeLabels[-1],
                           self.freqLabelsList[0][0],
                           self.freqLabelsList[-1][-1]])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()


    def printSpectrogramParams(self):
        """
        - Human freq resolution varies depending on the frequency but it would
          be between 3 Hz and 100 Hz (maybe?).
        - Human time resolution is roughly 1 ms
        - Human lower frequency limit is 20 Hz
        - Human upper frequency limit is 20 kHz
        - Piano is 27.5 Hz (A0) to 4186 Hz (C8)
        """
        self.params.printParams()

        print("\nSpectrogram Data Sizes: \n")
        print("\ttimeLabels Size: " + str(self.timeLabels.shape))

        for segIdx in range(self.params.numSegments):
            print("\nSegment: " + str(segIdx))
            print("\tfreqLabelsList Size: " +
                str(self.freqLabelsList[segIdx].shape))
            print("\tphaseComponents Size: " +
                str(self.phaseComponentsList[segIdx].shape))
            print("\tpowerDbComponents Size: " + \
                str(self.powerDbComponentsList[segIdx].shape))
            print("\n")

        print("\n")
