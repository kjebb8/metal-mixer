import numpy as np
import spectrogramUtils as specUtils

'''
File: spectrogramParams.py
Brief: Class data structure used by the Spectrogram Class to define the
       properties of the Spectrogram. Uses the SpectrogramInputs to determine
       all the other parameters needed to characterize and build a spectrogram.
'''
class SpectrogramParams:

    def __init__(self, spectrogramInputs, numInputSamples):

        self.sampleRate = spectrogramInputs.sampleRate
        self.numWindowSamples = spectrogramInputs.numWindowSamples
        self.segmentStartFreq = spectrogramInputs.segmentStartFreq
        self.overlap = spectrogramInputs.overlap
        self.logFreqFactorDs = spectrogramInputs.logFreqFactorDs
        self.numInputSamples = numInputSamples

        self.numSegments = None
        self.setupSegmentWindows()

        self.hopSize = None
        self.numFftWindows = None
        self.numSpectrogramSamples = None
        self.numUnusedSamples = None
        self.windowTimeSec = None
        self.timeResolution = None
        self.spectrogramTimeSec = None
        self.calculateSegmentTimeParams()

        self.numPositiveFreq = None
        self.maxFreq = None
        self.freqResolution = None
        self.segmentFreqIdx = None
        self.numSegmentFreq = None
        self.numPositiveFreqDs = None
        self.calculateSegmentFreqParams()

        self.numSpectrogramValues = None
        self.getNumSpectrogramValues()

        self.validateParams()


    def setupSegmentWindows(self):

        if isinstance(self.numWindowSamples, int):
            self.numWindowSamples = np.array([self.numWindowSamples])

        self.numSegments = len(self.numWindowSamples)

        # Sort windows for segments from largest to smallest
        if self.numSegments > 1:
            self.numWindowSamples[::-1].sort()


    def calculateSegmentTimeParams(self):
        self.hopSize = \
            np.floor(self.numWindowSamples * (1 - self.overlap)).astype(int)

        self.numFftWindows = np.zeros((self.numWindowSamples.shape), dtype=int)
        # Get the number of FFT windows for the first segment with the longest
        # time and most samples (numWindowSamples is sorted largest to smallest)
        self.numFftWindows[0] = \
            np.floor((self.numInputSamples - self.numWindowSamples[0]) / \
                      self.hopSize[0]) + 1

        # Get the maximum number of samples used in the spectrogram from the
        # segment with the largest window size (worst case)
        self.numSpectrogramSamples = self.hopSize[0] * \
                                     (self.numFftWindows[0] - 1) + \
                                     self.numWindowSamples[0]

        # Use the max number of samples to get the number of fft windows for
        # the remaining segments
        for segIdx in range(1, self.numSegments):
            newFftWindows = \
                (self.numSpectrogramSamples - self.numWindowSamples[segIdx]) / \
                self.hopSize[segIdx] + 1
            self.numFftWindows[segIdx] = newFftWindows

        self.numUnusedSamples = \
            self.numInputSamples - self.numSpectrogramSamples

        self.windowTimeSec = \
            specUtils.getTimefromSamples(self.sampleRate,
                                         self.numWindowSamples)

        self.timeResolution = \
            specUtils.getTimefromSamples(self.sampleRate,
                                         self.hopSize)

        self.spectrogramTimeSec = \
            specUtils.getTimefromSamples(self.sampleRate,
                                         self.numSpectrogramSamples)


    def calculateSegmentFreqParams(self):
        self.numPositiveFreq = \
            specUtils.getNumPositiveFreq(self.numWindowSamples)

        self.maxFreq = self.sampleRate / 2

        self.freqResolution = 1 / self.windowTimeSec

        # Determine the start (also end) frequencies for each segment
        if self.segmentStartFreq is None:
            if self.numSegments > 1:
                self.segmentStartFreq = \
                    np.append(0, self.freqResolution[1:] * 10)
            else:
                self.segmentStartFreq = np.array([0])
        self.segmentStartFreq = np.append(self.segmentStartFreq, self.maxFreq)

        # Determine the indices from the FFT to keep for each segment
        self.segmentFreqIdx = np.zeros((self.numSegments, 2), dtype=int)
        for segIdx in range(self.numSegments):

            startFreqIdx = np.ceil(self.segmentStartFreq[segIdx] / \
                                   self.freqResolution[segIdx])

            endFreqIdx = np.ceil(self.segmentStartFreq[segIdx + 1] / \
                                 self.freqResolution[segIdx])

            self.segmentFreqIdx[segIdx] = np.array([startFreqIdx, endFreqIdx])

        # Include the last data point in the range of the last segment
        self.segmentFreqIdx[-1][1] += 1


        # Determine the number of frequency values in each segment
        self.numSegmentFreq = np.zeros((self.numSegments,), dtype=int)
        for segIdx in range(self.numSegments):
            self.numSegmentFreq[segIdx] = self.segmentFreqIdx[segIdx][1] - \
                                          self.segmentFreqIdx[segIdx][0]

        if self.logFreqFactorDs is not None:
            self.numPositiveFreqDs = \
                int(self.numPositiveFreq[0] / float(self.logFreqFactorDs))


    def getNumSpectrogramValues(self):      
        if self.logFreqFactorDs is not None:
            self.numSpectrogramValues = \
                self.numFftWindows[0] * self.numPositiveFreqDs
        else:
            self.numSpectrogramValues = 0
            for segIdx in range(self.numSegments):
                self.numSpectrogramValues += \
                    self.numFftWindows[segIdx] * self.numSegmentFreq[segIdx]


    def validateParams(self):
        # Overlap is in valid range
        assert(self.overlap < 1 and self.overlap >= 0)

        # Largest window fits more than one
        assert(self.numFftWindows[0] >= 1)

        # All numWindowSamples are exact powers of 2. Best performance
        assert(np.all(np.mod(np.log2(self.numWindowSamples), 1) == 0))

        # numWindowSamples is a 1D vector, not a matrix
        assert(self.numWindowSamples.ndim == 1)

        # No segments start outside of the available frequency values
        assert(np.max(self.segmentStartFreq) <= self.maxFreq)

        # The logFreqFactorDs in valid range and only for non-composite spec
        if self.logFreqFactorDs is not None:
            assert(self.logFreqFactorDs >= 1)
            assert(self.numSegments == 1)


    def printParams(self):
        print("\nSpectrogram Params: \n")
        for par in vars(self).items():
            print("\t" + par[0] + ": " + str(par[1]))
        print("\n")


    @staticmethod
    def getNumSpectrogramSamples(numWindowSamples, numInputSamples, overlap):
        maxWindowSamples = None
        if isinstance(numWindowSamples, int):
            maxWindowSamples = numWindowSamples
        else:
            maxWindowSamples = np.max(numWindowSamples)
        hopSize = np.floor(maxWindowSamples * (1 - overlap)).astype(int)
        numFftWindows = \
            np.floor((numInputSamples - maxWindowSamples) / hopSize) + 1
        numSpectrogramSamples = hopSize * (numFftWindows - 1) + maxWindowSamples
        return int(numSpectrogramSamples)
