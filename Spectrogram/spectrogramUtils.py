import numpy as np

'''
File: spectrogramUtils.py
Brief: Utility functions for spectrogram related time/frequency functionality.
'''
def getSamplesfromTime(sampleRate, durationSec):
    return np.floor(sampleRate * durationSec).astype(int)


def getTimefromSamples(sampleRate, numSamples):
    return (numSamples / np.float(sampleRate))


def getTimeVectorFromTime(sampleRate, durationSec):
    numSamples = getSamplesfromTime(sampleRate, durationSec)
    timeVec = getTimeVectorFromSamples(sampleRate, numSamples)
    return timeVec


def getTimeVectorFromSamples(sampleRate, numSamples):
    timePerSample = 1. / sampleRate
    durationSec = getTimefromSamples(sampleRate, numSamples)
    endTimeSec = durationSec - timePerSample
    timeVec = np.linspace(0, endTimeSec, num=numSamples)
    return timeVec


def getNumPositiveFreq(numSamples):
    return (np.floor(numSamples / 2.) + 1).astype(int) # Includes zero


def getFirstNegativeFreqIdx(numSamples):
    return getNumPositiveFreq(numSamples)


def getNumNegativeFreq(numSamples):
    return numSamples - getNumPositiveFreq(numSamples)


def getFreqVectorFromTime(sampleRate, durationSec):
    numSamples = getSamplesfromTime(sampleRate, durationSec)
    return getFreqVectorFromSamples(sampleRate, numSamples)


def getFreqVectorFromSamples(sampleRate, numSamples):
    freqSpacing = np.float(sampleRate) / numSamples
    freqVec = np.linspace(0, sampleRate - freqSpacing, num=numSamples)
    firstNegativeFreqIdx = getFirstNegativeFreqIdx(numSamples)
    numNegativeFreq = getNumNegativeFreq(numSamples)
    freqVec[firstNegativeFreqIdx:] = \
        np.flipud(freqVec[1:numNegativeFreq + 1]) * -1
    return freqVec
