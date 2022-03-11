import numpy as np
from spectrogramParams import SpectrogramParams
from spectrogram import Spectrogram

'''
File: audioSpectrograms.py
Brief: Convert audio .wav data to a list of spectrograms in chronological order.
       Also contains functions which apply spectrogram changes to all the
       spectrograms in the List.
'''
class AudioSpectrograms:

    def __init__(self, spectrogramInputs, targetSpectrogramSamples, audioData):

        self.spectrogramInputs = spectrogramInputs
        self.targetSpectrogramSamples = targetSpectrogramSamples

        self.numSpectrogramSamples = None
        self.numSpectrograms = None
        self.calculateNumSpectrograms(audioData)

        self.validateInputParams(audioData)

        self.spectrogramList = []
        self.generateSpectrogramList(audioData)
        assert(len(self.spectrogramList) == self.numSpectrograms)

        self.numSpectrogramValues = \
            self.spectrogramList[0].params.numSpectrogramValues


    def calculateNumSpectrograms(self, audioData):
        self.numSpectrogramSamples = \
            SpectrogramParams.getNumSpectrogramSamples( \
                self.spectrogramInputs.numWindowSamples,
                self.targetSpectrogramSamples,
                self.spectrogramInputs.overlap)

        self.numSpectrograms = \
            int(np.floor(len(audioData) / self.numSpectrogramSamples))


    def validateInputParams(self, audioData):
        assert(self.numSpectrograms > 0)
        # Make sure the input audio is Mono
        assert(audioData.ndim == 1)


    def generateSpectrogramList(self, audioData):

        startAudioIndex = 0
        for specIdx in range(self.numSpectrograms):

            endAudioIndex = startAudioIndex + self.numSpectrogramSamples
            audioSegment = audioData[startAudioIndex : endAudioIndex]

            newSpectrogram = Spectrogram(self.spectrogramInputs,
                                         audioSegment)

            self.spectrogramList.append(newSpectrogram)

            startAudioIndex = endAudioIndex


    def applyPowerDbMasks(self, powerDbMasks):

        assert(powerDbMasks.shape == \
               (self.numSpectrogramValues, self.numSpectrograms))

        for specIdx in range(self.numSpectrograms):
            powerDbMask = powerDbMasks[:, specIdx]
            self.spectrogramList[specIdx].applyPowerDbMask(powerDbMask)


    def getTimeRepresentation(self):
        timeSignal = np.array([])
        for specIdx in range(self.numSpectrograms):
            timeSpec = self.spectrogramList[specIdx].getTimeRepresentation()
            timeSignal = np.append(timeSignal, timeSpec)

        return timeSignal


    def getPowerDbMatrixUnityScale(self):
        powerDbMatrix = np.zeros((self.numSpectrogramValues,
                                  self.numSpectrograms))
        for specIdx in range(self.numSpectrograms):
            powerDbMatrix[:, specIdx] = \
                self.spectrogramList[specIdx].getPowerDbVecUnityScale()

        return powerDbMatrix

