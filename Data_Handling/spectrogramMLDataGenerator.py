import numpy as np
import constants as const
from audioHandler import AudioHandler
from ..Spectrogram.spectrogram import Spectrogram

'''
File: spectrogramMLDataGenerator.py
Brief: Generate the inputs for Machine Learning. This includes spectrograms
       (as column vectors of power) to input to the model and ideal masks to as
       the ground truth values. This data is for training the model only.

       Input: mixed instrument list from the InstrumentMixer.
       Outputs: are numpy matrices with the examples in each column.

       The input spectrogram values are scaled to between 0 and 1 for ML.

       Ideal masks can be binary or decimal values.
'''
class SpectrogramMLDataGenerator:

    @staticmethod
    def getSpectrogramMLData(mixedInstrumentsList,
                             isoInstrumentList,
                             spectrogramInputs,
                             numSpectrogramValues,
                             numSpectrogramValuesMask = None,
                             spectrogramInputsMask = None,
                             useBinaryMasks = True):

        inputSpectrogramVecs = \
            SpectrogramMLDataGenerator.getInputSpectrogramVecs(\
                mixedInstrumentsList,
                spectrogramInputs,
                numSpectrogramValues)

        if spectrogramInputsMask == None:
            spectrogramInputsMask = spectrogramInputs
            numSpectrogramValuesMask = numSpectrogramValues

        idealMasks = None

        if useBinaryMasks:
            idealMasks = SpectrogramMLDataGenerator.getIdealBinaryMasks(\
                mixedInstrumentsList,
                isoInstrumentList,
                spectrogramInputsMask,
                numSpectrogramValuesMask)
        else:
            idealMasks = SpectrogramMLDataGenerator.getIdealMasks(\
                mixedInstrumentsList,
                isoInstrumentList,
                spectrogramInputsMask,
                numSpectrogramValuesMask)

        return (inputSpectrogramVecs, idealMasks)


    @staticmethod
    def getInputSpectrogramVecs(mixedInstrumentsList,
                                spectrogramInputs,
                                numSpectrogramValues):

        numSpectrograms = len(mixedInstrumentsList)

        inputSpectrogramVecs = np.zeros((numSpectrogramValues, numSpectrograms))

        for specIdx in range(numSpectrograms):

            # Get the mixed spectrogram
            mixedInstrumentsDict = mixedInstrumentsList[specIdx]

            mixedSpectrogramVec = \
                SpectrogramMLDataGenerator.getMixedSpectrogramVec( \
                    mixedInstrumentsDict,
                    spectrogramInputs,
                    True)
            inputSpectrogramVecs[:, specIdx] = mixedSpectrogramVec

        return inputSpectrogramVecs


    @staticmethod
    def getIdealMasks(mixedInstrumentsList,
                      isoInstrumentList,
                      spectrogramInputsMask,
                      numSpectrogramValuesMask):

        numSpectrograms = len(mixedInstrumentsList)
        numIsoInstruments = len(isoInstrumentList)
        numIdealMaskValues = numSpectrogramValuesMask * numIsoInstruments

        idealMasks = np.zeros((numIdealMaskValues, numSpectrograms))

        for specIdx in range(numSpectrograms):
            mixedInstrumentsDict = mixedInstrumentsList[specIdx]
            idealMask = np.zeros((numSpectrogramValuesMask, numIsoInstruments))

            # 1. Get the mixed spectrogram vector
            mixedSpectrogramVec = \
                SpectrogramMLDataGenerator.getMixedSpectrogramVec( \
                    mixedInstrumentsDict,
                    spectrogramInputsMask)

            for instIdx in range(numIsoInstruments):
                isoInstrument = isoInstrumentList[instIdx]

                # 2. If the isolate instrument is in the dictionary, get
                #      the spectrogram for the instrument
                isoInstrumentSpectrogramVec = \
                    np.zeros((numSpectrogramValuesMask,))

                if isoInstrument in mixedInstrumentsDict:
                    isoInstrumentSpectrogramVec = \
                        SpectrogramMLDataGenerator.getSpectrogramVec(\
                            spectrogramInputsMask,
                            mixedInstrumentsDict[isoInstrument])

                # 3. Get the ideal mask by comparing the spectrogram of the
                #      instrument to isolate with the mixed spectrogram
                idealMaskInstrument = \
                    SpectrogramMLDataGenerator.getIdealMaskInstrument(\
                        mixedSpectrogramVec,
                        isoInstrumentSpectrogramVec)

                idealMask[:, instIdx] = idealMaskInstrument

            idealMasks[:, specIdx] = idealMask.reshape(1, -1, order='F')

        return idealMasks


    @staticmethod
    def getIdealBinaryMasks(mixedInstrumentsList,
                            isoInstrumentList,
                            spectrogramInputsMask,
                            numSpectrogramValuesMask):

        numSpectrograms = len(mixedInstrumentsList)
        numIsoInstruments = len(isoInstrumentList)
        numIdealMaskValues = numSpectrogramValuesMask * numIsoInstruments

        idealMasks = np.zeros((numIdealMaskValues, numSpectrograms))
        for specIdx in range(numSpectrograms):
            mixedInstrumentsDict = mixedInstrumentsList[specIdx]
            idealMask = np.zeros((numSpectrogramValuesMask, numIsoInstruments))

            # 1. Get the spectrogram vector for each instrument
            numInst = len(mixedInstrumentsDict)
            instSpectrogramVecs = np.zeros((numSpectrogramValuesMask, numInst))
            instColumnsLabels = {}
            instIdx = 0
            for inst in mixedInstrumentsDict:
                instSpectrogramVecs[:, instIdx] = \
                    SpectrogramMLDataGenerator.getSpectrogramVec(\
                        spectrogramInputsMask,
                        mixedInstrumentsDict[inst])
                instColumnsLabels[inst] = instIdx
                instIdx += 1

            # 2. Set a threshold for the minimum power that can be considered
            #    for a 1 in the binary mask

            # TODO: Should this be the same as the minPowerDbValue from the
            #       spectrogram constants?
            instSpectrogramVecs[instSpectrogramVecs < const.minPowerBinMask] = 0

            # 3. Find the frequencies where the iso instrument has the greatest
            #    power compared to the other instruments in the sample
            for instIdx in range(numIsoInstruments):
                isoInstrument = isoInstrumentList[instIdx]
                idealMaskInstrument = np.zeros((numSpectrogramValuesMask,))
                if isoInstrument in mixedInstrumentsDict:
                    col = instColumnsLabels[isoInstrument]
                    vecs = np.copy(instSpectrogramVecs)
                    isoSpectrogramVec = np.copy(vecs[:, col])
                    isoSpectrogramVec = isoSpectrogramVec.reshape(-1, 1)
                    vecs[(isoSpectrogramVec >= instSpectrogramVecs) & \
                         (isoSpectrogramVec != 0)] = 1
                    vecs[isoSpectrogramVec < instSpectrogramVecs] = 0
                    idealMaskInstrument = np.min(vecs, axis=1)

                idealMask[:, instIdx] = idealMaskInstrument


            idealMasks[:, specIdx] = idealMask.reshape(1, -1, order='F')

        return idealMasks

    @staticmethod
    def getMixedSpectrogramVec(mixedInstrumentsDict,
                               spectrogramInputs,
                               useUnityScale = False):
        (mixedTimeData, _) = \
            AudioHandler.mixAudio(mixedInstrumentsDict.values())

        mixedSpectrogramVec = \
            SpectrogramMLDataGenerator.getSpectrogramVec(\
                spectrogramInputs,
                mixedTimeData,
                useUnityScale)

        return mixedSpectrogramVec


    @staticmethod
    def getSpectrogramVec(spectrogramInputs, timeData, useUnityScale = False):
        spectrogram = Spectrogram(spectrogramInputs, timeData)
        spectrogramVec = None
        if useUnityScale:
            spectrogramVec = spectrogram.getPowerDbVecUnityScale()
        else:
            spectrogramVec = spectrogram.getPowerDbVec()
        return spectrogramVec


    @staticmethod
    def getIdealMaskInstrument(mixedSpectrogramVec,
                               isoInstrumentSpectrogramVec):
        mixedSpectrogramVecCpy = np.copy(mixedSpectrogramVec)
        mixedSpectrogramVecCpy[mixedSpectrogramVecCpy == 0.0] = 1e-6
        idealMask = isoInstrumentSpectrogramVec / mixedSpectrogramVecCpy
        return idealMask
