import os
import numpy as np
from enum import Enum
from audioHandler import AudioHandler

'''
File: instrumentDataParser.py
Brief: Load the .wav files from a folder and create a dictionary with each
       instrument's data (time domain) divided into the number of samples to
       be used for Spectrograms.
'''
class InstrumentDataParser:

    class SupportedInstruments(Enum):
        acousticGuitar = "acousticGuitar"
        electricGuitar = "electricGuitar"
        drums = "drums"
        vocal = "vocal"
        bass = "bass"
        piano = "piano"


    @staticmethod
    def getInstrumentDataDict(folderPath,
                              sampleRate,
                              numSpectrogramSamples,
                              instrumentList = None):
        print("Read Instrument Files from " + folderPath)
        print("Divide data into " + str(int(numSpectrogramSamples)) + " samples")

        supportedInstrumentsList = \
            InstrumentDataParser.SupportedInstruments.__members__
        if instrumentList == None:
            instrumentList = supportedInstrumentsList
        else:
            for inst in instrumentList:
                assert(inst in supportedInstrumentsList)

        instrumentDataDict = {}

        numValidWavFiles = 0
        numInvalidWavFiles = 0

        for filename in os.listdir(folderPath):

            if filename.endswith(".wav"):
                instrument = filename.split('_')[0]

                if instrument in instrumentList:
                    numValidWavFiles += 1
                    instrumentDataPath = folderPath + filename

                    (instrumentWavData, sampleRateWav) = InstrumentDataParser.\
                        getInstrumentWavData(instrumentDataPath, sampleRate)

                    assert(sampleRate == sampleRateWav)

                    dividedInstrumentWavData = InstrumentDataParser.\
                        getDividedInstrumentWavData(instrumentWavData,
                                                    sampleRate,
                                                    numSpectrogramSamples)

                    if instrument in instrumentDataDict:
                        instrumentDataDict[instrument] = \
                            np.append(instrumentDataDict[instrument],
                                      dividedInstrumentWavData,
                                      axis = 1)
                    else:
                        instrumentDataDict[instrument] = \
                            dividedInstrumentWavData
                else:
                    numInvalidWavFiles += 1

        print("Number of valid .wav files found = " + str(numValidWavFiles))
        print("Number of invalid .wav files found = " + str(numInvalidWavFiles))
        assert(numValidWavFiles != 0)
        return instrumentDataDict


    @staticmethod
    def getInstrumentWavData(instrumentDataPath, sampleRate):
        (sampleRateWav, instrumentWavData) = \
            AudioHandler.getWavFile(instrumentDataPath, sampleRate)
        instrumentWavData = AudioHandler.convertToMono(instrumentWavData)
        return (instrumentWavData, sampleRateWav)


    @staticmethod
    def getDividedInstrumentWavData(instrumentWavData,
                                    sampleRate,
                                    numSpectrogramSamples):
        # print("Divide Instrument Data")
        numSections = \
            np.floor(len(instrumentWavData) / numSpectrogramSamples).astype(int)
        dividedInstrumentWavData = \
            np.zeros((numSpectrogramSamples, numSections))

        # print("Number of samples per section = " + str(numSpectrogramSamples))
        # print("Number of sections = " + str(numSections))
        # print("Total number of sample = " + str(len(instrumentWavData)))

        startSample = 0
        for section in range(numSections):
            endSample = startSample + numSpectrogramSamples
            dividedInstrumentWavData[:, section] = \
                instrumentWavData[startSample:endSample]
            startSample = endSample

        return dividedInstrumentWavData
