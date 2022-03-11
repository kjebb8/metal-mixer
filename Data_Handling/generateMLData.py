import numpy as np
from instrumentDataParser import InstrumentDataParser
from instrumentMixer import InstrumentMixer
from spectrogramMLDataGenerator import SpectrogramMLDataGenerator
from ..Spectrogram import spectrogramUtils as specUtils
from ..Spectrogram.spectrogramInputs import SpectrogramInputs
from ..Spectrogram.spectrogramParams import SpectrogramParams

'''
File: generateMLData.py
Brief: Generate ML data using the InstrumentDataParser, InstrumentMixer
       and SpectrogramMLDataGenerator. Provide a single function for other
       modules to use.
'''

class GenMLDataInputs:
    def __init__(self):
        self.instrumentDataPath = None
        self.instrumentList = None
        self.isoInstrumentList = None
        self.sampleRate = 2 ** 15
        self.numWindowSamples = 2 ** 10
        self.duration = None
        self.mixingFunction = None
        self.groupSize = None
        self.fractionUnmixed = None
        self.logFreqFactorDs = None
        self.isCompositeSpectrogram = False
        self.correctPowerForMusic = False
        self.spectrogramInputs = None
        self.numSpectrogramSamples = None
        self.numSpectrogramValues = None
        self.isCompositeSpectrogramMask = False
        self.correctPowerForMusicMask = False
        self.spectrogramInputsMask = None
        self.numSpectrogramValuesMask = None
        self.useBinMasks = True
        self.numExamples = None
        self.computeSpectrograms = True


    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.__dict__ == other.__dict__)


    def __str__(self):
        printStr = "\n"
        for par in vars(self).items():
            if par[0] == "spectrogramInputs" or \
               par[0] == "spectrogramInputsMask":
                printStr += "\n"
                printStr += "\t" + par[0] + ": " + str(par[1])
            else:
                printStr += "\t" + par[0] + ": " + str(par[1])
            printStr += "\n"
        return printStr


class GenMLDataOutputs:
    def __init__(self):
        # The instrumentDataDict isn't needed outside of data generation
        # so don't include in genOut to save memory
        self.mixedInstrumentsList = None
        self.inputSpectrogramVecs = None
        self.idealMasks = None


def setSpectrogramInput(genIn):

    genIn.spectrogramInputs = SpectrogramInputs(genIn.isCompositeSpectrogram)
    genIn.spectrogramInputs.correctPowerForMusic = genIn.correctPowerForMusic
    genIn.spectrogramInputs.sampleRate = genIn.sampleRate
    genIn.spectrogramInputs.numWindowSamples = genIn.numWindowSamples
    genIn.spectrogramInputs.logFreqFactorDs = genIn.logFreqFactorDs

    genIn.spectrogramInputsMask = \
        SpectrogramInputs(genIn.isCompositeSpectrogramMask)
    genIn.spectrogramInputsMask.correctPowerForMusic = \
        genIn.correctPowerForMusicMask
    genIn.spectrogramInputsMask.sampleRate = genIn.sampleRate
    genIn.spectrogramInputsMask.numWindowSamples = genIn.numWindowSamples
    genIn.spectrogramInputsMask.logFreqFactorDs = genIn.logFreqFactorDs

    targetSpectrogramSamples = specUtils.getSamplesfromTime(genIn.sampleRate,
                                                            genIn.duration)
    genIn.numSpectrogramSamples = \
        SpectrogramParams.getNumSpectrogramSamples( \
            genIn.spectrogramInputs.numWindowSamples,
            targetSpectrogramSamples,
            genIn.spectrogramInputs.overlap)

    # Make sure the number of time domain spectrogram samples for inputs and
    # masks are the same to ensure the the data is synchronized. (i.e. if the
    # mask uses less samples, when separating the audio using audio
    # spectrograms, the inputs (1000 samples) will make masks (900 samples) and
    # the masks won't match the spectrograms they are applied on)
    numSpectrogramSamplesMask = \
        SpectrogramParams.getNumSpectrogramSamples( \
            genIn.spectrogramInputsMask.numWindowSamples,
            targetSpectrogramSamples,
            genIn.spectrogramInputsMask.overlap)
    assert(genIn.numSpectrogramSamples == numSpectrogramSamplesMask)

    spectrogramParams = SpectrogramParams(genIn.spectrogramInputs,
                                          genIn.numSpectrogramSamples)
    print("Input Spectrogram Params:")
    spectrogramParams.printParams()
    genIn.numSpectrogramValues = spectrogramParams.numSpectrogramValues

    spectrogramParams = SpectrogramParams(genIn.spectrogramInputsMask,
                                          genIn.numSpectrogramSamples)
    print("Mask Spectrogram Params:")
    spectrogramParams.printParams()
    genIn.numSpectrogramValuesMask = spectrogramParams.numSpectrogramValues


def generateMLData(genIn):

    setSpectrogramInput(genIn)

    genOut = GenMLDataOutputs()

    instrumentDataDict = \
        InstrumentDataParser.getInstrumentDataDict(genIn.instrumentDataPath,
                                                   genIn.sampleRate,
                                                   genIn.numSpectrogramSamples,
                                                   genIn.instrumentList)

    genOut.mixedInstrumentsList = \
        InstrumentMixer.getMixedInstrumentsList(instrumentDataDict,
                                                genIn.mixingFunction,
                                                genIn.numExamples,
                                                genIn.groupSize,
                                                genIn.fractionUnmixed)

    # Since generating spectrograms takes a while, make it optional
    if genIn.computeSpectrograms:
        (genOut.inputSpectrogramVecs, genOut.idealMasks) = \
            SpectrogramMLDataGenerator.\
                getSpectrogramMLData(genOut.mixedInstrumentsList,
                                     genIn.isoInstrumentList,
                                     genIn.spectrogramInputs,
                                     genIn.numSpectrogramValues,
                                     genIn.numSpectrogramValuesMask,
                                     genIn.spectrogramInputsMask,
                                     genIn.useBinMasks)

    return (genOut)