import numpy as np
from trueSourceGenerator import TrueSourceGenerator
from audioHandler import AudioHandler
from instrumentMixer import MixingFunctions
from generateMLData import GenMLDataInputs
from generateMLData import generateMLData
from ..Spectrogram import spectrogramUtils as specUtils

'''
File: trueSourceGeneratorTest.py
Brief: Test the TrueSourceGenerator Class functions.
'''

# Test the true sources agains reference values (ear tested)
def trueSourceGeneratorTest1():

    print("\nTrue Source Generator Test 1 Started\n")

    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"

    genIn.sampleRate = 2 ** 15
    genIn.duration = 2.2  # seconds
    genIn.instrumentList = ["piano", "acousticGuitar"]
    genIn.isoInstrumentList = ["piano"]
    genIn.mixingFunction = MixingFunctions.fullMix
    genIn.computeSpectrograms = False
    np.random.seed(0)

    genOut = generateMLData(genIn)

    # Increase the audio amplitude for a better test of scaling
    for mixedInstrumentsDict in genOut.mixedInstrumentsList:
        for inst in mixedInstrumentsDict:
            mixedInstrumentsDict[inst] = mixedInstrumentsDict[inst] * 2

    trueSourcesDict = \
        TrueSourceGenerator.getTrueSources(genIn.instrumentList,
                                           genOut.mixedInstrumentsList)

    checkScaling(genOut.mixedInstrumentsList, trueSourcesDict)

    sectionsToTest = 5
    numTestSamples = \
        specUtils.getSamplesfromTime(genIn.sampleRate,
                                     sectionsToTest * genIn.duration)

    for inst in trueSourcesDict:
        samples = trueSourcesDict[inst][:numTestSamples]
        refSamples = \
            np.load('./Code/Data_Handling/true_source_test1_' + inst + '.npy')
        assert(np.array_equal(samples,refSamples))
        # np.save('./Code/Data_Handling/true_source_test1_' + inst + '.npy',
        #         samples)

    print("\nTrue Source Generator Test 1 Finish\n")


def checkScaling(mixedInstrumentsList, trueSourcesDict):

    maxAmp = np.iinfo(np.int16).max

    # Take the first Dict in the mixedInstrumentList to test. Make sure it has
    # multiple instruments.
    mixedInstrumentsDict = mixedInstrumentsList[0]
    numSamples = mixedInstrumentsDict.values()[0].shape[0]
    assert(len(mixedInstrumentsDict.values()) > 1)

    # Get the unscaled mix and verify the max value is above the bit limit
    # (i.e. this Dict is a valid example to test)
    unscaledMix = np.zeros(numSamples,)
    for inst in mixedInstrumentsDict:
        unscaledMix += mixedInstrumentsDict[inst]
    assert(np.max(np.abs(unscaledMix)) > maxAmp)

    # Get the scaled mix by adding the sources from the trueSourcesDict
    scaledMix = np.zeros(numSamples,)
    mixedSamples = np.zeros(numSamples,)
    for inst in trueSourcesDict:
        if inst == "mixed":
            mixedSamples = trueSourcesDict[inst][:numSamples]
        else:
            scaledMix += trueSourcesDict[inst][:numSamples]

    # Verify the summation of the individual sources matches the mixed samples
    assert(np.array_equal(np.around(scaledMix, decimals=8),
                          np.around(mixedSamples, decimals=8)))

    # Verify the scaled and unscaled samples are the same audio
    scaleFactor = (scaledMix[0] / unscaledMix[0])
    assert(np.array_equal(np.around(scaledMix, decimals=8),
                          np.around(unscaledMix * scaleFactor, decimals=8)))

    # Verify the summation of the individual sources is within the bit limit
    assert(np.max(np.abs(scaledMix)) <= maxAmp)


def main():
    print("\n\n----------True Source Gen Test Starting----------\n\n")

    trueSourceGeneratorTest1()

    print("\n\n----------True Source Gen Test Finished----------\n\n")
if __name__ == "__main__": main()