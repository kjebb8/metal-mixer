import copy
import numpy as np
import constants as const
import matplotlib.pyplot as plt
from audioHandler import AudioHandler
from instrumentMixer import MixingFunctions
from generateMLData import GenMLDataInputs
from generateMLData import generateMLData
from ..Spectrogram import spectrogramUtils as specUtils
from ..Spectrogram.spectrogramInputs import SpectrogramInputs
from ..Spectrogram.spectrogramParams import SpectrogramParams
from ..Spectrogram.spectrogram import Spectrogram

'''
File: spectrogramMLDataGeneratorTest.py
Brief: Test the SpectrogramMLDataGenerator Class functions.
'''
# Test one isolated instrument + mixed random groups + non-composite spectrogram
def specMLDataGeneratorTest1():

    print("\nSpectrogram ML Data Generator Test 1 Started\n")

    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"

    genIn.duration = 4.3  # seconds
    genIn.mixingFunction = MixingFunctions.randomGroups
    genIn.groupSize = 2
    genIn.isoInstrumentList = ["piano"]
    genIn.useBinMasks = False
    genIn.numExamples = 10
    np.random.seed(0)

    genOut = generateMLData(genIn)

    spectrogramInputs = SpectrogramInputs()
    sampleRate = spectrogramInputs.sampleRate
    targetSpectrogramSamples = specUtils.getSamplesfromTime(sampleRate,
                                                            genIn.duration)
    numSpectrogramSamples = \
        SpectrogramParams.getNumSpectrogramSamples( \
            spectrogramInputs.numWindowSamples,
            targetSpectrogramSamples,
            spectrogramInputs.overlap)

    spectrogramParams = SpectrogramParams(spectrogramInputs,
                                          numSpectrogramSamples)

    # Test the correctness of the array dimensions
    numSpectrogramValues = spectrogramParams.numSpectrogramValues
    numSpectrograms = len(genOut.mixedInstrumentsList)
    assert(genOut.inputSpectrogramVecs.shape == \
           (numSpectrogramValues, numSpectrograms))
    assert(genOut.idealMasks.shape == \
           (numSpectrogramValues * len(genIn.isoInstrumentList),
            numSpectrograms))

    # Test the correctness of the input spectrograms
    testSpecs = []
    testSpecVecs = np.zeros((numSpectrogramValues, numSpectrograms))
    for specIdx in range(numSpectrograms):
        testData = np.zeros((genIn.numSpectrogramSamples,))
        for instIdx in genOut.mixedInstrumentsList[specIdx]:
            testData += genOut.mixedInstrumentsList[specIdx][instIdx]

        testSpec = Spectrogram(genIn.spectrogramInputs, testData)
        testSpecs.append(testSpec)
        testSpecVecs[:, specIdx] = testSpec.getPowerDbVecUnityScale()

    assert(np.array_equal(genOut.inputSpectrogramVecs, testSpecVecs))

    # Test the correctness of the ideal masks
    mixedTimeData = None
    for specIdx in range(numSpectrograms):

        if specIdx == 0:
            assert(len(genOut.mixedInstrumentsList[specIdx]) == 2)
            mixedTimeData = testSpecs[specIdx].getTimeRepresentation()

        isoSpec = testSpecs[specIdx]
        idealMask = genOut.idealMasks[:, specIdx]
        isoSpec.applyPowerDbMask(idealMask)
        isoVec = isoSpec.getPowerDbVec()

        testData = np.zeros((genIn.numSpectrogramSamples,))
        if genIn.isoInstrumentList[0] in genOut.mixedInstrumentsList[specIdx]:
            testData = \
                genOut.mixedInstrumentsList[specIdx][genIn.isoInstrumentList[0]]

        testSpec = Spectrogram(genIn.spectrogramInputsMask, testData)
        testVec = testSpec.getPowerDbVec()

        assert(np.array_equal(np.around(testVec, decimals=8),
                              np.around(isoVec, decimals=8)))

        if specIdx == 0:
            assert(len(genOut.mixedInstrumentsList[specIdx]) == 2)
            idealMaskTest = np.load('./Code/Data_Handling/test1_idealMask.npy')
            assert(np.array_equal(np.around(idealMask, decimals=3),
                                  np.around(idealMaskTest, decimals=3)))

            # np.save('./Code/Data_Handling/test1_idealMask.npy', idealMask)

    print("\nSpectrogram ML Data Generator Test 1 Finish\n")


# Test one isolated instrument + fractional unmixed + composite spectrogram
# on input and non-composite spectrogram for BINARY masks
def specMLDataGeneratorTest2():

    print("\nSpectrogram ML Data Generator Test 2 Started\n")

    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"

    genIn.duration = 3.7  # seconds
    genIn.mixingFunction = MixingFunctions.fractionUnmixed
    genIn.fractionUnmixed = 0.1
    genIn.instrumentList = ["piano", "acousticGuitar"]
    genIn.isoInstrumentList = ["acousticGuitar"]
    genIn.isCompositeSpectrogram = True
    genIn.numExamples = 11 # 11 examples to get one with just acousticGuitar
    np.random.seed(0)

    genOut = generateMLData(genIn)

    verifyMLData(genIn, genOut, "2")

    print("\nSpectrogram ML Data Generator Test 2 Finish\n")


# Test two isolated instruments + fractional unmixed + composite spectrogram
# on input and BINARY masks
def specMLDataGeneratorTest3():

    print("\nSpectrogram ML Data Generator Test 3 Started\n")

    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"

    genIn.duration = 1.6  # seconds
    genIn.mixingFunction = MixingFunctions.fractionUnmixed
    genIn.fractionUnmixed = 0.15
    genIn.instrumentList = ["piano", "acousticGuitar"]
    genIn.isoInstrumentList = ["piano", "acousticGuitar"]
    genIn.isCompositeSpectrogram = True
    genIn.isCompositeSpectrogramMask = True
    genIn.numExamples = 10
    np.random.seed(0)

    genOut = generateMLData(genIn)

    verifyMLData(genIn, genOut, "3")

    print("\nSpectrogram ML Data Generator Test 3 Finish\n")



# Test two isolated instruments + full mix + non-composite
# spectrogram for input and BINARY masks
def specMLDataGeneratorTest4():

    print("\nSpectrogram ML Data Generator Test 4 Started\n")

    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"

    genIn.duration = 5.  # seconds
    genIn.instrumentList = ["piano", "acousticGuitar"]
    genIn.isoInstrumentList = ["piano", "acousticGuitar"]
    genIn.mixingFunction = MixingFunctions.fullMix
    genIn.numExamples = 10
    np.random.seed(0)

    genOut = generateMLData(genIn)

    verifyMLData(genIn, genOut, "4")

    print("\nSpectrogram ML Data Generator Test 4 Finish\n")


def testInputSpecVecs(spectrogramInputs,
                      numSpectrogramSamples,
                      numSpectrogramValues,
                      numExamples,
                      mixedInstrumentsList,
                      inputSpectrogramVecs,
                      testNo):

    # Test the correctness of the array dimensions
    numSpectrograms = len(mixedInstrumentsList)
    assert(numExamples == numSpectrograms)
    assert(inputSpectrogramVecs.shape == \
           (numSpectrogramValues, numSpectrograms))

    # Get correct values of the input spectrograms from the mixedInstrumentsList
    testSpecVecs = np.zeros((numSpectrogramValues, numSpectrograms))
    for specIdx in range(numSpectrograms):
        mixedInstrumentDict = mixedInstrumentsList[specIdx]
        (testData, _) = AudioHandler.mixAudio(mixedInstrumentDict.values())
        testSpec = Spectrogram(spectrogramInputs, testData)
        testSpecVecs[:, specIdx] = testSpec.getPowerDbVecUnityScale()

    # Test the correctness of a certain number if input spectrograms
    assert(np.array_equal(inputSpectrogramVecs[:, :numSpectrograms],
                          testSpecVecs))

    # Test one of the spectrogram entries (idx1) matches the test value
    testVec = np.load('./Code/Data_Handling/test' + testNo + '_inputSpec.npy')
    assert(np.array_equal(\
           np.around(inputSpectrogramVecs[:, 1], decimals=8),
           np.around(testVec, decimals=8)))
    # testVec = testSpecVecs[:, 1]
    # np.save('./Code/Data_Handling/test' + testNo + '_inputSpec.npy', testVec)


def testIdealBinMask(instrumentList,
                     isoInstrumentList,
                     spectrogramInputsMask,
                     numSpectrogramSamples,
                     numSpectrogramValuesMask,
                     numExamples,
                     mixedInstrumentsList,
                     idealMasks,
                     testNo):

    # Test the correctness of the array dimensions
    numSpectrograms = len(mixedInstrumentsList)
    assert(numExamples == numSpectrograms)
    assert(idealMasks.shape == \
           (numSpectrogramValuesMask * len(isoInstrumentList), numSpectrograms))

    # Test the correctness of the binary masks
    # Assume that the mixedInstrumentDictionaries have either solo instruments
    # or all the possible instruments mixed together

    # Get an example of masks for mixed and for each instrument by itself
    testDictIdx = {}
    for specIdx in range(numSpectrograms):
        mixedInstrumentsDict = mixedInstrumentsList[specIdx]
        if len(mixedInstrumentsDict) == len(instrumentList):
            if "mixed" not in testDictIdx:
                testDictIdx["mixed"] = specIdx
        elif len(mixedInstrumentsDict) == 1:
            inst = mixedInstrumentsDict.keys()[0] # Idx 0 since the len is 1
            if inst not in testDictIdx:
                testDictIdx[inst] = specIdx
        if len(testDictIdx) == (len(instrumentList) + 1):
            break

    # Test the single instrument masks
    numIsoInst = len(isoInstrumentList)
    for inst in instrumentList:
        instrumentsDict = mixedInstrumentsList[testDictIdx[inst]]
        spec = Spectrogram(spectrogramInputsMask, instrumentsDict[inst])
        specVec = spec.getPowerDbVec()
        instMasks = idealMasks[:, testDictIdx[inst]].\
                    reshape(-1, numIsoInst, order='F')
        for isoInstIdx in range(len(isoInstrumentList)):
            if isoInstrumentList[isoInstIdx] == inst:
                for freqIdx in range(numSpectrogramValuesMask):
                    isoPower = specVec[freqIdx]
                    if isoPower < const.minPowerBinMask:
                        assert(instMasks[freqIdx, isoInstIdx] == 0)
                    else:
                        assert(instMasks[freqIdx, isoInstIdx] == 1)
            else:
                zeroMask = instMasks[:, isoInstIdx]
                assert(np.all(zeroMask == 0))

    # Test the mixed mask
    mixedIdx = testDictIdx["mixed"]
    mask = idealMasks[:, mixedIdx].reshape(-1, numIsoInst, order='F')
    mixedInstrumentsDict = mixedInstrumentsList[mixedIdx]
    specVecDict = {}
    for inst in instrumentList:
        spec = Spectrogram(spectrogramInputsMask, mixedInstrumentsDict[inst])
        specVecDict[inst] = spec.getPowerDbVec()

    for isoInstIdx in range(len(isoInstrumentList)):
        isoInst = isoInstrumentList[isoInstIdx]
        for freqIdx in range(numSpectrogramValuesMask):
            isoPower = specVecDict[isoInst][freqIdx]
            if isoPower < const.minPowerBinMask:
                assert(mask[freqIdx, isoInstIdx] == 0)
                continue
            for inst in instrumentList:
                if inst == isoInst:
                    continue
                instPower = specVecDict[inst][freqIdx]
                if isoPower < instPower:
                    assert(mask[freqIdx, isoInstIdx] == 0)
                    break
                if inst == instrumentList[-1]:
                    assert(mask[freqIdx, isoInstIdx] == 1)

    testVec = np.load('./Code/Data_Handling/test' + testNo + '_idealMask.npy')
    assert(np.array_equal(idealMasks[:, mixedIdx], testVec))
    # np.save('./Code/Data_Handling/test' + testNo + '_idealMask.npy',
    #         idealMasks[:, mixedIdx])


def verifyMLData(genIn, genOut, testNo):

    # Test the correctness of the input spectrograms
    testInputSpecVecs(genIn.spectrogramInputs,
                      genIn.numSpectrogramSamples,
                      genIn.numSpectrogramValues,
                      genIn.numExamples,
                      genOut.mixedInstrumentsList,
                      genOut.inputSpectrogramVecs,
                      testNo)

    # Test the correctness of the ideal masks
    testIdealBinMask(genIn.instrumentList,
                     genIn.isoInstrumentList,
                     genIn.spectrogramInputsMask,
                     genIn.numSpectrogramSamples,
                     genIn.numSpectrogramValuesMask,
                     genIn.numExamples,
                     genOut.mixedInstrumentsList,
                     genOut.idealMasks,
                     testNo)


def main():
    print("\n\n----------Spec ML Data Gen Test Starting----------\n\n")

    specMLDataGeneratorTest1()
    specMLDataGeneratorTest2()
    specMLDataGeneratorTest3()
    specMLDataGeneratorTest4()

    print("\n\n----------Spec ML Data Gen Test Finished----------\n\n")
if __name__ == "__main__": main()
