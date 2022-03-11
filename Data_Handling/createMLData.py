import os
import time
import shutil
import numpy as np
from audioHandler import AudioHandler
from instrumentMixer import MixingFunctions
from generateMLData import GenMLDataInputs
from generateMLData import generateMLData
from trueSourceGenerator import TrueSourceGenerator
from ..Separation_Config.separationConfigInputs import SeparationConfigInputs

'''
File: createMLData.py
Brief: Script for generating ML data inputs (spectrograms and masks) and saving
       them to a file to be read by the ML program.
'''

trainingAudioName = "training_mixed.wav"

def createMLData(genIn, mLDataInputsPath):

    dataFolderName = getMLDataInputsFolderName(genIn, mLDataInputsPath)
    mLDataInputsFolder = mLDataInputsPath + dataFolderName + "/"
    createMLDataInputsFolder(mLDataInputsFolder)

    genOut = generateMLData(genIn)

    # Get the true audio sources and save wav data to a file
    trueSourcesDict = \
        TrueSourceGenerator.getTrueSources(genIn.instrumentList,
                                           genOut.mixedInstrumentsList)
    saveTrueSources(trueSourcesDict, genIn.sampleRate, mLDataInputsFolder)

    saveMLData(genOut.inputSpectrogramVecs,
               genOut.idealMasks,
               genIn.useBinMasks,
               mLDataInputsFolder)

    # Save a config file with info about the data for the experiment
    saveSeparationConfig(genIn, mLDataInputsFolder)


def getGenInValues(instrumentDataPath):
    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = instrumentDataPath
    genIn.instrumentList = ["drums", "piano"]
    genIn.isoInstrumentList = ["drums"]
    genIn.sampleRate = 2 ** 15
    genIn.numWindowSamples = 2 ** 10
    genIn.duration = 0.1  # seconds
    genIn.mixingFunction = MixingFunctions.fullMix
    genIn.groupSize = None
    genIn.fractionUnmixed = None
    genIn.logFreqFactorDs = 16
    genIn.isCompositeSpectrogram = False
    genIn.correctPowerForMusic = False
    genIn.isCompositeSpectrogramMask = False
    genIn.correctPowerForMusicMask = False
    genIn.useBinMasks = True
    genIn.numExamples = 50

    return genIn


def getMLDataInputsFolderName(genIn, mLDataInputsPath):

    dataFolderName = ""
    for inst in genIn.instrumentList:
        dataFolderName += inst + "-"
    dataFolderName = dataFolderName[:-1]

    nextFileIdx = 0
    for filename in os.listdir(mLDataInputsPath):
        filenameSplit = filename.split('_')
        if filenameSplit[0] == dataFolderName:
            fileIdx = int(filenameSplit[1])
            if fileIdx > nextFileIdx:
                nextFileIdx = fileIdx
    nextFileIdx += 1

    dataFolderName += "_" + str(nextFileIdx)

    return dataFolderName


def createMLDataInputsFolder(mLDataInputsFolder):
    if not os.path.isdir(mLDataInputsFolder):
        os.makedirs(mLDataInputsFolder)
    else:
        print("Error: folder " + mLDataInputsFolder + " already exists")
        assert(0)


def saveTrueSources(trueSourcesDict, sampleRate, mLDataInputsFolder):
    for inst in trueSourcesDict:
        samples = trueSourcesDict[inst]
        prefix = ""
        if inst != "mixed":
            prefix = "true_" + inst + "_"
        AudioHandler.writeWaveFile(\
            mLDataInputsFolder + prefix + trainingAudioName,
            sampleRate,
            samples)


def saveMLData(inputSpectrogramVecs, idealMasks, useBinMasks,
               mLDataInputsFolder):
    assert(np.max(inputSpectrogramVecs) <= 1)
    assert(np.min(inputSpectrogramVecs) >= 0)

    inputSpectrogramVecs = \
        (inputSpectrogramVecs * np.iinfo(np.uint16).max).astype(np.uint16)

    if useBinMasks:
        idealMasks = idealMasks.astype(np.bool)
    else:
        idealMasks = (idealMasks * np.iinfo(np.uint16).max).astype(np.uint16)

    print("Shape of input spectrograms " + str(inputSpectrogramVecs.shape))
    print("Shape of ideal masks " + str(idealMasks.shape))
    print("Input spectrograms type: " + str(inputSpectrogramVecs.dtype))
    print("Ideal masks type: " + str(idealMasks.dtype))

    np.save(mLDataInputsFolder + "inputSpectrogramVecs.npy",
            inputSpectrogramVecs)
    np.save(mLDataInputsFolder + "idealMasks.npy", idealMasks)


def saveSeparationConfig(genIn, mLDataInputsFolder):
    separationConfig = SeparationConfigInputs(genIn, mLDataInputsFolder)
    separationConfig.saveConfig()


def createMLDataTest():
    print("\nCreate ML Data Test Started\n")
    instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Create_ML_Data_Test/"
    mLDataInputsPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Code/" + \
        "Data_Handling/Create_ML_Data_Test/"

    genIn = getGenInValuesTest(instrumentDataPath)
    createMLData(genIn, mLDataInputsPath)

    refDataFolder = mLDataInputsPath + "Test_Data_Ref/"
    (_, truePianoRef) = \
        AudioHandler.getWavFile(refDataFolder + "true_piano_training_mixed.wav")
    (_, trueDrumsRef) = \
        AudioHandler.getWavFile(refDataFolder + "true_drums_training_mixed.wav")
    (_, trueMixedRef) = \
        AudioHandler.getWavFile(refDataFolder + "training_mixed.wav")
    inputSpectrogramVecsRef = \
        np.load(refDataFolder + "inputSpectrogramVecs.npy")
    idealMasksRef = np.load(refDataFolder + "idealMasks.npy")
    separationConfigRef = SeparationConfigInputs.loadConfig(refDataFolder)

    dataFolderName = "drums-piano_2"
    mLDataInputsFolder = mLDataInputsPath + dataFolderName + "/"
    (_, truePiano) = \
        AudioHandler.getWavFile(mLDataInputsFolder + \
                                "true_piano_training_mixed.wav")
    (_, trueDrums) = \
        AudioHandler.getWavFile(mLDataInputsFolder + \
                                "true_drums_training_mixed.wav")
    (_, trueMixed) = \
        AudioHandler.getWavFile(mLDataInputsFolder + "training_mixed.wav")
    inputSpectrogramVecs = \
        np.load(mLDataInputsFolder + "inputSpectrogramVecs.npy")
    idealMasks = np.load(mLDataInputsFolder + "idealMasks.npy")
    separationConfig = SeparationConfigInputs.loadConfig(mLDataInputsFolder)

    shutil.rmtree(mLDataInputsFolder)

    assert(np.array_equal(truePiano, truePianoRef))
    assert(np.array_equal(trueDrums, trueDrumsRef))
    assert(np.array_equal(trueMixed, trueMixedRef))
    assert(np.array_equal(inputSpectrogramVecs, inputSpectrogramVecsRef))
    assert(np.array_equal(idealMasks, idealMasksRef))
    assert(separationConfig == separationConfigRef)

    print("\nCreate ML Data Test Finished\n")


def getGenInValuesTest(instrumentDataPath):
    genIn = GenMLDataInputs()
    genIn.instrumentDataPath = instrumentDataPath
    genIn.instrumentList = ["drums", "piano"]
    genIn.isoInstrumentList = ["drums"]
    genIn.sampleRate = 2 ** 15
    genIn.duration = 0.2  # seconds
    genIn.mixingFunction = MixingFunctions.fullMix
    genIn.groupSize = None
    genIn.fractionUnmixed = None
    genIn.logFreqFactorDs = 16
    genIn.isCompositeSpectrogram = False
    genIn.correctPowerForMusic = False
    genIn.isCompositeSpectrogramMask = False
    genIn.correctPowerForMusicMask = False
    genIn.useBinMasks = True
    genIn.numExamples = 100

    return genIn


def main():
    print("\n\n----------Start Create ML Data Script----------\n\n")

    np.random.seed(0)
    createMLDataTest()

    instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "ML_Music/"
    mLDataInputsPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/" + \
        "ML_Experiments/ML_Data/Input_Data/"

    genIn = getGenInValues(instrumentDataPath)

    startTime = time.time()
    # createMLData(genIn, mLDataInputsPath)
    endTime = time.time()
    print("Time to Generate Data: " + str(endTime - startTime))

    print("\n\n----------Create ML Data Script Finished----------\n\n")
if __name__ == "__main__": main()
