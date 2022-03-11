import os
import numpy as np
import matplotlib.pyplot as plt
import constants as const
from separationStats import SeparationStats
from ..Data_Handling.audioHandler import AudioHandler
from ..Spectrogram.spectrogramInputs import SpectrogramInputs
from ..Spectrogram.audioSpectrograms import AudioSpectrograms
from ..Spectrogram.spectrogramUtils import *
from ..Separation_Config.separationConfig import SeparationConfig

'''
File: separationAnalysis.py
Brief: Analyze music separation results including:
        - Show the separation using a comparison of spectrograms
        - Calculate the Signal to Interference Ratio (SIR)
        - Calculate the Signal to Artifact Ratio (SAR)
        - Calculate the Normalized Signal to Distortion Ratio (NSDR)
'''

def separationAnalysis(separationConfigOutputs, musicPath, musicFile, useTrainingExamples = True,
                       showSpectrograms = False):

    separationConfigInputs = SeparationConfig.loadConfig(separationConfigOutputs.mLDataInputsFolder)
    instrumentList = separationConfigInputs.genIn.instrumentList
    sampleRate = separationConfigInputs.genIn.spectrogramInputsMask.sampleRate
    numSpectrogramSamples = separationConfigInputs.genIn.numSpectrogramSamples

    mLDataOutputsFolder = separationConfigOutputs.parentFolder

    (musicTimeData, truthInstrumentTimeData, isoInstrumentTimeData) = \
         loadMusicData(mLDataOutputsFolder, musicPath, musicFile, sampleRate, instrumentList)

    if truthInstrumentTimeData is not None:
        if useTrainingExamples:
            assert(len(musicTimeData) % numSpectrogramSamples == 0)
            numExamples = len(musicTimeData) / numSpectrogramSamples
        else:
            numExamples = 1

        (musicTimeDataExamples, truthTimeDataExamples, isoTimeDataExamples) = \
            getDataPerExample(musicTimeData, truthInstrumentTimeData,
                              isoInstrumentTimeData, numExamples)

        separationStatAnalysis(musicTimeDataExamples, truthTimeDataExamples,
                               isoTimeDataExamples, mLDataOutputsFolder, musicFile)
    else:
        print("Cannot run separationStatAnalysis. "
              "No ground truth values for this example.")

    if showSpectrograms:
        spectrogramAnalysis(musicTimeData, truthInstrumentTimeData,
                            isoInstrumentTimeData, sampleRate, instrumentList)


def loadMusicData(isoMusicPath, musicPath, musicFile, sampleRate, instrumentList):

    # Load the iso music data for each instrument
    (isoInstrumentTimeData, numSamples) = \
        getIsoInstrumentTimeData(isoMusicPath, musicFile, sampleRate, instrumentList)

    # If they exist, load the true time data for each instrument
    truthInstrumentTimeData = \
        getTruthInstrumentTimeData(musicPath, musicFile, sampleRate, instrumentList, numSamples)

    # Load the data for full mixed instrument music
    (sampleRateMusic, musicTimeData) = \
        AudioHandler.getWavFile(musicPath + musicFile)
    musicTimeData = AudioHandler.convertToMono(musicTimeData)
    musicTimeData = musicTimeData[:numSamples]
    assert(sampleRateMusic == sampleRate)

    return (musicTimeData, truthInstrumentTimeData, isoInstrumentTimeData)


def getIsoInstrumentTimeData(isoMusicPath, musicFile, sampleRate, instrumentList):
    isoInstrumentTimeData = {}
    for inst in instrumentList:
        isoFile = "iso_" + inst + "_" + musicFile
        (sampleRateMusic, isoTimeData) = \
            AudioHandler.getWavFile(isoMusicPath + isoFile)
        isoTimeData = AudioHandler.convertToMono(isoTimeData)
        assert(sampleRateMusic == sampleRate)
        isoInstrumentTimeData[inst] = isoTimeData

    assert(len(instrumentList) == len(isoInstrumentTimeData))

    numSamples = len(isoInstrumentTimeData.values()[0])
    for inst in isoInstrumentTimeData:
        assert(len(isoInstrumentTimeData[inst]) == numSamples)

    return (isoInstrumentTimeData, numSamples)


def getTruthInstrumentTimeData(musicPath, musicFile, sampleRate, instrumentList, numSamples):
    truthInstrumentTimeData = {}
    for inst in instrumentList:
        truthFile = "true_" + inst + "_" + musicFile

        if truthFile not in os.listdir(musicPath):
            return None

        (sampleRateMusic, truthTimeData) = \
            AudioHandler.getWavFile(musicPath + truthFile)
        truthTimeData = AudioHandler.convertToMono(truthTimeData)
        assert(sampleRate == sampleRateMusic)
        truthInstrumentTimeData[inst] = truthTimeData[:numSamples]

    assert(len(instrumentList) == len(truthInstrumentTimeData))

    return truthInstrumentTimeData


def separationStatAnalysis(musicTimeDataExamples, truthTimeDataExamples,
                           isoTimeDataExamples, mLDataOutputsFolder, musicFile):
    print("Start Separation Statistic Analysis \n")

    instrumentList = truthTimeDataExamples.keys()

    separationStats = \
        SeparationStats.loadStats(mLDataOutputsFolder + "separationStats.pkl")

    if separationStats is None:
        separationStats = SeparationStats()

    numExamples = musicTimeDataExamples.shape[1]
    print("Perform separation analysis on " + str(numExamples) + " examples")

    # For each example calculate the separation stats
    numTestedExamples = 0
    for exIdx in range(numExamples):

        truthTimeData = np.array([])
        isoTimeData = np.array([])
        mixedTimeData = np.array([])

        exampleInstList = []
        for inst in instrumentList:

            # Only use data for an instrument that has meaningful (non-zero)
            # true signal
            if not np.all(np.abs(truthTimeDataExamples[inst][:, exIdx]) < \
                                 const.minAudioAmp):
                exampleInstList.append(inst)
                truthTimeData = \
                    np.append(truthTimeData,
                              truthTimeDataExamples[inst][:, exIdx])
                isoTimeData = \
                    np.append(isoTimeData,
                              isoTimeDataExamples[inst][:, exIdx])
                mixedTimeData = \
                    np.append(mixedTimeData,
                              musicTimeDataExamples[:, exIdx])

        # If there are two or more sources, get the separation statistics
        numInst = len(exampleInstList)
        if numInst > 1:
            numTestedExamples += 1
            truthTimeData = truthTimeData.reshape(numInst, -1)
            isoTimeData = isoTimeData.reshape(numInst, -1)
            mixedTimeData = mixedTimeData.reshape(numInst, -1)

            separationStats.calculateStats(musicFile, exampleInstList,
                                           truthTimeData, isoTimeData,
                                           mixedTimeData)

        else:
            print("Eliminating example: " + str(exIdx))

    print("Number of tested examples: " + str(numTestedExamples))

    separationStats.printMeanStats()

    separationStats.saveStats(mLDataOutputsFolder + "separationStats.pkl")



def getDataPerExample(musicTimeData, truthInstrumentTimeData,
                     isoInstrumentTimeData, numExamples):
    # Reshape the examples so there is one example in each column
    musicTimeDataExamples = musicTimeData.reshape(-1, numExamples, order="F")
    truthTimeDataExamples = {}
    isoTimeDataExamples = {}
    for inst in truthInstrumentTimeData:
        truthTimeDataExamples[inst] = \
            truthInstrumentTimeData[inst].reshape(-1, numExamples, order="F")
        isoTimeDataExamples[inst] = \
            isoInstrumentTimeData[inst].reshape(-1, numExamples, order="F")
        assert(isoTimeDataExamples[inst].shape == \
               truthTimeDataExamples[inst].shape)
        assert(isoTimeDataExamples[inst].shape == musicTimeDataExamples.shape)

    return (musicTimeDataExamples, truthTimeDataExamples, isoTimeDataExamples)


def spectrogramAnalysis(musicTimeData, truthInstrumentTimeData,
                        isoInstrumentTimeData, sampleRate, instrumentList):
    print("\nStart Spectrogram Analysis \n")

    inst1 = instrumentList[0]
    targetSpectrogramTimeSec = 3.
    correctPowerForMusic = False
    isCompositeSpectrogram = False

    targetSpectrogramSamples = \
        getSamplesfromTime(sampleRate, targetSpectrogramTimeSec)

    spectrogramInputs = SpectrogramInputs(isCompositeSpectrogram)
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.correctPowerForMusic = correctPowerForMusic

    # Composite Spectrogram
    musicSpecs = AudioSpectrograms(spectrogramInputs,
                                   targetSpectrogramSamples,
                                   musicTimeData)

    if truthInstrumentTimeData is not None:
        trueSpecsDict = {}
        for inst in instrumentList:
            specs = AudioSpectrograms(spectrogramInputs,
                                    targetSpectrogramSamples,
                                    truthInstrumentTimeData[inst])
            trueSpecsDict[inst] = specs

    isoSpecsDict = {}
    for inst in instrumentList:
        specs = AudioSpectrograms(spectrogramInputs,
                                  targetSpectrogramSamples,
                                  isoInstrumentTimeData[inst])
        isoSpecsDict[inst] = specs

    musicTime = getTimefromSamples(sampleRate, len(musicTimeData))
    isoTime = getTimefromSamples(sampleRate,len(isoInstrumentTimeData[inst1]))
    print("Music time: " + str(musicTime))
    print("Iso time: " + str(isoTime))
    print("Num music specs: " + str(musicSpecs.numSpectrograms))
    print("Num iso specs: " + str(isoSpecsDict[inst1].numSpectrograms) + "\n")

    assert(musicSpecs.numSpectrograms == isoSpecsDict[inst1].numSpectrograms)

    numSpectrograms = musicSpecs.numSpectrograms
    startTime = 0
    for specIdx in range(min(numSpectrograms, 3)):

        endTime = startTime + targetSpectrogramTimeSec
        print("Start: " + str(startTime) + "\tEnd: " + str(endTime))
        startTime = endTime

        print("mixed Spectrogram:")
        musicSpec = musicSpecs.spectrogramList[specIdx]
        musicSpec.plotSpectrogram()

        for inst in isoSpecsDict:
            print("iso " + inst + " Spectrogram:")
            isoSpec = isoSpecsDict[inst].spectrogramList[specIdx]
            isoSpec.plotSpectrogram()

            if truthInstrumentTimeData is not None:
                print("true " + inst + " Spectrogram:")
                trueSpec = trueSpecsDict[inst].spectrogramList[specIdx]
                trueSpec.plotSpectrogram()


def separationStatAnalysisTest():
    print("\nSeparation Stat Analysis Test Started\n")

    testPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Code/" + \
        "Music_Separation/Separation_Stat_Analysis_Test/"

    mLDataOutputsFolder = testPath + "Test_Data_Outputs/"
    separationConfigOutputs = SeparationConfig.loadConfig(mLDataOutputsFolder)
    musicPath = testPath + "Test_Data_Inputs/"
    musicFile = "training_mixed.wav"

    separationAnalysis(separationConfigOutputs, musicPath, musicFile)

    refDataFolder = testPath + "Test_Data_Ref/"
    statsRef = SeparationStats.loadStats(refDataFolder + "separationStats.pkl")

    stats = SeparationStats.loadStats(mLDataOutputsFolder + "separationStats.pkl")

    os.remove(mLDataOutputsFolder + "separationStats.pkl")

    assert(statsRef == stats)

    print("\nSeparation Stat Analysis Test Finished\n")



def main():
    print("\n\n----------Music Separation Analysis Starting----------\n\n")

    separationStatAnalysisTest()

    useTrainingExamples = True
    showSpectrograms = True

    mLDataOutputsFolder = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/ML_Experiments/ML_Data/" + \
        "Exp_4/drums-piano_4_duration-500ms/Exp_4_7/"

    separationConfigOutputs = SeparationConfig.loadConfig(mLDataOutputsFolder)

    if useTrainingExamples:
        musicPath = separationConfigOutputs.mLDataInputsFolder
        musicFile = "training_mixed.wav"
    else:
        musicPath = \
            "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
            "Music_Separation_Test_Dataset/"
        musicFile = "drums_piano_mixed.wav"


    # separationAnalysis(separationConfigOutputs, musicPath, musicFile, useTrainingExamples,
    #                    showSpectrograms)


    print("\n\n----------Music Separation Analysis Finished----------\n\n")

if __name__ == "__main__": main()