import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from ..Machine_Learning.neuralNetwork import NeuralNetwork
from ..Data_Handling.audioHandler import AudioHandler
from ..Spectrogram.spectrogramInputs import SpectrogramInputs
from ..Spectrogram.audioSpectrograms import AudioSpectrograms
from ..Spectrogram.spectrogramUtils import *
from ..Separation_Config.separationConfig import SeparationConfig


'''
File: musicSeparation.py
Brief: Script that takes the learned weights and biases from a machine learning
       algorithm and applies them to spectrograms of music to separate out an
       instrument. Save the separated .wav file to hear the separation
'''

'''
Function: musicSeparation()
Description:
    1. Load the config file describing the ML data parameters
    2. Load the music .wav file, verify the sampling rate
    3. Transform the music into a list of spectrograms using AudioSpectrograms
    4. Get the spectrograms as a unity scale matrix of power
    5. Use forward propagation to predict a binary mask for each spectrogram
    6. Apply the binary masks to the spectrograms to isolate instruments
    7. Convert the spectrograms representation back to time representation
    8. Save the isolated instrument music to .wav file
'''
def musicSeparation(separationConfigOutputs, musicPath, musicFile):

    # 1. Load the config file describing the ML data parameters
    separationConfigInputs = SeparationConfig.loadConfig(separationConfigOutputs.mLDataInputsFolder)
    instrumentList = separationConfigInputs.genIn.instrumentList
    isoInstrumentList = separationConfigInputs.genIn.isoInstrumentList
    spectrogramInputsMask = separationConfigInputs.genIn.spectrogramInputsMask
    sampleRate = spectrogramInputsMask.sampleRate
    numSpectrogramSamples = separationConfigInputs.genIn.numSpectrogramSamples
    numSpectrogramValuesMask = separationConfigInputs.genIn.numSpectrogramValuesMask

    mLDataOutputsFolder = separationConfigOutputs.parentFolder
    layerDims = separationConfigOutputs.mLConfig.layerDims

    numSources = len(instrumentList)
    numIsoSources = len(isoInstrumentList)
    assert(numSources <= (numIsoSources + 1))

    # 2. Load the music .wav file, verify the sampling rate
    musicTimeData = getWavFile(musicPath, musicFile, sampleRate)

    # 3. Transform the music into a list of spectrograms using AudioSpectrograms
    startTimeConvertToSpec = time.time()
    musicSpecsList = getMusicSpecsList(numSources,
                                       spectrogramInputsMask,
                                       numSpectrogramSamples,
                                       musicTimeData)
    endTimeConvertToSpec = time.time()

    musicDuration = getTimefromSamples(sampleRate, len(musicTimeData))
    spectrogramDuration = \
        getTimefromSamples(sampleRate, numSpectrogramSamples)

    print("Music duration: " + str(musicDuration))
    print("Spectrogram duration: " + str(spectrogramDuration))
    print("Number of spectrograms " + str(musicSpecsList[0].numSpectrograms))
    print("\nTime to convert music to spectrograms: " + \
          str(endTimeConvertToSpec - startTimeConvertToSpec))

    # 4. Get the spectrograms as a unity scale matrix of power
    startTimeGetUnityMtx = time.time()
    specInputMatrix = musicSpecsList[0].getPowerDbMatrixUnityScale()
    endTimeGetUnityMtx = time.time()
    print("Time to get unity scale matrix: " + \
          str(endTimeGetUnityMtx - startTimeGetUnityMtx))

    # 5. Use forward propagation to predict a binary mask for each spectrogram
    startTimePredictMasks = time.time()
    predictedIsoMasks = predictIsoMasks(layerDims,
                                        mLDataOutputsFolder,
                                        specInputMatrix)
    sourceMasksList = getSourceMasksList(numSources,
                                         numIsoSources,
                                         numSpectrogramValuesMask,
                                         predictedIsoMasks)
    endTimePredictMasks = time.time()
    print("Time to predict the spectrogram masks: " + \
          str(endTimePredictMasks - startTimePredictMasks))

    # 6. Apply the binary masks to the spectrograms to isolate instruments
    startTimeApplyMasks = time.time()
    applyPowerDbMasks(numSources, musicSpecsList, sourceMasksList)
    endTimeApplyMasks = time.time()
    print("Time to apply the spectrogram masks: " + \
          str(endTimeApplyMasks - startTimeApplyMasks))

    # 7. Convert the spectrograms representation back to time representation
    startTimeConvertToTime = time.time()
    isoTimeDataList = getIsoTimeDataList(numSources, musicSpecsList)
    endTimeConvertToTime = time.time()
    print("Time to convert isolated spectrograms back to time: " + \
          str(endTimeConvertToTime - startTimeConvertToTime) + "\n")

    # 8. Save the isolated instrument music to .wav file
    saveIsoWavFiles(numSources, mLDataOutputsFolder, musicFile, sampleRate, instrumentList,
                    isoTimeDataList)


def getWavFile(musicPath, musicFile, sampleRate):
    (sampleRateMusic, musicTimeData) = \
        AudioHandler.getWavFile(musicPath + musicFile)
    musicTimeData = AudioHandler.convertToMono(musicTimeData)
    assert(sampleRateMusic == sampleRate)

    return musicTimeData


def getMusicSpecsList(numSources,
                      spectrogramInputsMask,
                      numSpectrogramSamples,
                      musicTimeData):
    musicSpecs = AudioSpectrograms(spectrogramInputsMask,
                                   numSpectrogramSamples,
                                   musicTimeData)
    musicSpecsList = [musicSpecs]
    for _ in range(1, numSources):
        musicSpecsList.append(copy.deepcopy(musicSpecs))

    return musicSpecsList


def getSourceMasksList(numSources,
                       numIsoSources,
                       numSpectrogramValuesMask,
                       predictedIsoMasks):
    sourceMasksList = []
    for src in range(numIsoSources):
        srcStart = src * numSpectrogramValuesMask
        srcEnd = srcStart + numSpectrogramValuesMask
        sourceMasksList.append(predictedIsoMasks[srcStart:srcEnd, :])

    # If one less iso source than source, get all the mask values for the iso
    # sources and set the last source mask to the logical not of the others.
    # Only works for BIN masks right now
    if numIsoSources < numSources:
        isoSrcOnlyMask = np.zeros((sourceMasksList[0].shape))
        for src in range(numIsoSources):
            isoSrcOnlyMask = \
                np.logical_or(isoSrcOnlyMask.astype(np.bool),
                              sourceMasksList[src].astype(np.bool))
        sourceMasksList.append(np.invert(isoSrcOnlyMask.astype(np.bool)))
        sourceMasksList[-1] = sourceMasksList[-1].astype(np.float64)

    assert(len(sourceMasksList) == numSources)

    return sourceMasksList


def predictIsoMasks(layerDims, mLDataOutputsFolder, specInputMatrix):
    nn = NeuralNetwork(layerDims)
    nn.loadParams(mLDataOutputsFolder)
    predictedIsoMasks = nn.predict(specInputMatrix)

    # Plot the weights and biases against frequency
    # Each point isn't a different frequency but a series of points in time
    # for one frequency then another series in time for the next frequency
    # plt.plot(np.sum(nn.weights, axis=1))
    # plt.title('Sum of Weights over Freq')
    # plt.show()
    # plt.plot(np.sum(nn.biases, axis=1))
    # plt.title('Sum of Biases over Freq')
    # plt.show()

    return predictedIsoMasks


def applyPowerDbMasks(numSources, musicSpecsList, sourceMasksList):
    for src in range(numSources):
        musicSpecsList[src].applyPowerDbMasks(sourceMasksList[src])


def getIsoTimeDataList(numSources, musicSpecsList):
    isoTimeDataList = []
    for musicSpecSrc in musicSpecsList:
        isoTimeDataList.append(musicSpecSrc.getTimeRepresentation())

    return isoTimeDataList


def saveIsoWavFiles(numSources, mLDataOutputsFolder, musicFile, sampleRate, instrumentList,
                    isoTimeDataList):
    for src in range(numSources):
        inst = instrumentList[src]
        fileName = mLDataOutputsFolder + "iso_" + inst + "_" + musicFile
        AudioHandler.writeWaveFile(fileName,
                                   sampleRate,
                                   isoTimeDataList[src])


def musicSeparationTest():
    print("\nMusic Separation Test Started\n")

    testPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Code/" + \
        "Music_Separation/Music_Separation_Test/"

    # Use these to avoid moving the large weights file into the Code repo
    mLDataOutputsFolder = testPath + "Test_Data_Outputs/"
    separationConfigOutputs = SeparationConfig.loadConfig(mLDataOutputsFolder)
    musicPath = testPath + "Test_Data_Inputs/"
    musicFile = "drums_piano_mixed.wav"

    musicSeparation(separationConfigOutputs, musicPath, musicFile)

    refDataFolder = testPath + "Test_Data_Ref/"

    (sampleRateRef1, drumsIsoRef) = \
        AudioHandler.getWavFile(refDataFolder + "iso_drums_" + musicFile)
    drumsIsoRef = AudioHandler.convertToMono(drumsIsoRef)

    (sampleRateRef2, pianoIsoRef) = \
        AudioHandler.getWavFile(refDataFolder + "iso_piano_" + musicFile)
    pianoIsoRef = AudioHandler.convertToMono(pianoIsoRef)


    (sampleRate1, drumsIso) = \
        AudioHandler.getWavFile(mLDataOutputsFolder + "iso_drums_" + musicFile)
    drumsIso = AudioHandler.convertToMono(drumsIso)

    (sampleRate2, pianoIso) = \
        AudioHandler.getWavFile(mLDataOutputsFolder + "iso_piano_" + musicFile)
    pianoIso = AudioHandler.convertToMono(pianoIso)

    os.remove(mLDataOutputsFolder + "iso_drums_" + musicFile)
    os.remove(mLDataOutputsFolder + "iso_piano_" + musicFile)

    assert(sampleRateRef1 == sampleRateRef2)
    assert(sampleRate1 == sampleRate2)
    assert(sampleRateRef1 == sampleRate1)

    assert(np.array_equal(drumsIso, drumsIsoRef))
    assert(np.array_equal(pianoIso, pianoIsoRef))

    print("\nMusic Separation Test Finished\n")



def main():
    print("\n\n----------Music Separation Starting----------\n\n")

    musicSeparationTest()

    useTrainingMixed = True

    mLDataOutputsFolder = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/ML_Experiments/ML_Data/" + \
        "Exp_4/drums-piano_4_duration-500ms/Exp_4_7/"

    separationConfigOutputs = SeparationConfig.loadConfig(mLDataOutputsFolder)

    if useTrainingMixed:
        musicPath = separationConfigOutputs.mLDataInputsFolder
        musicFile = "training_mixed.wav"
    else:
        musicPath = \
            "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
            "Music_Separation_Test_Dataset/"
        # musicFile = "holyWars.wav"
        # musicFile = "ironMan.wav"
        # musicFile = "drums_piano_duet_ds.wav"
        # musicFile = "drums_piano_mixed_2.wav"
        musicFile = "drums_piano_mixed.wav"

    # musicSeparation(separationConfigOutputs, musicPath, musicFile)

    print("\n\n----------Music Separation Finished----------\n\n")

if __name__ == "__main__": main()