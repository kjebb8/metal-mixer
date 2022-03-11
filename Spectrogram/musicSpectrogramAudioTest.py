import copy
import numpy as np
import spectrogramUtils as specUtils
from ..Data_Handling.audioHandler import AudioHandler
from audioSpectrograms import AudioSpectrograms
from spectrogramInputs import SpectrogramInputs

'''
File: musicSpectrogramAudioTest.py
Brief: Script to take input music, perform any spectrogram operations on the
       music, then write the time data back to a file for comparison to the
       original.
'''

# Plot spectrograms of music for visualization
def musicAudioTest1():
    print("\nMusic Audio Test 1 Start\n")

    audioFile = 'holyWars'
    downSampleFactor = 8

    sampleRateReq = 2 ** 15
    numWindowSamples = 2 ** 10
    logFreqFactorDs = downSampleFactor
    audioTestFile = './Music_Files/Spec_Tests/' + 'audioTest_log_ds_' + \
                     str(downSampleFactor) + 'x_' + audioFile + '.wav'

    # sampleRateReq = 2 ** 15 / downSampleFactor
    # numWindowSamples = 2 ** 10 / downSampleFactor
    # logFreqFactorDs = 1
    # audioTestFile = './Music_Files/Spec_Tests/' + 'audioTest_' + \
    #                  str(int(sampleRateReq / 1e3)) + 'kHz_' + audioFile + '.wav'

    sampleRate, audioData = \
        AudioHandler.getWavFile(AudioHandler.getAudioPath(audioFile),
                                sampleRateReq)
    numAudioSamples = len(audioData)
    targetTotalTime = 10.
    targetSpectrogramTimeSec = 5.

    targetTotalSamples = \
        min(specUtils.getSamplesfromTime(sampleRate, targetTotalTime),
            numAudioSamples)
    targetSpectrogramSamples = \
        min(specUtils.getSamplesfromTime(sampleRate, targetSpectrogramTimeSec),
            targetTotalSamples)

    audioTestData = AudioHandler.convertToMono(\
        audioData[targetTotalSamples * 0 : targetTotalSamples * 1])

    spectrogramInputs = SpectrogramInputs()
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.numWindowSamples = numWindowSamples

    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          audioTestData)

    spectrogramInputsProc = copy.deepcopy(spectrogramInputs)
    spectrogramInputsProc.logFreqFactorDs = logFreqFactorDs

    audioSpectrogramsProc = AudioSpectrograms(spectrogramInputsProc,
                                              targetSpectrogramSamples,
                                              audioTestData)

    startTime = 0
    for idx in range(len(audioSpectrograms.spectrogramList)):
        spec = audioSpectrograms.spectrogramList[idx]
        specProc = audioSpectrogramsProc.spectrogramList[idx]
        if startTime == 0:
            print("Original:")
            spec.printSpectrogramParams()
            print("Processed:")
            specProc.printSpectrogramParams()

        endTime = startTime + targetSpectrogramTimeSec
        print("Start: " + str(startTime) + "\tEnd: " + str(endTime))
        startTime = endTime

        print("Original:")
        spec.plotSpectrogram()

        print("Processed:")
        specProc.plotSpectrogram()

    regeneratedTimeSignal = audioSpectrogramsProc.getTimeRepresentation()
    print("Saving test audio file to " + audioTestFile)
    # AudioHandler.writeWaveFile(audioTestFile,
    #                            sampleRate,
    #                            regeneratedTimeSignal)

    print("\nMusic Audio Test 1 Finish\n")


def main():
    print("\n\n-----Music Audio Test Starting-----\n\n")

    musicAudioTest1()

    print("\n\n-----Music Audio Test Finished-----\n\n")

if __name__ == "__main__": main()
