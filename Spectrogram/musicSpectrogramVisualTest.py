import numpy as np
import matplotlib.pyplot as plt
import spectrogramUtils as specUtils
from ..Data_Handling.audioHandler import AudioHandler
from audioSpectrograms import AudioSpectrograms
from spectrogramInputs import SpectrogramInputs

'''
File: musicSpectrogramVisualTest.py
Brief: Script to visualize audio or music in one or more spectrograms. The
       spectrograms can be formatted in any way including composite and with
       equal loudness correction.
'''

# Plot spectrograms of music for visualization
def musicVisualTest1():
    print("\nMusic Visual Test 1 Start\n")

    sampleRate, audioData = \
        AudioHandler.getWavFile(AudioHandler.getAudioPath('piano'))
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
        audioData[targetTotalSamples*0:targetTotalSamples*1])

    spectrogramInputs = SpectrogramInputs(True)
    spectrogramInputs.sampleRate = sampleRate
    spectrogramInputs.correctPowerForMusic = True

    # Composite Spectrogram
    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          audioTestData)

    startTime = 0
    for spec in audioSpectrograms.spectrogramList:
        if startTime == 0:
            spec.printSpectrogramParams()
        endTime = startTime + targetSpectrogramTimeSec
        print("Start: " + str(startTime) + "\tEnd: " + str(endTime))
        startTime = endTime
        spec.plotSpectrogram()

        # Check the percent error
        # regeneratedTimeSignal = spec.getTimeRepresentation()
        # averageError = \
        #     np.sum(np.abs(regeneratedTimeSignal - audioTestData)) / \
        #                   targetSpectrogramSamples
        # averageValue = np.sum(np.abs(audioTestData)) / targetSpectrogramSamples
        # print(averageError/averageValue)

        # AudioHandler.writeWaveFile('./Music_Files/test_gstreamer.wav',
        #                            sampleRate,
        #                            regeneratedTimeSignal[200:-200])

    # Non-composite Spectrogram
    # spectrogramInputs = SpectrogramInputs()
    # spectrogramInputs.sampleRate = sampleRate
    # spectrogramInputs.correctPowerForMusic = True
    #
    # for i in range(7, 13):
    #     print(i)
    #     numWindowSamples = 2 ** i
    #
    #     spectrogramInputs.numWindowSamples = numWindowSamples
    #
    #    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
    #                                          targetSpectrogramSamples,
    #                                          audioTestData)
    #
    #     startTime = 0
    #     for spec in audioSpectrograms.spectrogramList:
    #         endTime = startTime + targetSpectrogramTimeSec
    #         print("Start: " + str(startTime) + "\tEnd: " + str(endTime))
    #         startTime = endTime
    #         spec.plotSpectrogram()
    #
    #         maxPowerDb = np.max(spec.powerDbComponentsList[0])
    #         print("Max Power: " + str(maxPowerDb))
    #
    #         regeneratedTimeSignal = spec.getTimeRepresentation()
    #
    #         AudioHandler.writeWaveFile('../../Music_Files/' + str(i) + 'test_gstreamer.wav',
    #                                    sampleRate,
    #                                    regeneratedTimeSignal[200:-200])


    print("\nMusic Visual Test 1 Finish\n")


# Compare re-recorded music to original
def musicVisualTest2():
    print("\nMusic Visual Test 2 Start\n")
    songName = 'snare'
    songPath = AudioHandler.getAudioPath(songName)
    sampleRate, audioData = AudioHandler.getWavFile(songPath)

    songPathRec = AudioHandler.getAudioPath(songName + '_recorded')
    sampleRateRec, audioDataRec = AudioHandler.getWavFile(songPathRec)
    assert(sampleRate == sampleRateRec)

    numAudioSamples = min(len(audioData), len(audioDataRec))
    targetTotalTime = 10.

    targetTotalSamples = \
        min(specUtils.getSamplesfromTime(sampleRate, targetTotalTime),
            numAudioSamples)

    # Start where the music starts
    audioTestDataRec = \
        AudioHandler.convertToMono(audioDataRec[:targetTotalSamples])
    musicStartIdxRec = np.where(audioTestDataRec > 1000)[0][0] - 520 # Shift
    audioTestDataRec = audioTestDataRec[musicStartIdxRec:]
    totalSamples = len(audioTestDataRec)

    audioTestData = \
        AudioHandler.convertToMono(audioData[:targetTotalSamples + sampleRate])
    musicStartIdx = np.where(audioTestData > 1000)[0][0]
    audioTestData = audioTestData[musicStartIdx:totalSamples + musicStartIdx]
    assert(len(audioTestDataRec) == len(audioTestData))

    # Scale the recorded data
    audioTestDataRec *= np.max(audioTestData) / float(np.max(audioTestDataRec))

    plt.plot(audioTestData)
    plt.plot(audioTestDataRec)
    plt.show()

    spectrogramInputs = SpectrogramInputs(True)
    spectrogramInputs.sampleRate = sampleRate

    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          totalSamples,
                                          audioTestData)
    audioSpectrogramsRec = AudioSpectrograms(spectrogramInputs,
                                             totalSamples,
                                             audioTestDataRec)

    assert(len(audioSpectrograms.spectrogramList) == 1)
    assert(len(audioSpectrogramsRec.spectrogramList) == 1)

    spec = audioSpectrograms.spectrogramList[0]
    specRec = audioSpectrogramsRec.spectrogramList[0]
    print("Start: 0\tEnd: " + str(targetTotalTime))

    # Power Error
    averageError = 0.
    averageValue = 0.
    numPowerValues = 0.
    for segIdx in range(spec.params.numSegments):
        averageError += np.sum(np.abs(spec.powerDbComponentsList[segIdx] - \
                                      specRec.powerDbComponentsList[segIdx]))
        averageValue += np.sum(np.abs(spec.powerDbComponentsList[segIdx]))
        numPowerValues += spec.params.numFftWindows[segIdx] * \
                          spec.params.numSegmentFreq[segIdx]

    averageError /= numPowerValues
    averageValue /= numPowerValues
    print("Average Error = " + str(averageError))
    print("Average Value = " + str(averageValue))
    print("Power Error = " + str(averageError / averageValue))

    # Time Error
    averageError = \
        np.sum(np.abs(audioTestDataRec - audioTestData)) / \
                      float(targetTotalSamples)
    averageValue = np.sum(np.abs(audioTestData)) / float(targetTotalSamples)
    print("Average Error = " + str(averageError))
    print("Average Value = " + str(averageValue))
    print("Time Error = " + str(averageError / averageValue))

    spec.plotSpectrogram()
    specRec.plotSpectrogram()

    print("\nMusic Visual Test 2 Finish\n")


def main():
    print("\n\n-----Music Visual Test Starting-----\n\n")

    musicVisualTest1()
    # musicVisualTest2()

    print("\n\n-----Music Visual Test Finished-----\n\n")

if __name__ == "__main__": main()
