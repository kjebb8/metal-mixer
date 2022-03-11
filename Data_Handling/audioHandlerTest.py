import numpy as np
from ..Spectrogram import spectrogramUtils as specUtils
from audioHandler import AudioHandler

'''
File: audioHandlerTest.py
Brief: Test the AudioHandler Class functions.
'''
def getAudioTestFiles():
    print("Piano")
    sampleRatePiano, dataPiano = \
        AudioHandler.getWavFile(AudioHandler.getAudioPath('piano'))
    print("Sample Rate: " + str(sampleRatePiano))
    print("Shape: " + str(dataPiano.shape) + "\n")

    print("Guitar")
    sampleRateGuitar, dataGuitar = \
        AudioHandler.getWavFile(AudioHandler.getAudioPath('guitar'))
    print("Sample Rate: " + str(sampleRateGuitar))
    print("Shape: " + str(dataGuitar.shape) + "\n")

    assert(sampleRatePiano == sampleRateGuitar)
    sampleRate = sampleRateGuitar
    return (sampleRate, dataPiano, dataGuitar)


def audioTest1():
    print("\nAudio Test 1 Start\n")

    segmentStartSec = 10
    segmentDurationSec = 5
    plotDurationSec = 0.005

    (sampleRate, dataPiano, dataGuitar) = getAudioTestFiles()

    startSample = \
        specUtils.getSamplesfromTime(sampleRate, segmentStartSec)
    numSamples = \
        specUtils.getSamplesfromTime(sampleRate, segmentDurationSec)
    numPlotSamples = \
        specUtils.getSamplesfromTime(sampleRate, plotDurationSec)

    dataPianoSegment = AudioHandler.getAudioSegment(startSample,
                                                    numSamples,
                                                    dataPiano)

    dataGuitarSegment = AudioHandler.getAudioSegment(startSample,
                                                     numSamples,
                                                     dataGuitar)

    AudioHandler.plotAudio(numPlotSamples, dataPianoSegment)

    (combinedDataSegment, _) = \
        AudioHandler.mixAudio([dataPianoSegment, dataGuitarSegment])
    combinedDataSegmentMono = AudioHandler.convertToMono(combinedDataSegment)
    assert(combinedDataSegment.shape == (numSamples, 2))
    assert(combinedDataSegmentMono.shape == (numSamples,))

    AudioHandler.plotAudio(numPlotSamples, combinedDataSegmentMono)

    # AudioHandler.writeWaveFile( \
    #     "../../Music_Files/combined_segment_piano_guitar_mono.wav",
    #     sampleRate,
    #     combinedDataSegmentMono)
    #
    # AudioHandler.writeWaveFile( \
    #     "../../Music_Files/combined_segment_piano_guitar.wav",
    #     sampleRate,
    #     combinedDataSegment)

    print("\nAudio Test 1 Finish\n")


# Test the audio mixing functionality
def audioTest2():
    print("\nAudio Test 2 Start\n")

    segmentStartSec = 10
    segmentDurationSec = 5
    plotDurationSec = 0.005

    (sampleRate, dataPiano, dataGuitar) = getAudioTestFiles()

    startSample = \
        specUtils.getSamplesfromTime(sampleRate, segmentStartSec)
    numSamples = \
        specUtils.getSamplesfromTime(sampleRate, segmentDurationSec)
    numPlotSamples = \
        specUtils.getSamplesfromTime(sampleRate, plotDurationSec)

    pianoSegment = AudioHandler.getAudioSegment(startSample,
                                                numSamples,
                                                dataPiano)

    guitarSegment = AudioHandler.getAudioSegment(startSample,
                                                 numSamples,
                                                 dataGuitar)

    int16Max = np.iinfo(np.int16).max

    # Make the audio sources larger than 16 bit max
    audioSrcList = [pianoSegment * 4., guitarSegment * 4.]
    assert(np.max(np.abs(audioSrcList[0])) > int16Max)
    assert(np.max(np.abs(audioSrcList[1])) > int16Max)

    # Check the mixed audio fits in 16 bits but the sources are unchanged
    (mixedAudio, _) = AudioHandler.mixAudio(audioSrcList)
    assert(np.max(np.abs(mixedAudio)) == int16Max)
    assert(np.array_equal(audioSrcList[0], pianoSegment * 4.))
    assert(np.array_equal(audioSrcList[1], guitarSegment * 4.))

    # Check the fitAmpBounds function on one of the sources
    AudioHandler.fitAmpBounds(np.iinfo(np.int16), audioSrcList[0])
    assert(np.max(np.abs(audioSrcList[0])) == int16Max)

    # Get the scale factor for the updated sources
    (mixedAudio, scaleFactor) = AudioHandler.mixAudio(audioSrcList)
    assert(np.max(np.abs(np.around(mixedAudio, decimals=5))) == int16Max)
    assert(np.max(np.abs(audioSrcList[0])) == int16Max)
    assert(np.max(np.abs(audioSrcList[1])) > int16Max)

    # Scale the sources and check the sum of the new sources gives the int16 max
    scaledSrcMix = 0
    for audioSrc in audioSrcList:
        scaledSrcMix += audioSrc * scaleFactor
    assert(np.max(np.abs(np.around(scaledSrcMix, decimals=5))) == int16Max)
    assert(np.array_equal(np.around(scaledSrcMix, decimals=5),
                          np.around(mixedAudio, decimals=5)))

    print("\nAudio Test 2 Finish\n")


# Test the linear resampling function
def audioTest3():
    print("\nAudio Test 3 Start\n")

    sampleRateReq = 2 ** 13  # 8129
    audioPathOriginal = AudioHandler.getAudioPath('piano')
    audioPath8kHz = AudioHandler.getAudioPath('piano_8kHz')

    sampleRate, dataPiano = \
        AudioHandler.getWavFile(audioPathOriginal)

    sampleRateDs, dataPianoDs = \
        AudioHandler.getWavFile(audioPathOriginal, sampleRateReq)

    sampleRate8kHz, dataPiano8kHz = \
        AudioHandler.getWavFile(audioPath8kHz)

    print("sampleRate: " + str(sampleRate))
    print("sampleRateReq: " + str(sampleRateReq))
    print("sampleRateDs: " + str(sampleRateDs))
    assert(sampleRateDs != sampleRate)
    assert(sampleRateDs == sampleRateReq)
    assert(sampleRateDs == sampleRate8kHz)
    assert(dataPianoDs.shape == dataPiano8kHz.shape)

    avgError = np.sum(np.abs(dataPianoDs - dataPiano8kHz)) / dataPianoDs.size
    assert(float(avgError) / np.iinfo(np.int16).max < 0.0013)

    print("\nAudio Test 3 Finish\n")



def main():
    print("\n\n----------Audio Test Starting----------\n\n")

    audioTest1()
    audioTest2()
    audioTest3()

    print("\n\n----------Audio Test Finished----------\n\n")

if __name__ == "__main__": main()
