import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

'''
File: audioHandler.py
Brief: Handle basic audio file .wav functions including:
         - Reading and writing to a file
         - Converting to 16 bit
         - Converting to mono
         - Mixing audio sources and scaling to 16 bit
'''
class AudioHandler:

    # Get the path for the spectrogram test audio files
    @staticmethod
    def getAudioPath(audioName):
        audioPath  = "/Users/keeganjebb/Documents/Programming_2/" + \
                     "Metal_Mixer/Music_Files/Spec_Tests/"
        return audioPath + str(audioName) + '_gstreamer_ds.wav'


    @staticmethod
    def getWavFile(path, sampleRateReq = None):
        sampleRate, audio = wavfile.read(path)

        if sampleRateReq != None:
            audio = AudioHandler.getResampledAudioLinear(audio,
                                                         sampleRate,
                                                         sampleRateReq)
            sampleRate = sampleRateReq

        if audio.dtype == np.float32:
            audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        elif audio.dtype == np.int32:
            print("Cannot read 24 bit .wav files")
            assert(0)
        assert(audio.dtype == np.int16)

        return (sampleRate, audio)


    @staticmethod
    def getResampledAudioLinear(audio, sampleRate, sampleRateReq):
        sampleRate = float(sampleRate)
        sampleRateReq = float(sampleRateReq)
        assert(sampleRateReq <= sampleRate)

        if sampleRateReq < sampleRate:
            resampleFactor = sampleRate / sampleRateReq
            assert(resampleFactor.is_integer())
            audio = audio[::int(resampleFactor)]

        return audio


    @staticmethod
    def getAudioSegment(startSample, numSamples, audio):
        segmentEnd = startSample + numSamples
        segment = audio[startSample : segmentEnd]
        return segment


    @staticmethod
    def convertToMono(audio):
        if audio.ndim == 1:
            return audio
        audioMono = np.sum(audio, axis=1) / 2
        return audioMono


    @staticmethod
    def mixAudio(audioList):
        mixedAudio = np.zeros(audioList[0].shape)
        for audioSrc in audioList:
            assert(audioSrc.shape == mixedAudio.shape)
            mixedAudio += audioSrc

        scaleFactor = \
            AudioHandler.getBoundScaleFactor(np.iinfo(np.int16), mixedAudio)

        mixedAudio *= scaleFactor

        return (mixedAudio, scaleFactor)


    @staticmethod
    def plotAudio(numSamples, audio):
        plotSegment = audio[:numSamples]

        if audio.ndim == 2:
            plotSegmentChan1 = plotSegment[:, 0]
            plotSegmentChan2 = plotSegment[:, 1]
            plotSegmentMono = AudioHandler.convertToMono(plotSegment)

            assert(plotSegmentChan1.ndim == 1)
            assert(plotSegmentChan2.ndim == 1)
            assert(plotSegmentMono.ndim == 1)

            plt.plot(plotSegmentChan1)
            plt.plot(plotSegmentChan2)
            plt.plot(plotSegmentMono)

        else:
            plt.plot(plotSegment)

        plt.show()


    @staticmethod
    def getBoundScaleFactor(typeInfo, audio):
        maxBound = typeInfo.max
        minBound = typeInfo.min
        if np.max(audio) > maxBound or np.min(audio) < minBound:
            maxAudio = np.max(np.abs(audio))
            # Fit to the max positive value to be safe/simple
            scaleFactor = (maxBound / maxAudio)
        else:
            scaleFactor = 1.
        return scaleFactor


    @staticmethod
    def fitAmpBounds(typeInfo, audio):
        scaleFactor = AudioHandler.getBoundScaleFactor(typeInfo, audio)
        audio *= scaleFactor


    @staticmethod
    def writeWaveFile(path, sampleRate, audio):
        AudioHandler.fitAmpBounds(np.iinfo(np.int16), audio)
        wavfile.write(path,
                      sampleRate,
                      audio.astype(np.int16))
