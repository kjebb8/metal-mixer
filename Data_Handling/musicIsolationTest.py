import copy
import numpy as np
from audioHandler import AudioHandler
from instrumentDataParser import InstrumentDataParser
from spectrogramMLDataGenerator import SpectrogramMLDataGenerator
from ..Spectrogram import spectrogramUtils as specUtils
from ..Spectrogram.spectrogramInputs import SpectrogramInputs
from ..Spectrogram.spectrogramParams import SpectrogramParams
from ..Spectrogram.spectrogram import Spectrogram
from ..Spectrogram.audioSpectrograms import AudioSpectrograms

'''
File: musicIsolationTest.py
Brief: Example eye and ear test for applying ideal masks to mixed time data.
       Sets an example of the best possible separation quality from masks.

       Some distortions are seen after applying ideal masks, likely due to the
       unchanged phase components which are applied to the masked Spectrogram to
       get the time domain representation (ear test).
'''
def musicIsoTest1():
    print("\nMusic Isolation Test 1 Start\n")
    instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Music_Iso_Test_2/"

    duration = 0.6 # seconds
    isoInstrumentList = ["piano", "drums"]

    spectrogramInputs = SpectrogramInputs()
    # spectrogramInputs.correctLowerPowerLimit = True
    sampleRate = spectrogramInputs.sampleRate

    targetSpectrogramSamples = specUtils.getSamplesfromTime(sampleRate,
                                                            duration)
    numSpectrogramSamples = \
        SpectrogramParams.getNumSpectrogramSamples( \
            spectrogramInputs.numWindowSamples,
            targetSpectrogramSamples,
            spectrogramInputs.overlap)

    spectrogramParams = SpectrogramParams(spectrogramInputs,
                                          numSpectrogramSamples)
    numSpectrogramValues = spectrogramParams.numSpectrogramValues

    instrumentDataDict = \
        InstrumentDataParser.getInstrumentDataDict(instrumentDataPath,
                                                   sampleRate,
                                                   numSpectrogramSamples)

    assert(instrumentDataDict["piano"].shape == \
           instrumentDataDict["drums"].shape)

    numSpectrograms = instrumentDataDict["piano"].shape[1]

    # Mix the data together 1 to 1
    mixedInstrumentsList = []
    for specIdx in range(numSpectrograms):
        mixedInstrumentsDict = \
            {"piano" : instrumentDataDict["piano"][:, specIdx],
             "drums" : instrumentDataDict["drums"][: , specIdx]}
        mixedInstrumentsList.append(mixedInstrumentsDict)

    (_, idealMasks) = SpectrogramMLDataGenerator.getSpectrogramMLData(\
        mixedInstrumentsList,
        isoInstrumentList,
        spectrogramInputs,
        numSpectrogramValues)

    mixedTimeData = (instrumentDataDict["piano"] + instrumentDataDict["drums"])
    mixedTimeVec = mixedTimeData.reshape(-1, order='F')
    # AudioHandler.writeWaveFile(instrumentDataPath + "mixed.wav",
    #                            sampleRate,
    #                            mixedTimeVec)

    audioSpectrograms = AudioSpectrograms(spectrogramInputs,
                                          targetSpectrogramSamples,
                                          mixedTimeVec)

    assert(numSpectrograms == audioSpectrograms.numSpectrograms)

    numSpectrogramValues = \
        audioSpectrograms.spectrogramList[0].params.numSpectrogramValues

    pianoSpectrograms = copy.deepcopy(audioSpectrograms)
    pianoMasks = idealMasks[:numSpectrogramValues, :]
    pianoSpectrograms.applyPowerDbMasks(pianoMasks)

    drumSpectrograms = copy.deepcopy(audioSpectrograms)
    drumMasks = idealMasks[numSpectrogramValues:, :]
    drumSpectrograms.applyPowerDbMasks(drumMasks)

    # Write time domain iso data to file
    # pianoTimeDataIsolated = pianoSpectrograms.getTimeRepresentation()
    # drumTimeDataIsolated = drumSpectrograms.getTimeRepresentation()
    # AudioHandler.writeWaveFile(instrumentDataPath + "pianoIsoBinThresh.wav",
    #                            sampleRate,
    #                            pianoTimeDataIsolated)
    # AudioHandler.writeWaveFile(instrumentDataPath + "drumIsoBinThresh.wav",
    #                            sampleRate,
    #                            drumTimeDataIsolated)

    # Plot the original spectrograms vs the isolated spectrograms
    # mixedSpec0 = Spectrogram(spectrogramInputs, mixedTimeData[:, 1])
    # pianoSpecOrig = Spectrogram(spectrogramInputs, instrumentDataDict["piano"][:, 1])
    # drumSpecOrig = Spectrogram(spectrogramInputs, instrumentDataDict["drums"][:, 1])
    #
    # mixedSpec0.plotSpectrogram()
    # audioSpectrograms.spectrogramList[1].plotSpectrogram()
    # pianoSpecOrig.plotSpectrogram()
    # pianoSpectrograms.spectrogramList[1].plotSpectrogram()
    # drumSpecOrig.plotSpectrogram()
    # drumSpectrograms.spectrogramList[1].plotSpectrogram()

    # Plot the masks as spectrograms
    # pianoSpecOrig.setPowerDbFromVec(idealMasks[:numSpectrogramValues, 1])
    # pianoSpecOrig.plotSpectrogram()
    # drumSpecOrig.setPowerDbFromVec(idealMasks[numSpectrogramValues:, 1])
    # drumSpecOrig.plotSpectrogram()

    # Spectrogram of iso instruments after going back to time domain
    # pianoIsoSpec = Spectrogram(spectrogramInputs, pianoTimeDataIsolated)
    # drumIsoSpec = Spectrogram(spectrogramInputs, drumTimeDataIsolated)
    # pianoSpec = Spectrogram(spectrogramInputs, instrumentDataDict["piano"].reshape(-1, order='F'))
    # drumSpec = Spectrogram(spectrogramInputs, instrumentDataDict["drums"].reshape(-1, order='F'))
    # pianoSpec.plotSpectrogram()
    # pianoIsoSpec.plotSpectrogram()
    # drumSpec.plotSpectrogram()
    # drumIsoSpec.plotSpectrogram()


    print("\nMusic Isolation Test 1 Finish\n")

def main():
    print("\n\n-----Music Isolation Test Starting-----\n\n")

    musicIsoTest1()

    print("\n\n-----Music Isolation Test Finished-----\n\n")

if __name__ == "__main__": main()
