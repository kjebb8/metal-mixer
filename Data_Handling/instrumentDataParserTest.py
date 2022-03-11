import numpy as np
from audioHandler import AudioHandler
from instrumentDataParser import InstrumentDataParser
from ..Spectrogram import spectrogramUtils as specUtils
from ..Spectrogram.spectrogramInputs import SpectrogramInputs
from ..Spectrogram.spectrogramParams import SpectrogramParams

'''
File: instrumentDataParserTest.py
Brief: Test the InstrumentDataParser Class functions.
'''
def instrumentDataParserTest1():
    instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"

    duration = 0.05  # seconds
    spectrogramInputs = SpectrogramInputs()
    sampleRate = spectrogramInputs.sampleRate

    targetSpectrogramSamples = specUtils.getSamplesfromTime(sampleRate,
                                                            duration)
    numSpectrogramSamples = \
        SpectrogramParams.getNumSpectrogramSamples( \
            spectrogramInputs.numWindowSamples,
            targetSpectrogramSamples,
            spectrogramInputs.overlap)

    instrumentDataDict = \
        InstrumentDataParser.getInstrumentDataDict(instrumentDataPath,
                                                   sampleRate,
                                                   numSpectrogramSamples)

    assert(instrumentDataDict["piano"].shape == (1536, 4580))
    assert(instrumentDataDict["acousticGuitar"].shape == (1536, 4639))
    assert(not("electricGuitar" in instrumentDataDict))

    (sampleRateWav, instrumentWavTest) = \
        AudioHandler.getWavFile(instrumentDataPath + "piano_gstreamer_ds.wav")
    instrumentWavTest = AudioHandler.convertToMono(instrumentWavTest)

    assert(sampleRateWav == sampleRate)
    assert(np.array_equal(instrumentWavTest[:numSpectrogramSamples],
                          instrumentDataDict["piano"][:,0]))

def main():
    print("\n\n----------Instrument Data Parser Test Starting----------\n\n")

    instrumentDataParserTest1()

    print("\n\n----------Instrument Data Parser Test Finished----------\n\n")
if __name__ == "__main__": main()
