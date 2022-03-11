import numpy as np
from audioHandler import AudioHandler
from generateMLData import GenMLDataInputs
from generateMLData import generateMLData
from instrumentMixer import InstrumentMixer
from instrumentMixer import MixingFunctions

'''
File: instrumentMixerTest.py
Brief: Test the InstrumentMixer Class functions.
'''
# Test the getMixedRandomGroups function
def instrumentMixerTest1():
    print("\nInstrument Mixer Test 1 Started\n")

    genIn = GenMLDataInputs()

    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"
    genIn.duration = 1.3  # seconds
    genIn.mixingFunction = MixingFunctions.randomGroups
    genIn.groupSize = 3
    genIn.computeSpectrograms = False
    np.random.seed(0)

    genOut = generateMLData(genIn)

    # Test the correctness of the mix random groups algorithm
    (sampleRateWav, pianoData) = \
        AudioHandler.getWavFile(genIn.instrumentDataPath + \
                                "piano_gstreamer_ds.wav")
    pianoData = AudioHandler.convertToMono(pianoData)
    (sampleRateWav, guitarData) = \
        AudioHandler.getWavFile(genIn.instrumentDataPath + \
                                "acousticGuitar_gstreamer_ds.wav")
    guitarData = AudioHandler.convertToMono(guitarData)

    # Verify case where all three instruments in the group were piano or guitar
    startGuitar = [122, 66, 142]
    startPiano = [71, 0, 14] # [238, 167, 181]
    guitarTest = np.zeros((genIn.numSpectrogramSamples,))
    pianoTest = np.zeros((genIn.numSpectrogramSamples,))
    for i in range(genIn.groupSize):
        guitarTest += \
            guitarData[startGuitar[i] * genIn.numSpectrogramSamples:\
                       (startGuitar[i] + 1) * genIn.numSpectrogramSamples]
        pianoTest += \
            pianoData[startPiano[i] * genIn.numSpectrogramSamples:\
                      (startPiano[i] + 1) * genIn.numSpectrogramSamples]
    assert(np.array_equal(genOut.mixedInstrumentsList[0]["acousticGuitar"],
                          guitarTest))
    assert(np.array_equal(genOut.mixedInstrumentsList[43]["piano"],
                          pianoTest))

    # Verify case where instruments were mixed in the group
    startMix = [39, 26, 65] # [39, 193, 314]
    mixTest = np.zeros((genIn.numSpectrogramSamples,))
    mixTest += guitarData[startMix[0] * genIn.numSpectrogramSamples:\
                          (startMix[0] + 1) * genIn.numSpectrogramSamples]
    mixTest += pianoData[startMix[1] * genIn.numSpectrogramSamples:\
                          (startMix[1] + 1) * genIn.numSpectrogramSamples]
    mixTest += pianoData[startMix[2] * genIn.numSpectrogramSamples:\
                         (startMix[2] + 1) * genIn.numSpectrogramSamples]

    assert(np.array_equal(genOut.mixedInstrumentsList[-6]["piano"] + \
                          genOut.mixedInstrumentsList[-6]["acousticGuitar"],
                          mixTest))

    print("\nInstrument Mixer Test 1 Finished\n")


# Test the getFractionalUnmixed function
def instrumentMixerTest2():
    print("\nInstrument Mixer Test 2 Started\n")

    genIn = GenMLDataInputs()

    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"
    genIn.duration = 1.2  # seconds
    genIn.mixingFunction = MixingFunctions.fractionUnmixed
    genIn.fractionUnmixed = 1. / 4
    genIn.computeSpectrograms = False
    np.random.seed(0)

    genOut = generateMLData(genIn)

    (sampleRateWav, pianoData) = \
        AudioHandler.getWavFile(genIn.instrumentDataPath + \
                                "piano_gstreamer_ds.wav")
    pianoData = AudioHandler.convertToMono(pianoData)
    (sampleRateWav, guitarData) = \
        AudioHandler.getWavFile(genIn.instrumentDataPath + \
                                "acousticGuitar_gstreamer_ds.wav")
    guitarData = AudioHandler.convertToMono(guitarData)
    # Verify the number of mixed and unmixed groups
    numSectionsPiano = \
        np.floor(len(pianoData) / genIn.numSpectrogramSamples) * 2 # 2 files
    numSectionsGuitar = \
        np.floor(len(guitarData) / genIn.numSpectrogramSamples)

    unmixedPianoSections = \
        int(np.floor(numSectionsPiano * genIn.fractionUnmixed))
    unmixedGuitarSections = \
        int(np.floor(numSectionsGuitar * genIn.fractionUnmixed))

    mixedPianoSections = numSectionsPiano - unmixedPianoSections
    mixedGuitarSections = numSectionsGuitar - unmixedGuitarSections
    mixedSections = int(min(mixedPianoSections, mixedGuitarSections))

    unmixedPianoSections += mixedPianoSections - mixedSections
    unmixedGuitarSections += mixedGuitarSections - mixedSections

    actualUnmixedPianoSections = 0
    actualUnmixedGuitarSections = 0
    actualMixedSections = 0
    for dict in genOut.mixedInstrumentsList:
        if len(dict) == 2:
            actualMixedSections += 1
        elif len(dict) == 1:
            if "piano" in dict:
                actualUnmixedPianoSections += 1
            elif "acousticGuitar" in dict:
                actualUnmixedGuitarSections += 1

    assert(actualMixedSections == mixedSections)
    assert(actualUnmixedPianoSections == unmixedPianoSections)
    assert(actualUnmixedGuitarSections == unmixedGuitarSections)

    # Test the correctness of the first value of the list
    # TestIdx were obtained by searching through the file for a 3 value sequence
    guitarTestIdx = 12
    pianoTestIdx = 62
    mixTest = np.zeros((genIn.numSpectrogramSamples,))
    mixTest += guitarData[guitarTestIdx * genIn.numSpectrogramSamples:\
                          (guitarTestIdx + 1) * genIn.numSpectrogramSamples]
    mixTest += pianoData[pianoTestIdx * genIn.numSpectrogramSamples:\
                          (pianoTestIdx + 1) * genIn.numSpectrogramSamples]


    assert(np.array_equal(genOut.mixedInstrumentsList[0]["piano"] + \
                          genOut.mixedInstrumentsList[0]["acousticGuitar"],
                          mixTest))

    print("\nInstrument Mixer Test 2 Finished\n")


# Test the getFullMix function
def instrumentMixerTest3():
    print("\nInstrument Mixer Test 3 Started\n")

    genIn = GenMLDataInputs()

    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"
    genIn.duration = 1.2  # seconds
    genIn.mixingFunction = MixingFunctions.fullMix
    genIn.computeSpectrograms = False
    np.random.seed(0)

    genOut = generateMLData(genIn)

    (sampleRateWav, pianoData) = \
        AudioHandler.getWavFile(genIn.instrumentDataPath + \
                                "piano_gstreamer_ds.wav")
    pianoData = AudioHandler.convertToMono(pianoData)
    (sampleRateWav, guitarData) = \
        AudioHandler.getWavFile(genIn.instrumentDataPath + \
                                "acousticGuitar_gstreamer_ds.wav")
    guitarData = AudioHandler.convertToMono(guitarData)

    # Verify the number of mixed and unmixed groups
    unmixedPianoSections = 180
    unmixedGuitarSections = 180
    mixedSections = 183

    actualUnmixedPianoSections = 0
    actualUnmixedGuitarSections = 0
    actualMixedSections = 0
    for dict in genOut.mixedInstrumentsList:
        if len(dict) == 2:
            actualMixedSections += 1
        elif len(dict) == 1:
            if "piano" in dict:
                actualUnmixedPianoSections += 1
            elif "acousticGuitar" in dict:
                actualUnmixedGuitarSections += 1

    assert(actualMixedSections == mixedSections)
    assert(actualUnmixedPianoSections == unmixedPianoSections)
    assert(actualUnmixedGuitarSections == unmixedGuitarSections)

    # Test the correctness of a mixed value of the list
    # TestIdx were obtained by shuffling arange matrices in the test
    guitarTestIdx = 91
    pianoTestIdx = 75
    mixTest = np.zeros((genIn.numSpectrogramSamples,))
    mixTest += guitarData[guitarTestIdx * genIn.numSpectrogramSamples:\
                          (guitarTestIdx + 1) * genIn.numSpectrogramSamples]
    mixTest += pianoData[pianoTestIdx * genIn.numSpectrogramSamples:\
                          (pianoTestIdx + 1) * genIn.numSpectrogramSamples]

    assert(np.array_equal(genOut.mixedInstrumentsList[4]["piano"] + \
                          genOut.mixedInstrumentsList[4]["acousticGuitar"],
                          mixTest))

    print("\nInstrument Mixer Test 3 Finished\n")


# Test the numExamples parameter
def instrumentMixerTest4():
    print("\nInstrument Mixer Test 4 Started\n")

    genIn = GenMLDataInputs()

    genIn.instrumentDataPath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Data_Handling_Test/"
    genIn.duration = 1.2  # seconds
    genIn.mixingFunction = MixingFunctions.fullMix
    genIn.computeSpectrograms = False
    genIn.numExamples = 110
    np.random.seed(0)

    genOut = generateMLData(genIn)

    assert(len(genOut.mixedInstrumentsList) == genIn.numExamples)

    print("\nInstrument Mixer Test 4 Finished\n")


def main():
    print("\n\n----------Instrument Mixer Test Starting----------\n\n")

    instrumentMixerTest1()
    instrumentMixerTest2()
    instrumentMixerTest3()
    instrumentMixerTest4()

    print("\n\n----------Instrument Mixer Test Finished----------\n\n")
if __name__ == "__main__": main()
