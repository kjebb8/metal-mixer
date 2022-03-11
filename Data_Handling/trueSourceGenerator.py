import numpy as np
from audioHandler import AudioHandler

'''
File: trueSourceGenerator.py
Brief: Generate true source audio for all the sources used for machine learning.
       These true sources will be used as the inputs to qualitatively and
       quantitatively test the separation performance of the trained ML model.

       Input:  - Instrument list giving all the instruments being considered.
               - Mixed instrument list from the InstrumentMixer where each
                 element in the list has the audio from one or more instruments
                 to be mixed together for a single example.
       Output: - A dictionary with a vector of samples for each source (i.e.
                 each instrument in the instrument list) and a mixed source.

       Note: The wav file can be reshaped so that each column is the sound
             for one example of the source.

             If an instrument is not present in an example, all zeros are used
             for the true source such that all sources have wav files where the
             samples align with the examples.
             {
                 src1 : [Ex1 data | Ex2 zero | Ex3 data | ...]
                 src2 : [Ex1 zero | Ex2 data | Ex3 data | ...]
                 src2 : [Ex1 zero | Ex2 zero | Ex3 data | ...]
             }
'''

class TrueSourceGenerator:

    @staticmethod
    def getTrueSources(instrumentList, mixedInstrumentsList):

        numSamples = mixedInstrumentsList[0].values()[0].shape[0]
        numExamples = len(mixedInstrumentsList)
        numTotalSamples = numSamples * numExamples
        numScaledExamples = 0

        # Initialize the dictionary and include a "mixed" key for mixed audio
        trueSourcesDict = {"mixed" : np.zeros((numSamples, numExamples))}

        # Initialize the structure to hold each example in a column
        for inst in instrumentList:
            trueSourcesDict[inst] = np.zeros((numSamples, numExamples))

        # Fill in the data by looping over each example and each instrument
        for idx in range(numExamples):
            mixedInstrumentsDict = mixedInstrumentsList[idx]

            # Get the mixed audio and scale factor
            (mixedAudio, scaleFactor) = \
                AudioHandler.mixAudio(mixedInstrumentsDict.values())
            if scaleFactor != 1.0:
                numScaledExamples += 1

            # # Add the mixed audio to the dictionary
            trueSourcesDict["mixed"][:, idx] = mixedAudio

            # If an instrument is not in the example, the zeros remain
            for inst in mixedInstrumentsDict:
                sourceAudio = mixedInstrumentsDict[inst] * scaleFactor
                trueSourcesDict[inst][:, idx] = sourceAudio

        # Reshape the data into a vector
        for inst in trueSourcesDict:
            trueSourcesDict[inst] = trueSourcesDict[inst].reshape(-1,order='F')
            assert(len(trueSourcesDict[inst]) == numTotalSamples)

        print("Number of true source samples per instrument: " + \
              str(numTotalSamples))
        print("Number of examples that reqired scaling: " + \
              str(numScaledExamples))

        return trueSourcesDict