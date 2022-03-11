import numpy as np
from enum import Enum

'''
File: instrumentMixer.py
Brief: Take a dictionary where data is divided by instrument and mix segments
       of different instrument together to ceate new time domain audio with
       multiple instruments.

       The output is a List where each element is a dict with 1 or more
       instruments, reperesenting one Spectrogram worth of time data in total.

       The instrument mixer has several different options for mixing algorithms.
'''

class MixingFunctions(Enum):
    randomGroups = "randomGroups"
    fractionUnmixed = "fractionUnmixed"
    fullMix = "fullMix"

class InstrumentMixer:

    # API for the Instrument Mixer. Use this function only externally.
    @staticmethod
    def getMixedInstrumentsList(instrumentDataDict,
                                mixingFunction,
                                numExamples = None,
                                groupSize = None,
                                fractionUnmixed = None):

        if mixingFunction == MixingFunctions.randomGroups:
            assert(groupSize != None)
            mixedInstrumentsList = \
                InstrumentMixer.getMixedRandomGroups(instrumentDataDict,
                                                     groupSize)
        elif mixingFunction == MixingFunctions.fractionUnmixed:
            assert(fractionUnmixed != None)
            mixedInstrumentsList = \
                InstrumentMixer.getFractionalUnmixed(instrumentDataDict,
                                                     fractionUnmixed)
        elif mixingFunction == MixingFunctions.fullMix:
            mixedInstrumentsList = \
                InstrumentMixer.getFullMix(instrumentDataDict)
        else:
            print("Invalid mixing function input")
            assert(False)

        if numExamples != None:
            mixedInstrumentsList = mixedInstrumentsList[:numExamples]

        return mixedInstrumentsList


    # Randomly match segments together into groups of a certain size
    @staticmethod
    def getMixedRandomGroups(instrumentDataDict, groupSize = 2):

        mixedInstrumentsList = []

        # 1. Establish which instruments correspond to which sections
        instrumentSectionsDict = {}
        startSection = 0
        endSection = 0
        totalSections = 0

        for instrument in instrumentDataDict:

            numSections = instrumentDataDict[instrument].shape[1]
            endSection = startSection + numSections

            instrumentSectionsDict[instrument] = {}
            instrumentSectionsDict[instrument]["start"] = startSection
            instrumentSectionsDict[instrument]["end"] = endSection

            startSection = endSection
            totalSections += numSections
        # print(instrumentSectionsDict)

        # 2. Generate a random vector of sections by shuffling
        # Make number of sections even
        totalSections = totalSections  - (totalSections % groupSize)
        randomGroups = np.arange(totalSections)
        np.random.shuffle(randomGroups)
        randomGroups = randomGroups.reshape(totalSections / groupSize,
                                            groupSize)
        # print(randomGroups)

        # 3. Populate the mixedInstrumentsList with the random groups
        numSumGroups = 0
        for group in randomGroups:
            mixedInstrumentsDict = {}
            summed = False
            for val in group:
                instrumentKey = None
                section = None
                for instrument in instrumentSectionsDict:
                    startSection = instrumentSectionsDict[instrument]["start"]
                    endSection = instrumentSectionsDict[instrument]["end"]
                    if val >= startSection and val < endSection:
                       instrumentKey = instrument
                       section = val - startSection
                       break

                instrumentData = instrumentDataDict[instrumentKey][:, section]
                if instrumentKey in mixedInstrumentsDict:
                    if summed == False:
                        numSumGroups += 1
                        summed = True
                    mixedInstrumentsDict[instrumentKey] += instrumentData
                else:
                    mixedInstrumentsDict[instrumentKey] = instrumentData

            mixedInstrumentsList.append(mixedInstrumentsDict)

        totalGroups = totalSections / groupSize
        numMixGroups = totalGroups - numSumGroups
        print("Number of groups with summed instruments = " + \
              str(numSumGroups))
        print("Number of groups with only mixed instruments = " + \
               str(numMixGroups))

        return mixedInstrumentsList


    # Keep a percentage of each instrument separated and unmixed. Mix the rest
    # of the instrument sections together in groups with each instrument in each
    # group until there are no more sections left. Each section gets used
    # exactly once.
    @staticmethod
    def getFractionalUnmixed(instrumentDataDict, fractionUnmixed):

        mixedInstrumentsList = []

        # 1. Randomly shuffle the data and separate into mixed and unmixed
        dataUnmixedDict = {}
        dataToMixDict = {}
        maxSectionsToMix = 0

        for instrument in instrumentDataDict:
            instData = instrumentDataDict[instrument]
            # Shuffle randomizes the first axis (rows) so use transpose to
            # shuffle the audio sections in the columns.
            np.random.shuffle(instData.T)

            numSections = instData.shape[1]
            numSectionsUnmixed = int(np.floor(numSections * fractionUnmixed))
            dataUnmixedDict[instrument] = instData[:, :numSectionsUnmixed]
            dataToMixDict[instrument] = instData[:, numSectionsUnmixed:]

            maxSectionsToMix = max(maxSectionsToMix,
                                   dataToMixDict[instrument].shape[1])
            print("Shape of unmixed " + instrument + " dictionary " + \
                  str(dataUnmixedDict[instrument].shape))
            print("Shape of available mixing " + instrument + " dictionary " + \
                  str(dataToMixDict[instrument].shape))
        # print(maxSectionsToMix)

        # 2. Add the unmixed data to the mixedInstrumentsList
        for instrument in dataUnmixedDict:
            for section in range(dataUnmixedDict[instrument].shape[1]):
                mixedInstrumentsList.append( \
                    {instrument : dataUnmixedDict[instrument][:, section]})

        # 3. Loop over the maxSectionsToMix and fill in the mixed portion of
        #    mixedInstrumentsList with as many mixed groups as possible
        for section in range(maxSectionsToMix):
            mixedInstrumentsDict = {}
            for instrument in dataToMixDict:
                if section < dataToMixDict[instrument].shape[1]:
                    mixedInstrumentsDict[instrument] = \
                        dataToMixDict[instrument][:, section]
            mixedInstrumentsList.append(mixedInstrumentsDict)

        # 4. Shuffle the final list
        np.random.shuffle(mixedInstrumentsList)

        return mixedInstrumentsList


    # Take the instrument with the fewest sections and that number of sections
    # will be unmixed for all instruments. Then mix all sections
    # together with each instrument in each group. The mixing continues until
    # all the sections for the instrument with the most sections has been
    # included in one mixed group. The instruments with fewer sections start
    # mixing from the beginning when their last section is mixed. Each section
    # gets used at least once.
    @staticmethod
    def getFullMix(instrumentDataDict):

        mixedInstrumentsList = []

        # 1. Randomly shuffle the data and get the sections to be mixed+unmixed
        dataUnmixedDict = {}
        numSectionsUnmixed = None
        numSectionsToMix = 0

        for instrument in instrumentDataDict:
            instData = instrumentDataDict[instrument]
            # Shuffle randomizes the first axis (rows) so use transpose to
            # shuffle the audio sections in the columns.
            np.random.shuffle(instData.T)

            numSections = instData.shape[1]
            if numSectionsUnmixed == None:
                numSectionsUnmixed = numSections
            else:
                numSectionsUnmixed = np.minimum(numSectionsUnmixed, numSections)
            numSectionsToMix = np.maximum(numSectionsToMix, numSections)

            print("Shape of " + instrument + " dictionary: " + \
                  str(instData.shape))

        print("Number of unmixed sections: " + str(numSectionsUnmixed))
        print("Number of mixed sections: " + str(numSectionsToMix))

        # 2. Add the unmixed data to the mixedInstrumentsList
        for instrument in instrumentDataDict:
            for section in range(numSectionsUnmixed):
                mixedInstrumentsList.append( \
                    {instrument : instrumentDataDict[instrument][:, section]})

        # 3. Loop over the numSectionsToMix and fill in the mixed portion of
        #    mixedInstrumentsList with as many mixed groups as possible.
        #    Use mod operator to start at the beginning again if no more new
        #    sections for an instrument.
        for section in range(numSectionsToMix):
            mixedInstrumentsDict = {}
            for instrument in instrumentDataDict:
                sectionToMix = section % instrumentDataDict[instrument].shape[1]
                mixedInstrumentsDict[instrument] = \
                    instrumentDataDict[instrument][:, sectionToMix]
            mixedInstrumentsList.append(mixedInstrumentsDict)

        # 4. Shuffle the final list
        np.random.shuffle(mixedInstrumentsList)

        print("Total number of elements in mixedInstrumentsList: " + \
              str(len(mixedInstrumentsList)))

        return mixedInstrumentsList
