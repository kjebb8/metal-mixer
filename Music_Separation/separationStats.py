import os
import cPickle as pickle
import numpy as np
from mir_eval.separation import bss_eval_sources

'''
File: separationStats.py
Brief: Class to hold the separation statistics using a dictionary member.

    separationStatsDict =
    {
       musicFile :
       {
           instrumentSrc1 : {
               "sir"   : np.array([])
               "sar"   : np.array([])
               "sdr"   : np.array([])
               "sdrMix : np.array([])
           }
           instrumentSrc2 : {
               "sir"   : np.array([])
               "sar"   : np.array([])
               "sdr"   : np.array([])
               "sdrMix : np.array([])
           }
           ...
       }
    }

       instrumentSrc1 = "drums" for example.
'''

class SeparationStats:

    statsList = ["sir", "sar", "sdr", "sdrMix"]

    def __init__(self):
        self.separationStatsDict = {}


    def calculateStats(self, musicFile, instrumentList, truthTimeData,
                       isoTimeData, mixedTimeData):
        stats = {}
        (stats["sdr"], stats["sir"], stats["sar"], _) = \
            bss_eval_sources(truthTimeData, isoTimeData, False)

        (stats["sdrMix"], _, _, _) = \
            bss_eval_sources(truthTimeData, mixedTimeData, False)

        self.addStats(musicFile, instrumentList, stats)


    def addStats(self, musicFile, instrumentList, stats):
        if musicFile not in self.separationStatsDict:
            self.separationStatsDict[musicFile] = {}

        musicStatsDict = self.separationStatsDict[musicFile]

        for instIdx in range(len(instrumentList)):
            inst = instrumentList[instIdx]

            if inst not in musicStatsDict:
                musicStatsDict[inst] = {}
                for stat in SeparationStats.statsList:
                    musicStatsDict[inst][stat] = np.array([])

            for stat in SeparationStats.statsList:
                musicStatsDict[inst][stat] = \
                    np.append(musicStatsDict[inst][stat], stats[stat][instIdx])


    def printMeanStats(self):
        for musicFile in self.separationStatsDict:
            print("\n")
            print(musicFile + ":")
            for inst in self.separationStatsDict[musicFile]:
                print("\t" + inst + " stats:")
                for stat in SeparationStats.statsList:
                    statVec = self.separationStatsDict[musicFile][inst][stat]
                    mean = np.mean(statVec)
                    print("\t\t mean " + stat + ": " + "\t" + str(mean))
                    # print(statVec)


    def saveStats(self, statsPath):
        # Overwrites any existing file.
        with open(statsPath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
    
        if self.separationStatsDict.keys() != other.separationStatsDict.keys():
            return False

        for musicFile in self.separationStatsDict:

            if self.separationStatsDict[musicFile].keys() != \
               other.separationStatsDict[musicFile].keys():
                return False

            for inst in self.separationStatsDict[musicFile]:
                for stat in SeparationStats.statsList:
                    selfStat = self.separationStatsDict[musicFile][inst][stat]
                    otherStat = other.separationStatsDict[musicFile][inst][stat]
                    if not np.array_equal(selfStat, otherStat):
                        return False

        return True


    @staticmethod
    def loadStats(statsPath):
        if os.path.exists(statsPath):
            with open(statsPath, 'rb') as input:
                separationStats = pickle.load(input)
            return separationStats
        else:
            return None