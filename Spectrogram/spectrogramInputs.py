import constants as const

'''
File: spectrogramInputs.py
Brief: Class data structure containing the input parameters needed to create
       a specific type of spectrogram.
'''
class SpectrogramInputs:
    def __init__(self, isCompositeSpectrogram = False):
        self.sampleRate = 2 ** 15
        self.overlap = 0.5
        self.correctPowerForMusic = False  # Equal loudness + lower power limit
        self.correctEqualLoudness = False
        self.correctLowerPowerLimit = False
        self.logFreqFactorDs = None

        if isCompositeSpectrogram:
            self.setOptCompositeSpectrogamInputs()
        else:
            self.numWindowSamples = 2 ** 10
            self.segmentStartFreq = None

    def setOptCompositeSpectrogamInputs(self):
        self.numWindowSamples = const.optWindowSamples32kHz
        self.segmentStartFreq = const.optSegmentStartFreq32kHz


    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.__dict__ == other.__dict__)


    def __str__(self):
        printStr = "\n"
        for par in vars(self).items():
            printStr += "\t" + par[0] + ": " + str(par[1])
            printStr += "\n"
        return printStr