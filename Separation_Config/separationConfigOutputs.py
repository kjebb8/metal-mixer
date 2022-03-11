from separationConfig import SeparationConfig

'''
File: separationConfigOutputs.py
Brief: Class to hold the all the configuration data for the output of a machine
       learning music separation experiment.
'''

class SeparationConfigOutputs(SeparationConfig):

    def __init__(self, mLConfig, mLDataInputsFolder, parentFolder):
        self.mLConfig = mLConfig
        self.mLDataInputsFolder = mLDataInputsFolder
        self.parentFolder = parentFolder
        self.configFileName = SeparationConfigOutputs.configFileName