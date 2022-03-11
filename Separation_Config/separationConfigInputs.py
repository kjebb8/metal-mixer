from separationConfig import SeparationConfig

'''
File: separationConfigInputs.py
Brief: Class to hold the all the configuration data for the input of a machine
       learning music separation experiment.
'''

class SeparationConfigInputs(SeparationConfig):

    def __init__(self, genIn, parentFolder):
        self.genIn = genIn
        self.parentFolder = parentFolder
        self.configFileName = SeparationConfigInputs.configFileName
