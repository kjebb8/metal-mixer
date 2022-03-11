import cPickle as pickle

'''
File: separationConfig.py
Brief: Class to hold the all the configuration data for a machine learning
       music separation experiment. The config is to be shared across the
       other ML modules.
'''

class SeparationConfig:

    # All config files use default name
    configFileName = "separationConfig.pkl"

    def saveConfig(self, parentFolder = None):
        if parentFolder:
            configPath = parentFolder + self.configFileName
        else:
            configPath = self.parentFolder + self.configFileName
        # Overwrites any existing file.
        with open(configPath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.__dict__ == other.__dict__)


    def __str__(self):
        print("\nSeparation Config:\n")
        printStr = ""
        for par in vars(self).items():
            printStr += par[0] + ": " + str(par[1])
            printStr += "\n"
        return printStr


    @staticmethod
    def loadConfig(parentFolder):
        configPath = parentFolder + SeparationConfig.configFileName
        with open(configPath, 'rb') as input:
            separationConfig = pickle.load(input)

        if separationConfig.parentFolder != parentFolder:
            print("Warning: parentFolder name does not match.\n" + \
                  separationConfig.parentFolder + " in config\n" +\
                  parentFolder + " given")
        #     separationConfig.parentFolder = parentFolder
        #     separationConfig.saveConfig()

        return separationConfig