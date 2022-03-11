'''
File: mLConfig.py
Brief: Class to hold the all the configuration data for machine learning and
       training a model.
'''

class MLConfig:
    def __init__(self):
        self.layerDims = None
        self.numHiddenLayers = 0
        self.alpha = None
        self.lambd = None
        self.numIters = None
        self.miniBatchSize = None
        self.adamOn = None
        self.plotCostInterval = None
    
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