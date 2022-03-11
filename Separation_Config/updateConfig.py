from separationConfig import SeparationConfig

'''
File: updateConfig.py
Brief: Tool to update the data in a SeparationConfig instance if necessary.
       For example, the mLDataFolder naming convention changes.
'''

def updateConfig():

    # mLDataFolder = "/Users/keeganjebb/Documents/Programming_2/" + \
    #             "Metal_Mixer/ML_Experiments/ML_Data/" + \
    #             "drums-piano_drumsIso_32kHz_200ms_fullMix_std_bin_500ex/"

    mLDataFolder = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Code/" + \
        "Data_Handling/Create_ML_Data_Test/Test_Data_Ref/"

    # mLDataFolder = \
    #     "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Code/" + \
    #     "Machine_Learning/Train_NN_Test/Test_Data_Ref/"

    separationConfig = SeparationConfig.loadConfig(mLDataFolder)
    
    # Note: To update mLDataFolder name, uncomment lines in
    # SeparationConfig.loadConfig
    # print(separationConfig.mLDataFolder)

    # separationConfig.genIn.numWindowSamples = 2 ** 10

    # separationConfig.genIn.logFreqFactorDs = None
    # separationConfig.genIn.spectrogramInputs.logFreqFactorDs = None
    # separationConfig.genIn.spectrogramInputsMask.logFreqFactorDs = None

    # separationConfig.saveConfig(mLDataFolder)


def main():
    print("\n\n----------Start Create ML Data Script----------\n\n")

    updateConfig()

    print("\n\n----------Create ML Data Script Finished----------\n\n")
if __name__ == "__main__": main()