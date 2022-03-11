import numpy as np

'''
File: constants.py
Brief: Constant values for the Spectrogram Classes/Functions/Files.
'''

# Minimum spectrogram magnitude to avoid taking log(0) to get power Db
# Equivalent to 0 dB, which is not
minMagnitudeValue = 1

# A minimum power value for spectrograms. Optional to implement.
minPowerDbValue = 0

# The smallest value to divide by when correcting the ends of a regenerated time
# signal to account for the half hanning window.
windowCorrectionCutoff = 0.3

# Manually computed "optimal" values for composite spectrograms
optWindowSamples32kHz = 2 ** (np.arange(6) + 7)
optSegmentStartFreq32kHz = np.array([0, 506, 1525, 4005, 8005, 12170])
