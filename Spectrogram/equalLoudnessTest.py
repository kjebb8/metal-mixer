import numpy as np
import matplotlib.pyplot as plt
import equalLoudness as eql

'''
File: equalLoudnessTest.py
Brief: Test the equalLoudness function getEqualLoudnessCorrection.
'''
def equalLoudnessTest1():
    print("\nEqual Loudness Test 1 Start\n")

    np.random.seed(0)
    testFreq = np.random.rand(5) * 100
    testFreq = np.append(testFreq, np.random.rand(5) * 900 + 100)
    testFreq = np.append(testFreq, np.random.rand(5) * 9000 + 1000)
    testFreq = np.append(testFreq, np.random.rand(5) * 6000 + 10000)
    testPowerDb = np.array([np.random.rand(len(testFreq)) * 70 + 20]).T

    correction = eql.getEqualLoudnessCorrection(testFreq)
    testPowerDb += correction

    plt.plot(eql.logFreqRange, eql.equalLoudnessCorrectionPerFreq)
    plt.scatter(np.log10(testFreq), correction, marker='x', color='r')
    plt.show()

    assert(np.array_equal(np.around(testPowerDb, decimals=4),
                          np.around(test1PowerDbRef, decimals=4)))
    assert(np.array_equal(np.around(correction, decimals=4),
                          np.around(test1CorrectionRef, decimals=4)))

    print("\nEqual Loudness Test 1 Finish\n")


test1PowerDbRef = np.array([ \
55.4192105, 48.26262718, 21.15043411, 41.40514105, -9.8765111, 63.91276256,
27.45124249, 86.12682419, 56.52938252, 45.74445101, 26.1697561, 67.67752988,
43.7591129, 46.79037642, 22.38493824, 50.90278871, 50.00484274, 55.18331118,
77.90062532, 59.72742094
]).reshape(20, 1)

test1CorrectionRef = np.array([ \
-33.08407346, -27.67847232, -31.15312124, -33.23190129, -38.15572091,
-0.88170894, -2.58348763, 0., 0., -3.28188478, -12.34913675, -6.51882838,
-8.17141036, -13., 1.06965221, -12.33169609, -12.84185785, -8.00206861,
-8.16174017, -8.
]).reshape(20, 1)


def main():

    print("\n\n-----Equal Loudness Test Starting-----\n\n")

    equalLoudnessTest1()

    print("\n\n-----Equal Loudness Test Finished-----\n\n")

if __name__ == "__main__": main()
