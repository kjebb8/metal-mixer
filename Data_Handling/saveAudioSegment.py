import os
import numpy as np
from audioHandler import AudioHandler
from ..Spectrogram import spectrogramUtils as specUtils

'''
File: saveAudioSegment.py
Brief: Saves segment(s) of instrument audio loaded from a folder to a different
       folder with the correct name for the InstrumentDataParser to process.

       Only one instrument can be processed at a time. If processing multiple
       files, the name must be "<string><idx>_ds.wav".

       The output file name is "<instrument>_<nextIdx>.wav".
'''
def saveAudioSegment():
    audioFilePath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "ML_Music_To_Save/"
    saveFilePath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        ""

    # Inputs
    audioName = ""
    instrument = ""
    maxFilesIndex = 0
    startFileIndex = 0

    # Use either useFullAudio = True or specify a start and end time
    useFullAudio = False
    startTime = 0  # seconds
    endTime = 1  # seconds

    for i in range(startFileIndex, maxFilesIndex + 1):
        print(i)

        if maxFilesIndex > 0:
            audioFile = audioName + str(i) + "_ds.wav"
        else:
            audioFile = audioName + "_ds.wav"

        (sampleRate, audio) = AudioHandler.getWavFile(audioFilePath + audioFile)

        if useFullAudio:
            audioSegment = audio
        else:
            startSample = specUtils.getSamplesfromTime(sampleRate, startTime)
            endSample = specUtils.getSamplesfromTime(sampleRate, endTime)
            audioSegment = audio[startSample:endSample]

        audioSegment = np.copy(audioSegment).astype(np.float64)

        nextInstIdx = 0
        for filename in os.listdir(saveFilePath):
            if filename.endswith(".wav"):
                filenameSplit = filename.split('_')
                if filenameSplit[0] == instrument:
                    instIdx = int(filenameSplit[1].split('.')[0])
                    if instIdx > nextInstIdx:
                        nextInstIdx = instIdx
        nextInstIdx += 1
        AudioHandler.writeWaveFile(\
            saveFilePath + instrument + "_" + str(nextInstIdx) + ".wav",
            sampleRate,
            audioSegment)


def main():
    print("\n\n----------Save Audio Segment Script----------\n\n")

    saveAudioSegment()

    print("\n\n----------Save Audio Segment Finished----------\n\n")
if __name__ == "__main__": main()
