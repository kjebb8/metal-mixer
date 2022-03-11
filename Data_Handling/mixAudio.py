from audioHandler import AudioHandler

'''
File: mixAudio.py
Brief: Load multiple audio files, mix (add) them together, and save to a file.
'''

def mixAudio():
    audioFilePath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Music_Separation_Test_Dataset/"
    saveFilePath = \
        "/Users/keeganjebb/Documents/Programming_2/Metal_Mixer/Music_Files/" + \
        "Music_Separation_Test_Dataset/"

    audioSrc1Name = "true_piano_drums_piano_mixed_2_500ex"
    audioSrc2Name = "true_drums_drums_piano_mixed_2_500ex"
    audioMixedName = "drums_piano_mixed_2_500ex"


    print("Source 1: " + audioSrc1Name)
    print("Source 2: " + audioSrc2Name)

    (sampleRate1, audioSrc1) = \
        AudioHandler.getWavFile(audioFilePath + audioSrc1Name + ".wav")
    audioSrc1 = AudioHandler.convertToMono(audioSrc1)
    (sampleRate2, audioSrc2) = \
        AudioHandler.getWavFile(audioFilePath + audioSrc2Name + ".wav")
    audioSrc2 = AudioHandler.convertToMono(audioSrc2)
    assert(sampleRate1 == sampleRate2)

    audioSrcList = [audioSrc1, audioSrc2]
    (mixedAudio, _) = AudioHandler.mixAudio(audioSrcList)

    mixedFileName = saveFilePath + audioMixedName + ".wav"
    print("Saving mixed audio to:\n" + mixedFileName)

    AudioHandler.writeWaveFile(\
        mixedFileName,
        sampleRate1,
        mixedAudio)


def main():
    print("\n\n----------Mix Audio Script----------\n\n")

    mixAudio()

    print("\n\n----------Mix Audio Finished----------\n\n")
if __name__ == "__main__": main()