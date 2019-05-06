import os
import librosa
from random import shuffle

path = r'C:\GitHub\speech_er_demo\casia'


def getData():
    wav_file_path = []
    # wavs = os.listdir(path)
    # for wav in wavs:
    #     wav_file_path.append(os.path.join(path, wav))
    person_dirs = os.listdir(path)
    for person in person_dirs:
        if person.endswith('txt'):
            continue
        emotion_dir_path = os.path.join(path, person)
        emotion_dirs = os.listdir(emotion_dir_path)
        for emotion_dir in emotion_dirs:
            if emotion_dir.endswith('.ini'):
                continue
            emotion_file_path = os.path.join(emotion_dir_path, emotion_dir)
            emotion_files = os.listdir(emotion_file_path)
            for file in emotion_files:
                if not file.endswith('wav'):
                    continue
                wav_path = os.path.join(emotion_file_path, file)
                wav_file_path.append(wav_path)

    # 将语音文件随机排列
    shuffle(wav_file_path)
    return wav_file_path


def get_max_min(files):
    min_, max_ = 100, 0
    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_


# if __name__ == "__main__":
#     files = getData()
#     max, min = get_max_min(files)
#     print(files[0])
