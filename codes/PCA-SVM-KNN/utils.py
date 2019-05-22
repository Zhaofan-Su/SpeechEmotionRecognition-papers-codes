import os
import librosa
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


def getData(path):
    wav_file_path = []

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


def draw(data_prob, class_labels: tuple, num_classes: int):
    plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
    # 数据
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    data = np.concatenate((data_prob, [data_prob[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    fig = plt.figure(1)
    # polar参数
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, 'bo-', linewidth=2)
    ax.fill(angles, data, facecolor='r', alpha=0.25)
    ax.set_thetagrids(
        angles * 180 / np.pi, class_labels, fontproperties="SimHei")
    ax.set_title("Emotion Recognition", va='bottom', fontproperties="SimHei")
    # 在这里设置雷达图的数据最大值
    ax.set_rlim(0, 1)
    ax.grid(True)
    plt.pause(1)  # 暂停时间
