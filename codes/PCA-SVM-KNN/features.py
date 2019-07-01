import librosa
import librosa.display
import numpy as np

EMOTION_LABEL_6 = {
    'angry': '生气',
    'fear': '害怕',
    'happy': '高兴',
    'neutral': '中性',
    'sad': '悲伤',
    'surprise': '惊讶'
}
EMOTION_LABEL_3 = {'neutral': '中性', 'positive': '正向', 'negative': '负向'}
EMOTION_LABEL_7 = {
    'angry': '生气',
    'boring': '无聊',
    'disgust': '厌恶',
    'fear': '害怕',
    'happy': '高兴',
    'neutral': '中性',
    'sad': '悲伤'
}


def extract_feature_data_augmentation(file_name, max_):
    # 使用原始采样率读取音频文件
    X, sample_rate = librosa.load(file_name, sr=None)

    startPad = 0.5 * sample_rate
    length = (max_ * sample_rate) + startPad - X.shape[0]

    wn = np.random.randn(X.shape[0])
    X1 = X + 0.002 * wn

    X0 = np.pad(X, (0, int(length)), 'constant')
    X1 = np.pad(X1, (0, int(length)), 'constant')
    X2 = np.pad(X, (int(startPad), int(length - startPad)), 'constant')

    return features(X0, sample_rate), features(X1, sample_rate), features(
        X2, sample_rate)


def extract_features(file, pad=False):
    X, sample_rate = librosa.load(file, sr=None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return features(X, sample_rate)


def features(X, sample_rate):
    stft = np.abs(librosa.stft(X))

    # fmin和fmax对应于人类语音的最小最大基本频率
    pitches, magnitudes = librosa.piptrack(X,
                                           sr=sample_rate,
                                           S=stft,
                                           fmin=70,
                                           fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    # pitchmin = np.min(pitch)

    # 频谱质心
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # 谱平面
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # 使用系数为50的MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,
                    axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,
                      axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T,
                     axis=0)

    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                     axis=0)

    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # ottava对比
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft,
                                                         sr=sample_rate).T,
                       axis=0)

    # 过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # 均方根能量
    rmse = librosa.feature.rmse(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate(
        (ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features


def only_mfcc(X, sample_rate):
    mfccs = librosa.feature.mfcc(X,
                                 sample_rate,
                                 n_mfcc=13,
                                 hop_length=int(0.010 * sample_rate),
                                 n_fft=int(0.025 * sample_rate))
    ext_features = mfccs[0]
    temp = ext_features.shape[0]
    for mfcc in mfccs:
        ext_features = np.concatenate((ext_features, mfcc), axis=0)
    return ext_features[temp:]


def onlyPitch(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    pitches, magnitudes = librosa.piptrack(X,
                                           sr=sample_rate,
                                           S=stft,
                                           fmin=70,
                                           fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch.append(pitches[index, i])
    return np.asarray(pitch)


def analize_file(f, max_, mfcc_data, label_num):
    # fn = mypath + f
    print(f)
    ext_features = extract_features(f, max_)
    if label_num == '6':
        mfcc_data.append([f, ext_features, EMOTION_LABEL_6[f.split('\\')[-2]]])
    elif label_num == '3':
        label = f.split('\\')[-2]
        if label == 'angry' or label == 'fear':
            mfcc_data.append([f, ext_features, EMOTION_LABEL_3['negative']])
        elif label == 'happy' or label == 'surprise':
            mfcc_data.append([f, ext_features, EMOTION_LABEL_3['positive']])
        else:
            mfcc_data.append([f, ext_features, EMOTION_LABEL_3['neutral']])
    elif label_num == '7':
        mfcc_data.append([f, ext_features, EMOTION_LABEL_7[f.split('\\')[-2]]])
    print(f, 'end')