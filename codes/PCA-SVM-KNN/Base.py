import pickle
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import zero_one_loss
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import SVC
from utils import draw, get_max_min, getData
import os
from features import extract_features, analize_file
from config import Config
import pyaudio
import wave

classfiers_num = 50
nsplits = 10


def get_predicts_path(config):
    predict_paths = []
    for predict_path in os.listdir(config.PREDICT_PATH):
        predict_paths.append(os.path.join(config.PREDICT_PATH, predict_path))

    return predict_paths


def processing(database, label_nums, model_name, model_type):
    print(
        '------------------------------特征预处理开始------------------------------')
    config = Config(model_type)
    path = os.path.join(config.DATA_PATH, database)
    files = getData(path)
    max_, min_ = get_max_min(files)
    mfcc_data = []
    for file in files:
        analize_file(file, max_, mfcc_data, label_nums)

    cols = ['file_name', 'features', 'emotion']
    mfcc_pd = pd.DataFrame(data=mfcc_data, columns=cols)
    # 保存音频数据的特征
    pickle.dump(
        mfcc_data,
        open(
            config.BASE_DIR + '\\' + 'preFeatures' + '\\' +
            path.split('\\')[-1] + '_' + model_name + '_features_' +
            label_nums + '.p', 'wb'))
    print(
        '------------------------------特征预处理结束------------------------------')
    return mfcc_pd


def train_svm(database, features, label_num, model_name, model_type):
    print('------------------------------训练开始------------------------------')
    config = Config(model_type)
    y = features['emotion']
    x = features.drop(['file_name', 'emotion'], 1)
    # nsplits折交叉验证
    kf = KFold(nsplits, shuffle=False, random_state=3)
    folds = kf.split(x, y)

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for fold in folds:
        train_data.append(np.array(list(x.iloc[fold[0]]['features'])))
        train_label.append(np.array(list(y.iloc[fold[0]])))
        test_data.append(np.array(list(x.iloc[fold[1]]['features'])))
        test_label.append(np.array(list(y.iloc[fold[1]])))

    # 数据标准化
    train_std = []
    test_std = []
    datatemp = np.concatenate((train_data[0], test_data[0]))
    scaler = StandardScaler().fit(datatemp)
    joblib.dump(
        scaler,
        os.path.join(
            config.BASE_DIR, 'scalers/' + database + '_' + model_name +
            '_scaler_' + label_num + '.m'))
    for f in range(0, nsplits):
        train_std.append(scaler.transform(train_data[f]))
        test_std.append(scaler.transform(test_data[f]))

    # PCA, 数据降维
    pca = PCA()
    train_pca = []
    test_pca = []
    pca.fit(datatemp)

    tot = 0
    varianza_coperta = 0.99
    grather_than_one = []
    for i in pca.explained_variance_ratio_:
        grather_than_one.append(i)
        tot += i
        if (tot >= varianza_coperta):
            break

    pca = PCA(n_components=len(grather_than_one))
    pca.fit(datatemp)
    for f in range(0, nsplits):
        train_pca.append(pca.transform(train_std[f]))
        test_pca.append(pca.transform(test_std[f]))

    # SVM, 线性核, 降维
    svm_k_accuracy = []
    svm_training_errors = []
    svm_test_errors = []
    svm_prediction_test = []
    svms = []
    for f in range(0, nsplits):
        svm = SVC(kernel='rbf', probability=True, gamma='auto')
        svm.fit(train_std[f], train_label[f])
        svms.append(svm)
        svm_prediction_test.append(svm.predict(test_std[f]))
        prediction_training = svm.predict(train_std[f])
        svm_k_accuracy.append(
            metrics.accuracy_score(test_label[f], svm_prediction_test[f]))
        svm_training_errors.append(
            zero_one_loss(train_label[f], prediction_training))
        svm_test_errors.append(
            zero_one_loss(test_label[f], svm_prediction_test[f]))

    print("平均准确率： {:f}".format(sum(svm_k_accuracy) / nsplits))
    print("训练集上的平均损失： {:f}".format(sum(svm_training_errors) / nsplits))
    print("测试集上的平均损失: {:f}".format(sum(svm_test_errors) / nsplits))
    best = svm_k_accuracy.index(max(svm_k_accuracy))
    joblib.dump(
        svms[best],
        os.path.join(
            config.BASE_DIR,
            'models/' + database + '_' + model_name + '_' + label_num + '.m'))
    print('------------------------------训练结束------------------------------')


def predict(database, model_name, label_num, radar, model_type):
    print('------------------------------预测开始------------------------------')
    config = Config(model_type)
    predict_paths = get_predicts_path(config)
    model = joblib.load(
        os.path.join(config.MODEL_PATH,
                     database + '_' + model_name + '_' + label_num + '.m'))
    scaler = joblib.load(
        os.path.join(
            config.SCALER_PATH,
            database + '_' + model_name + '_scaler_' + label_num + '.m'))
    p = pyaudio.PyAudio()

    for predict_path in predict_paths:
        print(predict_path)
        # 语音播放
        f = wave.open(predict_path, 'rb')
        stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                        channels=f.getnchannels(),
                        rate=f.getframerate(),
                        output=True)
        data = f.readframes(f.getparams()[3])
        stream.write(data)
        stream.stop_stream()
        stream.close()
        f.close()

        # 情感检测
        data_feature = extract_features(predict_path)
        feature = data_feature.flatten().reshape(1, 312)
        # 数据归一化
        feature_std = scaler.transform(feature)

        # 预测结果
        print("predict_result:", model.predict(feature_std))
        print("predict_probability", model.predict_proba(feature_std))

        # 绘制雷达图
        if radar:
            if label_num == '6':
                draw(
                    model.predict_proba(feature_std)[0],
                    np.array(config.EMOTION_LABELS_6), 6)
            elif label_num == '3':
                draw(
                    model.predict_proba(feature_std)[0],
                    np.array(config.EMOTION_LABELS_3), 3)
            elif label_num == '7':
                draw(
                    model.predict_proba(feature_std)[0],
                    np.array(config.EMOTION_LABELS_7), 7)
    print('------------------------------预测结束------------------------------')
