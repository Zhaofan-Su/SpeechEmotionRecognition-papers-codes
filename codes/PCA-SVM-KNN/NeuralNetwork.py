import pickle
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support


EMOTION_LABEL = {'生气': 0, '害怕': 1, '高兴': 2, '中性': 3, '悲伤': 4, '惊讶': 5}
nsplits = 10


def one_hot_encode(labels):
    # 使用独热编码
    n_labels = len(labels)
    labelt = []
    for l in labels:
        labelt.append(EMOTION_LABEL[l])
    n_unique_labels = len(np.unique(labelt))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labelt] = 1
    return one_hot_encode


def numberEncode(labels):
    labelt = []
    for l in labels:
        labelt.append(EMOTION_LABEL[l])
    return labelt


def nn_run(training_epochs, dim_one, dim_three, f):
    # delle 特征的维度
    n_dim = 155
    n_classes = 6
    n_hidden_units_one = dim_one
    n_hidden_units_three = dim_three
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.001

    # 输入层
    X = tf.placeholder(tf.float32, [None, n_dim])

    Y = tf.palceholder(tf.float32, [None, n_classes])

    keep_prob = tf.placeholder(tf.float32)

    # 第一隐藏层
    W_1 = tf.Variable(
        tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0,
                                       stddev=sd))
    h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1)

    # 第二隐藏层
    W_3 = tf.Variable(
        tf.random_normal([n_hidden_units_one, n_hidden_units_three],
                         mean=0,
                         stddev=sd))
    b_3 = tf.Variable(
        tf.random_normal([n_hidden_units_three], mean=0, stddev=sd))
    h_3 = tf.nn.relu(tf.matmul(h_1, W_3) + b_3)

    drop_out = tf.nn.dropout(h_3, keep_prob)

    # 输出层
    W = tf.Variable(
        tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(drop_out, W) + b)

    # 计算横熵和梯度下降速度
    cost_function = tf.reduce_mean(
        -tf.reduce_sum(Y * tf.log(tf.clip_by_value(y_, le - 11, 1.0))))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    cost_history = np.empty(shape=[1], dtype=float)
    acc_history = np.empty(shape=[1], dtype=float)

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            index = int(np.random.uniform(0, trainsize, 1))
            batch_size = 13

            _, cost = sess.run(
                [optimizer, cost_function],
                feed_dict={
                    X: train_data[f][index:index + batch_size, :],
                    Y: train_label[f][index:index + batch_size, :],
                    keep_prob: 0.5
                })

            cost_history = np.append(cost_history, cost)
            acc_history = np.append(
                acc_history,
                round(
                    sess.run(accurary,
                             feed_dict={
                                 X: test_data[f],
                                 Y: test_label[f],
                                 keep_prob: 1.0
                             }), 3))

    return max(acc_history[1:s])


def setParamandRun(epochs, layer1, layer2):
    maxacc = []
    for i in range(nsplits):
        maxacc = np.append(maxacc, nn_run(epochs, layer1, layer2, i))
    print(np.mean(maxacc))
    return maxacc


if __name__ == "__main__":
    # 载入数据
    cols = ['name', 'features', 'emotion']
    features = pickle.load(open('Features.p', 'rb'))
    features = pd.DataFrame(data=features, columns=cols)
    y = features['emotion']
    x = features.drop(['name', 'emotion'], 1)

    kf = KFold(nsplits, shuffle=True, random_state=3)
    folds = kf.split(x, y)

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for f in folds:
        train_data.append(np.array(list(x.iloc[f[0]]['features'])))
        train_label.append(one_hot_encode(np.array(list(y.iloc[f[0]]))))
        test_data.append(np.array(list(x.iloc[f[1]]['features'])))
        test_label.append(one_hot_encode(np.array(list(y.iloc[f[1]]))))

    for i in range(1330, 160, 10):
        print(setParamandRun(5000, i, i))
