import pickle
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import SVC

classfiers_num = 50
nsplits = 10


# 使用KNN分类器，K：1-50
def knnClassify(classfiers_num, train_data, train_label, test_data,
                test_label):
    k_s = range(1, classfiers_num + 1)
    knn_classifiers = [
        KNeighborsClassifier(n_neighbors=k, n_jobs=-1) for k in k_s
    ]
    training_errors = list(range(1, classfiers_num + 1))
    test_errors = list(range(1, classfiers_num + 1))
    k_accuracy = np.zeros(classfiers_num)

    for i in range(0, classfiers_num):
        # print(i)
        fold_training_errors = list(range(0, nsplits))
        fold_test_errors = list(range(0, nsplits))
        fold_accuracy = list(range(0, nsplits))
        for fold in range(0, nsplits):
            knn_classifiers[i].fit(train_data[fold], train_label[fold])
            prediction_training = knn_classifiers[i].predict(train_data[fold])
            prediction_test = knn_classifiers[i].predict(test_data[fold])
            fold_accuracy[fold] = metrics.accuracy_score(
                test_label[fold], prediction_test)
            fold_training_errors[fold] = zero_one_loss(train_label[fold],
                                                       prediction_training)
            fold_test_errors[fold] = zero_one_loss(test_label[fold],
                                                   prediction_test)
        k_accuracy[i] = sum(fold_accuracy) / nsplits
        # print("k_accuracy" + str(i) + ":" + str(k_accuracy[i]))
        training_errors[i] = sum(fold_training_errors) / nsplits
        # print("traing_errors" + str(i) + ":" + str(training_errors[i]))
        test_errors[i] = sum(fold_test_errors) / nsplits
        # print("test_errors" + str(i) + ":" + str(test_errors[i]))

    joblib.dump(knn_classifiers, "knn_classifiersm.m")
    return training_errors, test_errors, k_accuracy


if __name__ == "__main__":
    # 载入数据
    cols = ['name', 'features', 'emotion']
    features = pickle.load(open('Features.p', 'rb'))
    features = pd.DataFrame(data=features, columns=cols)
    y = features['emotion']
    x = features.drop(['name', 'emotion'], 1)

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

    # training_errors, test_errors, k_accuracy = knnClassify(
    #     classfiers_num, train_data, train_label, test_data, test_label)

    # # k_accuracy最高时对应的k：
    # best_k = k_accuracy.argmax() + 1
    # print(best_k)
    # plt.figure()
    # plt.plot(k_accuracy)
    # plt.ylabel("accuracy")
    # plt.xlabel("k")
    # plt.title("Accuracy of each $K$")
    # plt.savefig("k_accuracy.png")
    # plt.show()

    # plt.figure()
    # plt.plot(training_errors)
    # plt.plot(test_errors)
    # plt.legend(['Training error', 'Test error'])
    # plt.title("Errors for each $K$")
    # plt.xlabel("k")
    # plt.ylabel("errors")
    # plt.savefig("train_test_errors.png")
    # plt.show()
    # min_test_error_index = test_errors.index(min(test_errors))
    # min_test_value = min(test_errors)
    # print(min_test_error_index)
    # print(min_test_value)

    # 数据标准化
    train_std = []
    test_std = []
    datatemp = np.concatenate((train_data[0], test_data[0]))
    scaler = StandardScaler().fit(datatemp)
    joblib.dump(scaler, 'models/scaler.m')
    for f in range(0, nsplits):
        train_std.append(scaler.transform(train_data[f]))
        test_std.append(scaler.transform(test_data[f]))

    # PCA, 数据降维
    # pca = PCA()
    # train_pca = []
    # test_pca = []
    # pca.fit(datatemp)

    # tot = 0
    # varianza_coperta = 0.99
    # grather_than_one = []
    # for i in pca.explained_variance_ratio_:
    #     grather_than_one.append(i)
    #     tot += i
    #     if (tot >= varianza_coperta):
    #         break

    # pca = PCA(n_components=len(grather_than_one))
    # pca.fit(datatemp)
    # for f in range(0, nsplits):
    #     train_pca.append(pca.transform(train_std[f]))
    #     test_pca.append(pca.transform(test_std[f]))

    # SVM, 线性核, 降维
    svm_k_accuracy = []
    svm_training_errors = []
    svm_test_errors = []
    svm_prediction_test = []
    # for f in range(0, nsplits):
    #     print(f)
    #     print(train_std[f].shape)
    #     print(train_label[f].shape)

    #     svm = SVC(kernel='rbf', probability=True, gamma='auto')
    #     svm.fit(train_std[f], train_label[f])
    #     joblib.dump(svm, 'models/rbf_svm_' + str(f) + '.m')
    #     svm_prediction_test.append(svm.predict(test_std[f]))
    #     prediction_training = svm.predict(train_std[f])
    #     svm_k_accuracy.append(
    #         metrics.accuracy_score(test_label[f], svm_prediction_test[f]))
    #     svm_training_errors.append(
    #         zero_one_loss(train_label[f], prediction_training))
    #     svm_test_errors.append(
    #         zero_one_loss(test_label[f], svm_prediction_test[f]))

    # print("平均准确率： {:f}".format(sum(svm_k_accuracy) / nsplits))
    # print("训练集上的平均损失： {:f}".format(sum(svm_training_errors) / nsplits))
    # print("测试集上的平均损失: {:f}".format(sum(svm_test_errors) / nsplits))
    # print("最佳的k：" + str(svm_k_accuracy.index(max(svm_k_accuracy))))
    # print("最小的测试集损失对应的k:" + str(svm_test_errors.index(min(svm_test_errors))))
    # pca_training_errors, pca_test_errors, pca_k_accuracy = knnClassify(
    #     classfiers_num, train_pca, train_label, test_pca, test_label)

    # print("pca_k_acc_max_index:" + str(pca_k_accuracy.argmax() + 1))
    # print("min_pca_test_error:" + str(min(pca_test_errors)))
    # print("min_pca_test_error_index:" +
    #       str(pca_test_errors.index(min(pca_test_errors))))

    # plt.figure()
    # plt.plot(pca_k_accuracy)
    # plt.ylabel("pca_accuracy")
    # plt.xlabel("k")
    # plt.title("Accuracy of each $K$ after pca")
    # plt.savefig("pca_k_accuracy.png")
    # plt.show()

    # plt.figure()
    # plt.plot(pca_training_errors)
    # plt.plot(pca_test_errors)
    # plt.legend(['Training error', 'Test error'])
    # plt.title("Errors for each $K$ after pca")
    # plt.xlabel("k")
    # plt.ylabel("errors")
    # plt.savefig("pca_train_test_errors.png")
    # plt.show()
