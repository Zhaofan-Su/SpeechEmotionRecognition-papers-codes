KNN only:
1. best k_accuracy, k=12(非索引，索引为11)
2. min_test_value: 0.295
3. min_test_value_index: k=12
4. 最好精确度：0.705

KNN after PCA:
1. best k_accuracy, k=31(非索引，索引为30)
2. min_test_value: 0.317
3. min_test_value_index: k=31
4. 最好精确度：0.683

SVM after Scaler, kernel——linear:
1. best k_accuracy, k=4(索引对应为3)
2. 平均测试集损失： 0.225
3. 测试集损失最小时，对应的k：10
4. 平均准确率：0.775

SVM after Scaler， kernel——rbf, gamma——auto：（目前最好）
1. best k_accuracy, k=9（索引对应为8）
2. 平均测试集损失：0.222
3. 测试集损失最小时，对应的k：9
4. 平均准确率：0.778

SVM only, kernel——linear:
1. 最好准确率对应的k: k=1
2. 平均测试集损失：0.308
3. 测试集损失最小对应的k：1
4. 平均准确率：0.693

SVM only, kernel——rbf:
1. 最好准确率对应的k: k=1
2. 平均测试集损失：0.888
3. 测试集损失最小对应的k：1
4. 平均准确率：0.112