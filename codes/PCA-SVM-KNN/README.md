# The usages of this Demo

## About this demo
1. The codes are based on this [paper](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/blob/master/papers/PCA-SVM-KNN.pdf)
2. Implement 3 and 6 classification

>> 3 classification: 
>>> - neutral
>>> - positive
>>> - negative

>> 6 classification: 
>>> - neutral
>>> - angry
>>> - surprise
>>> - happy
>>> - sad
>>> - fear

3. The train dataset is in `svm/database`
4. The models are in `svm/models`, named as `databasename_modelname_labels.m`
5. Put all your wavs in the `svm/predicts`
6. The features extracted in the preprograssing are put in `svm/preFeatures`, named as `databasename_modelname_labels.m`
7. The scaler models are in the  `databasename_modelname_scaler_labels.m`, every scaler correspondents to a model
8. If you use other database, you'd better rewrite the function of `utils.getData()`. I suggest you rebuild the `svm/database`, put every kind of wavs in one folder


9. The arguments you can use to run this demo:

>>| argument  | choices | description |  
>>| ------ | ------ | ------ |
>>| --help [-h] | | |
>>| --option [-o] | `t`, `p` | train, predict [default is `p`]  |
>>| --database [-db] | `casia`... | your database name [default is `default`]|
>>| --radar [-r] | `1`, `0` | use radar image or not [default is 1]|
>>| --model [-m] | `default` | the name of your model [default is `default`] |
>>| --labels [-l] | `3`, `6` | choose the number of labels [default is `6`]|

10. Examples:
>> `python main.py -o`  to predict your vedio use default model

>> `python main.py -o t -m mymodel -l 3 -db mydatabase` to train your database use 3 labels and name the model as mymodel

>> `python main.py -o p -r 0` to predict your video without radat images



 