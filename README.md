# SpeechEmotionRecognition-papers-codes

### Papers 
1. **3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition** *Mingyi Chen, Xuanji He, Jing Yang, and Han Zhang* [[paper]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/blob/master/papers/CRNN_IEMOCAP.pdf)

  > CRNN
  >
  > IEMOCAP数据库

2. **Speech Emotion Recognition Using Deep Neural Network and Extreme Learning Machine** *Kun Han, Dong Yu, Ivan Tashev* [[paper]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/blob/master/papers/DNN_ExtremeLearingMachine.pdf)
  
  > DNN
  >
  > Extreme Learning Machine

3. **Interpretazione affettiva di feature audio** *Marco Faleri* [[paper]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/blob/master/papers/PCA-SVM-KNN.pdf)
  
  > Italy
  >
  > PCA + SVM + KNN
  >
  > 复现了代码，使用PCA对声音特征数据降维之后，再使用SVM，在中文语音数据集上准确率最高为77.8%
  >
  > 使用KNN神经网络调参还没有复现，作者实现之后的准确率最高达到89.62%，数据集为柏林数据集，不过二者相差结果不大
  >
  > 目前代码已经整理完成，具体使用方法参见[[README.md]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/tree/master/codes/PCA-SVM-KNN)

4. **Emotion Recognition from Chinese Speech for Smart Affective Services Using a Combination of SVM and DBN** *Lianzhang Zhu, Leiming Chen, Dehai Zhao, Jiehan Zhou and Weishan Zhang* [[paper]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/blob/master/papers/SVM_DBN.pdf)
   
  > SVM + DBN
  >
  > 中文语音情感集，作者的准确度达到95.8%??
  
### Codes
1. 复现**Interpretazione affettiva di feature audio**的代码[[codes]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/tree/master/codes/PCA-SVM-KNN)
  
  > PCA + SVM + KNN
 
2. **CRNN** [[codes]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/tree/master/codes/CRNN_IEMOCAP)
  
  > IEMOCAP数据集

3. **BLSTM** [[codes]](https://github.com/Zhaofan-Su/SpeechEmotionRecognition-papers-codes/tree/master/codes/BLSTM_68.6)
  
  > 柏林数据集
  >
  > 作者最好的准确率68.6%
