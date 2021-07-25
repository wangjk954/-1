人体姿态序列分类任务

我们采用LSTM，CNN，ResNet三个模型进行训练，分别对应程序LSTM_classifier.py，人体姿态分类(CNN).py，ResNet_train.py，最终ResNet模型的分类效果最好

我们还做了数据增强进行模型改进，使分类效果进一步提升，对应程序为flip1.py，flip2.py，flip3.py，noise.py

其中LSTM和ResNet使用原数据得到的结果，对应数据文件data，CNN则采用了原数据和一部分增强过的数据得到的结果，对应数据文件data1

只需要将程序和数据放在同一目录下即可运行
