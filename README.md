人体姿态序列分类任务

我们采用LSTM，CNN，ResNet三个模型进行训练，分别对应程序LSTM_classifier.py，人体姿态分类(CNN).py，ResNet_train.py，最终ResNet模型的分类效果最好

我们还做了数据增强进行模型改进，使分类效果进一步提升，对应程序为flip01.py，flip02.py，flip03.py，noise.py

其中flip01.py做了左右翻转，flip02.py做了逆序操作，flip03.py做了左右翻转和逆序操作，noise.py则是添加噪声

其中LSTM和ResNet使用原数据得到的结果，对应数据文件data，CNN则采用了原数据和一部分增强过的数据得到的结果，对应数据文件Data1

只需要将对应程序和相应数据放在同一目录下即可运行
