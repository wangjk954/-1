from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import random
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings('ignore')

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def data_transforms(data):
    (a, b, c, d, e) = data.shape
    S = np.zeros((a, b - 1, c, d, e - 1))
    for i in range(0, b - 1):
        if i == 1:
            j = i + 1
        else:
            j = i
        S[:, i, :, :, 0] = data[:, j, :, :, 0]
    S = np.resize(S, (1, 65, 65))
    S = torch.tensor(S)
    S = S.float()
    return S



class my_dataset(Dataset):
    def __init__(self, store_path, split, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        for file in glob.glob(self.store_path + '/' + split + '/000/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(0)
        for file in glob.glob(self.store_path + '/' + split + '/001/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(1)
        for file in glob.glob(self.store_path + '/' + split + '/002/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(2)
        for file in glob.glob(self.store_path + '/' + split + '/003/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(3)
        for file in glob.glob(self.store_path + '/' + split + '/004/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/004A/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/004B/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/004C/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/0004/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/0004A/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/0004B/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/0004C/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/1004/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/1004A/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/1004B/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/1004C/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/2004/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/2004A/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/2004B/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)
        for file in glob.glob(self.store_path + '/' + split + '/2004C/*.npy'):
            # print(file)
            cur_path = file.replace('\\', '/')
            self.img_list.append(cur_path)
            self.label_list.append(4)



    def __getitem__(self, item):
        data = np.load(self.img_list[item])
        data = data_transforms(data)
        label = self.label_list[item]
        return data, label

    def __len__(self):
        return len(self.img_list)


####################模型定义（卷积神经网络，CNN）####################
def define_model():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv3 = nn.Conv2d(1, 16, 3, stride=2)  # 卷积层C3，3x3、步长为2的卷积核，feature_map=16
            self.pool3 = nn.MaxPool2d(2, 2)  # 2x2池化层
            self.conv4 = nn.Conv2d(16, 32, 3)  # 卷积层C4，3x3、步长为1的卷积核，feature_map=32
            # 三个全连接层
            self.fc1 = nn.Linear(32 * 14 * 14, 1200)
            self.fc15 = nn.Linear(1200, 84)
            self.fc2 = nn.Linear(84, 5)  # 最后输出为1x5的向量

        def forward(self, x):
            # 卷积层采用relu作为激活函数，池化层无激活函数
            x = self.pool3(torch.relu(self.conv3(x)))
            x = torch.relu(self.conv4(x))
            # 全连接层采用relu作为激活函数
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc15(x))
            x = self.fc2(x)
            return x

    net = Net()
    return net


################损失函数定义（CrossEntropyLoss）##################
def define_loss():
    Loss = nn.CrossEntropyLoss()
    return Loss


##############优化器定义#############
def define_optimizer(learning_rate):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    return optimizer



def acc(y_pred, y):
    t = 0
    (X, Y) = y_pred.shape
    for j in range(0, X):
        max = -999
        index = -1
        for i in range(0, Y):
           if y_pred[j, i] > max:
               max = y_pred[j, i]
               index = i
        if y[j] == index:
            t = t + 1
    return t


###################模型训练#################
def train(loader, net, Loss, num, optimizer):
    print('start training:')
    sum_loss = 0
    i = 0
    t = 0
    j = 0
    for epoch in range(15):
        for x, y in loader:
            x = x.cuda(0)
            y = y.cuda(0)
            y_pred = net(x)  # 前向传播：通过像模型输入x计算预测的y
            loss = Loss(y_pred, y)  # 计算loss
            print("第{}次,CrossEntropyLoss为 {}".format(i + 1, loss.item()))
            i = i + 1
            optimizer.zero_grad()  # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零
            loss.backward()  # 反向传播：根据模型的参数计算loss的梯度
            optimizer.step()  # 调用Optimizer的step函数使它所有参数更新
    for x, y in loader:
        x = x.cuda(0)
        y = y.cuda(0)
        y_pred = net(x)    # 前向传播：通过向模型输入x计算预测的y
        (X) = len(y)
        t = t + acc(y_pred, y)    #训练正确数
        j = j + X         #训练样本数
        loss = Loss(y_pred, y)  # 计算最终的训练误差
        sum_loss = sum_loss + loss   #计算总的训练误差
    print("训练完成,CrossEntropyLoss为 {}".format(sum_loss / j))
    print("训练完成，训练正确率为 {}".format(t / j))    #计算并输入训练正确率
    return net


###################模型测试###################
def test(loader, net, Loss, num):
    sum_loss = 0
    i = 0
    t = 0
    for x, y in loader:
        y_pred = net(x)
        #print(y_pred)
        #print(y)
        loss = (Loss(y_pred, y))  # 计算loss
        sum_loss = sum_loss + loss
        t = t + acc(y_pred, y)   #测试正确数
        i = i + 1           #测试样本数
    print("测试完成,CrossEntropyLoss为 {}".format(sum_loss / i))
    print("测试完成，测试正确率为 {}".format(t / i))     #计算并输出测试正确率
    return 0


if __name__ == '__main__':
    start = time.time()
    #store_path = 'D:/rengongzhinengshiyan/人体姿态序列分类/data'
    store_path = '.'
    split = 'Data1/train'
    train_dataset = my_dataset(store_path, split, data_transforms)
    num = 4        #batch_size
    dataset_loader = DataLoader(train_dataset, batch_size=num, shuffle=True, num_workers=1)  #读取训练数据集
    net = define_model()
    net = net.cuda(0)
    Loss = define_loss()
    optimizer = define_optimizer(1e-4)
    Net1 = train(dataset_loader, net, Loss, num, optimizer)
    Net1.eval()
    Net1 = Net1.cpu()
    split = 'data/test'
    num = 1        #batch_size
    test_dataset = my_dataset(store_path, split, data_transforms)
    test_dataset_loader = DataLoader(test_dataset, batch_size=num, shuffle=True, num_workers=1)  #读取训练数据集
    test(test_dataset_loader, Net1, Loss, num)
    end = time.time()
    Time = end - start
    print("运行时间为 {} s".format(Time))