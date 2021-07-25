import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def main():
    # 定义自己的数据集
    class my_dataset(Dataset):
        def __init__(self, store_path, split, transform):
            self.store_path = store_path
            self.split = split
            self.transform = transform
            self.img_list = []
            self.label_list = []
            for sample in glob.glob(store_path + '/' + split + '/0004/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(0)
            for sample in glob.glob(store_path + '/' + split + '/004A/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(1)
            for sample in glob.glob(store_path + '/' + split + '/004B/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(2)
            for sample in glob.glob(store_path + '/' + split + '/004C/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(3)
            for sample in glob.glob(store_path + '/' + split + '/0004/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(4)
            for sample in glob.glob(store_path + '/' + split + '/000/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(0)
            for sample in glob.glob(store_path + '/' + split + '/001/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(0)
            for sample in glob.glob(store_path + '/' + split + '/002/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(0)
            for sample in glob.glob(store_path + '/' + split + '/003/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(0)
            for sample in glob.glob(store_path + '/' + split + '/004/*.npy'):
                cur_path = sample.replace('\\', '/')
                self.img_list.append(cur_path)
                self.label_list.append(0)

        def __getitem__(self, item):
            data = np.load(self.img_list[item])
            data = transform(data)
            label = self.label_list[item]
            return data, label

        def __len__(self):
            return len(self.label_list)

    def transform(data):
        batch_size, cod, frame, point, number = data.shape
        temp = np.zeros((batch_size, cod - 1, frame, point, number - 1))
        for i in range(0, cod - 1):
            if i == 1:
                j = i + 1
            else:
                j = i
            temp[:, i, :, :, 0] = data[:, j, :, :, 0]
        temp = np.resize(temp, (65, 65))
        temp = torch.tensor(temp)
        temp = temp.float()
        return temp

    # 定义LSTM模型
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size=65, hidden_size=64, num_layers=3, batch_first=True)
            self.out = nn.Linear(64, 5)

        def forward(self, x):
            r_out, (h_n, h_c) = self.lstm(x, None)
            out = self.out(r_out[:, -1, :])
            return out

    data_store_path1 = './data'
    data_store_path2 = './data_augmentation'
    split_train = 'train'
    split_test = 'test'

    lstm = LSTM()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-4)  # 定义优化器
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数

    train_dataset = my_dataset(data_store_path2, split_train, transform)
    train_dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_dataset = my_dataset(data_store_path1, split_test, transform)
    test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 训练模型
    i = 1
    sum_loss = 0
    for epoch in range(20):
        for x, y in train_dataset_loader:
            output = lstm(x)
            loss = loss_function(output, y)
            sum_loss = sum_loss + loss.item()
            if i % 100 == 0:
                print('第{}次训练，CrossEntropyLoss为：{}'.format(i, loss.item()))
            i = i + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('训练完成，平均误差为：{}'.format(sum_loss/i))

    # 测试模型
    acc_num = 0
    total_num = 0
    for x, y in test_dataset_loader:
        y_pred = lstm(x)
        y_pred = torch.max(y_pred, 1)[1].data.numpy().squeeze()
        total_num = total_num + 1
        if y_pred == y.item():
            acc_num = acc_num + 1
    print('test accuracy: %.4f' % (acc_num/total_num))


if __name__ == '__main__':
    main()




