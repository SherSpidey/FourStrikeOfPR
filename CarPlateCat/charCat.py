import os
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class Tools(object):
    @staticmethod
    def get_label(label):
        label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z',
                      'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui',
                      'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
                      'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng',
                      'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong', 'zh_shan',
                      'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin',
                      'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun', 'zh_zang', 'zh_zhe']
        index = label_list.index(label)
        label = np.zeros(len(label_list))
        label[index] = 1
        return label

    @staticmethod
    def get_char(label_hot):
        char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
                      '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁',
                      '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋', '皖',
                      '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
        return char_table[label_hot.argmax()]

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


class Data(object):
    def __init__(self, root='./data/plate_data/'):
        self.label = None
        self.data = None
        self.root = root
        self.data_list = os.listdir(root)
        self.data_size = self.get_data_size()
        self.init_data()

    def init_data(self):  # 读取图片，生成标签并保存
        data = []
        label = []
        print('开始导入数据，请稍等...')
        start = int(time.time())
        for char in self.data_list:
            char_dir = self.root + char
            data_x = []
            data_y = []
            for file in os.listdir(char_dir):
                filename = os.path.join(char_dir, file).replace('\\', '/')
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                target = Tools.get_label(char)
                data_x.append(img)
                data_y.append(target)
            data.append(data_x)
            label.append(data_y)
        self.data = data
        self.label = label
        print('数据集导入完毕，耗时%d秒。' % (int(time.time()) - start))

    def get_data_size(self):
        num = 0
        for char in self.data_list:
            char_dir = self.root + char
            num += len(os.listdir(char_dir))
        return num

    def next_Batch(self, size=10):
        images = []
        labels = []
        for _ in range(size):
            char_index = np.random.randint(0, len(self.data))  # 类别随机采样，样本过采样
            file_index = np.random.randint(0, len(self.data[char_index]))
            img = self.data[char_index][file_index]
            label = self.label[char_index][file_index]
            images.append(img)
            labels.append(label)
        return np.array(images), np.array(labels)


class CNN(object):
    @staticmethod
    def load_model(net, name='./model/net.pth'):
        net.load_state_dict(torch.load(name))

    @staticmethod
    def array2tensor(imgs, train=True):
        if train:
            return torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).cuda()
        else:
            return torch.tensor(imgs, dtype=torch.float32).cuda()

    @staticmethod
    def build_network(drop_rate=0.5):
        network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(5 * 5 * 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 65)
        )
        return network.cuda()


if __name__ == '__main__':
    data = Data()
    batch_size = 15
    network = CNN.build_network()
    # CNN.load_model(network, name='./model/charCat.pth')
    optimizer = optim.SGD(network.parameters(), lr=0.00001)
    print("开始训练！")
    for epoch in range(50):
        epoch_loss = 0
        epoch_accuracy = 0
        for _ in range(int(data.data_size / batch_size) + 1):
            images, labels = data.next_Batch(batch_size)
            images = CNN.array2tensor(images)
            labels = torch.tensor(labels, dtype=torch.long).cuda()
            labels = torch.argmax(labels, -1)  # 注意pytorch计算cross entropy会自动转化onehot
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += Tools.get_num_correct(preds, labels)

        print("epoch:", epoch, " total loss:", epoch_loss, " Accuracy:{}%"
              .format(epoch_accuracy / ((int(data.data_size / batch_size) + 1)
                                        * batch_size) * 100))

    torch.save(obj=network.state_dict(), f='./model/charCat.pth')
