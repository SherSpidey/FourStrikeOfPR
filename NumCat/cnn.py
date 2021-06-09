import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


class NetWork(object):
    @staticmethod
    def built(drop=0):
        net_work = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9),  # 第一个卷积层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),  # batch norm
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),  # 第二个卷积层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=32 * 3 * 3, out_features=512),  # 全连接层
            nn.Dropout(drop),  # DropOut防止过拟合                #切记！！！！！！使用网络时需要关闭网络的DropOut，不然你训练的网络只有一部分在工作啊！
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=64),  # 特征分类
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),  # 输出层)
        )
        return net_work.cuda()

    @staticmethod
    def load(net_work):
        net_work.load_state_dict(torch.load("./model/net.pth"))

    @staticmethod
    def array2tensor(imgs):
        return torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).cuda()


if __name__ == '__main__':

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


    network = NetWork.built(drop=0.5)  # 训练时开启DropOut，设置比例为0.5

    train_set = torchvision.datasets.MNIST(
        root='./data/MNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
    data = next(iter(loader))
    mean = data[0].mean()
    std = data[0].std()

    train_set_normal = torchvision.datasets.MNIST(  # 训练集正则化
        root='./data/MNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    )

    network.cuda()
    loader = DataLoader(train_set_normal, batch_size=200, num_workers=1)
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    print("Start training!\n")

    for epoch in range(100):
        total_loss = 0
        total_correct = 0

        for batch in loader:
            images = batch[0].cuda()
            labels = batch[1].cuda()
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print("epoch:", epoch, " total loss:", total_loss, " Accuracy:",
              total_correct / len(train_set_normal))

    torch.save(obj=network.state_dict(), f='./model/net.pth')
