import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Data(object):

    def __init__(self, data_dir='./data/locate_data/'):
        self.root = data_dir
        self.data_img = []
        self.data_mask = []
        self.data_size = 0
        self.load_data()

    def load_data(self):
        print('开始导入数据，请稍等...')
        start = int(time.time())
        img_root = self.root + 'train/'
        mask_root = self.root + 'mask/'
        size = len(os.listdir(img_root))
        for i in range(1, size + 1):
            img_name = img_root + str(i) + '.png'
            mask_name = mask_root + str(i) + '.png'
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            self.data_img.append(img)
            self.data_mask.append(mask)
        self.data_size = size
        print('数据集导入完毕，耗时%d秒。' % (int(time.time()) - start))

    def next_Batch(self, size=10):
        images = []
        masks = []
        for _ in range(size):
            index = np.random.randint(0, self.data_size)
            images.append(self.data_img[index])
            masks.append(self.data_mask[index])
        return images, masks


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, drop_rate, out_pad=0):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2, output_padding=out_pad),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv_relu = DoubleDown(middle_channels, out_channels, 3, 1, 0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.dropout(x1)
        x1 = self.conv_relu(x1)
        return x1


class DoubleDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, drop_rate):
        super(DoubleDown, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UNet(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(UNet, self).__init__()
        self.conv1 = DoubleDown(1, 8, 3, 1, 0)

        self.decoder1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False)
        )

        self.conv2 = DoubleDown(8, 16, 3, 1, 0)
        self.decoder2 = Decoder(16, 8 + 8, 8, drop_rate)

        self.conv3 = DoubleDown(16, 32, 3, 1, 0)
        self.decoder3 = Decoder(32, 16 + 16, 16, drop_rate)

        self.conv4 = DoubleDown(32, 64, 3, 1, 0)
        self.decoder4 = Decoder(64, 32 + 32, 32, drop_rate)

        self.conv5 = DoubleDown(64, 128, 3, 1, drop_rate)
        self.decoder5 = Decoder(128, 64 + 64, 64, drop_rate)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool(conv4)
        conv5 = self.conv5(pool4)
        decode4 = self.decoder5(conv5, conv4)
        decode3 = self.decoder4(decode4, conv3)
        decode2 = self.decoder3(decode3, conv2)
        decode1 = self.decoder2(decode2, conv1)
        decode0 = self.decoder1(decode1)
        out = self.out(decode0)
        return out

    @staticmethod
    def array2tensor(imgs):
        imgs = np.array(imgs)
        if len(imgs.shape) == 3:
            return torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).cuda()
        else:
            imgs = imgs.transpose(0, 3, 1, 2)
            return torch.tensor(imgs, dtype=torch.float32).cuda()

    @staticmethod
    def load_model(net, name='./model/net.pth'):
        net.load_state_dict(torch.load(name))

    @staticmethod
    def multi_gpu(net):
        return nn.DataParallel(net).cuda()


if __name__ == '__main__':
    data = Data()
    batch_size = 14
    network = UNet()
    network = nn.DataParallel(network, device_ids=[0, 1]).cuda()
    # UNet.load_model(network, name='./model/model.pth')
    optimizer = optim.SGD(network.parameters(), lr=0.000001)
    print("开始训练！")
    for epoch in range(100):
        total_loss = 0

        for i in range(int(data.data_size / batch_size)):
            imgs, masks = data.next_Batch(batch_size)
            imgs = UNet.array2tensor(imgs)
            imgs = imgs.cuda()
            masks = UNet.array2tensor(masks)
            masks = masks.cuda()

            preds = network(imgs)
            loss = F.mse_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print("current loss:", loss.item())

        print("epoch:", epoch, " total loss:", total_loss)

    torch.save(obj=network.state_dict(), f='./model/model.pth')
