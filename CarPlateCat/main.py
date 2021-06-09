import sys
import numpy as np
from imgCat import CAT
from UNet import UNet
from charCat import CNN, Tools

from GUI import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap

import os
import cv2

# 最新版opencv与pyqt存在环境变量的冲突，需要删除这些环境变量（仅限于服务器等远程桌面）
for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


class RunGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img = None
        self.img_gray = None
        self.UNet = None
        self.CNN = None
        self.filename = None
        self.display_clear(self.label)
        self.display_clear(self.label_2)
        self.net_init()
        self.connect_all()

    # 模型初始化与加载
    def net_init(self):
        self.UNet = UNet()
        self.UNet = UNet.multi_gpu(self.UNet)
        UNet.load_model(self.UNet, name='./model/model.pth')
        self.CNN = CNN.build_network()
        CNN.load_model(self.CNN, name='./model/charCat.pth')
        self.UNet.eval()
        self.CNN.eval()

    def cat(self):
        image = CAT.get_test_img(self.img_gray)
        img = image.reshape(-1, 720, 720)
        img = UNet.array2tensor(img)
        mask = self.UNet(img)
        mask = mask.cpu().detach().numpy().reshape(720, 720)
        mask = CAT.get_mask(mask)
        plate = CAT.get_plate(mask, image)
        self.display(plate, self.label_2, main_screen=False)
        char_list, flag = CAT.char_cat(plate)
        if flag:
            self.textBrowser.clear()
            plate_cat = ''
            images = CNN.array2tensor(char_list)
            chars = self.CNN(images)
            chars = chars.cpu().detach().numpy()
            for i in range(len(chars)):
                if i == 2:
                    plate_cat += '·'
                plate_cat += Tools.get_char(chars[i])
            plate_cat = '<h1>'+plate_cat+'</h1>'
            self.textBrowser.append(plate_cat)
        else:
            self.textBrowser.clear()
            text = '<h1>识别失败！</h1>'
            self.textBrowser.append(text)

    def display_clear(self, label):
        img = np.ones((540, 540, 3), dtype='uint8') * 180
        img = QImage(img.data, img.shape[0], img.shape[1], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

    def display(self, img, label, main_screen=True):
        if main_screen:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (540, 540), None, interpolation=cv2.INTER_AREA)
            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        else:
            img = cv2.resize(img, (210, 70), None, interpolation=cv2.INTER_AREA)
            img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                         QImage.Format_Indexed8)
        label.setPixmap(QPixmap.fromImage(img))

    def open_file(self):
        self.filename, _ = QFileDialog. \
            getOpenFileName(None, '选择文件', '', 'Image Files(*.png *.jpg);;All Files(*.*)')
        if self.filename != '':
            self.img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
            self.img_gray = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            self.display(self.img, self.label)

    def plate_cat(self):
        if self.filename is None:
            QMessageBox.warning(self, "警告", "你需要先打开文件！")
        self.cat()

    def connect_all(self):
        self.pushButton.clicked.connect(self.open_file)
        self.pushButton_2.clicked.connect(self.plate_cat)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RunGUI()
    gui.show()
    sys.exit(app.exec_())
