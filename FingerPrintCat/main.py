import os
import sys

import cv2
import numpy as np

import Tools

from gui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap

# 最新版opencv与pyqt存在环境变量的冲突，需要删除这些环境变量（仅限于服务器等远程桌面）
for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


class RunGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img = None
        self.filename = None
        self.enhancer = None

        self.button_init()
        self.display_init()
        self.connect_all()

    @staticmethod
    def display_clear(label):
        img = np.ones((200, 200, 3), dtype='uint8') * 180
        img = QImage(img.data, img.shape[0], img.shape[1], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

    # 在特定label显示图片
    @staticmethod
    def display(img, label, color=False):
        if color:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        else:
            img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                         QImage.Format_Indexed8)
        label.setPixmap(QPixmap.fromImage(img))

    def open_file(self):
        self.button_init()
        self.filename, _ = QFileDialog. \
            getOpenFileName(None, '选择文件', '', 'Image Files(*.png *.jpg *.tif);;All Files(*.*)')
        if self.filename != '':
            self.img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            self.img = cv2.resize(self.img, (200, 200))
            self.display(self.img, self.label)
            self.pushButton_2.setEnabled(True)

    # 指纹增强
    def enhance(self):
        self.enhancer = Tools.Enhancer()
        self.img = self.enhancer.enhance(self.img)
        self.display(self.img, self.label_2)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(True)

    # 指纹细化
    def thin(self):
        # 形态操作是对于255像素值而言，所以需要在闭运算前反色
        kernel = np.ones((2, 2), np.uint8)
        temp = 255 - self.img
        # 闭运算连通含糊断点
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        self.img = np.array(255 - temp, np.uint8)
        thinner = Tools.Thinner(self.img)
        self.img = thinner.thin()
        self.display(self.img, self.label_3)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(True)

    # 特征提取
    def cat(self):
        feature = Tools.FeaturesCat(self.img, self.enhancer.get_orient())
        self.img = feature.cat()
        self.display(self.img, self.label_4, color=True)
        points = feature.get_features()
        self.textBrowser.clear()
        for point in points:
            detail = '<h3>' + point[0] + '，  ' + point[1] + '，  ' + point[2] + '</h3>'
            self.textBrowser.append(detail)
        self.button_init()

    # 按键功能初始化
    def button_init(self):
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)

    # 显示初始化
    def display_init(self):
        self.display_clear(self.label)
        self.display_clear(self.label_2)
        self.display_clear(self.label_3)
        self.display_clear(self.label_4)
        self.textBrowser.clear()

    # 功能映射
    def connect_all(self):
        self.pushButton.clicked.connect(self.open_file)
        self.pushButton_2.clicked.connect(self.enhance)
        self.pushButton_3.clicked.connect(self.thin)
        self.pushButton_4.clicked.connect(self.cat)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RunGUI()
    gui.show()
    sys.exit(app.exec_())
