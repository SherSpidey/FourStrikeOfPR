import sys
import numpy as np
import threading
import time

from cnn import NetWork
from cat import Cat

from gui import Ui_MainWindow
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
        self.checkBox.setChecked(True)  # 默认检测视频
        self.video_mode = True
        self.camera_mode = False
        self.save = False
        self.filename = None
        self.cap = None
        self.video_saver =None
        self.frameRate = 30
        self.network = NetWork.built()
        self.run_flag = threading.Event()  # 用以开始与停止线程
        self.running = threading.Event()  # 用以暂停与恢复线程
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.display_clear()
        self.connect_all()
        NetWork.load(self.network)

    def display_clear(self):
        img = np.ones((1024, 576, 3), dtype='uint8') * 180
        img = QImage(img.data, img.shape[0], img.shape[1], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    def display(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (1024, 576), interpolation=cv2.INTER_AREA)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    def open_file(self):
        if self.camera_mode:
            QMessageBox.warning(self, "警告", "当前设置为摄像头！")
        else:
            self.filename, _ = QFileDialog. \
                getOpenFileName(None, '选择文件', '', 'Video Files(*.mp4 *.avi);;All Files(*.*)')

    def video_set(self):
        self.checkBox.setChecked(True)
        self.checkBox_2.setChecked(False)
        self.video_mode = True
        self.camera_mode = False

    def camera_set(self):
        self.checkBox_2.setChecked(True)
        self.checkBox.setChecked(False)
        self.video_mode = False
        self.camera_mode = True

    def save_set(self):
        self.save = not self.save

    def play(self):
        self.run_flag.clear()
        self.running.set()  # 显示初始化
        if self.filename is None or len(self.filename) == 0:
            QMessageBox.warning(self, "警告", "你需要先打开文件！")
        else:
            if self.video_mode:
                self.cap = cv2.VideoCapture(self.filename)
                self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
                self.run_flag.set()
                self.running.set()
            else:
                self.cap = cv2.VideoCapture(0)

        # 创建视频显示的线程
        run_display = threading.Thread(target=self.run_thread)
        run_display.start()

    def run_thread(self):
        while self.run_flag.is_set():
            self.pushButton_3.setText('暂停')
            self.pushButton.setEnabled(False)
            self.pushButton_3.setEnabled(True)
            self.pushButton_4.setEnabled(True)
            if self.save:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_saver = cv2.VideoWriter('out.mp4', fourcc, 30.0, (1920, 1080))
            while self.cap.isOpened():
                self.running.wait()  # False时暂停线程
                if not self.run_flag.is_set():
                    self.display_clear()
                    break
                success, frame = self.cap.read()
                if not success:
                    self.run_flag.clear()
                    self.display_clear()
                    self.pushButton.setEnabled(True)
                    break
                cat = Cat(frame)
                imgs = cat.__catNums__()
                if len(imgs) != 0:
                    imgs = NetWork.array2tensor(imgs)
                    preds = self.network(imgs)
                    cat.__putNums__(preds.argmax(dim=1))
                if self.save:
                    self.video_saver.write(cat.output)
                if self.cap.isOpened():  # 防止停止后由于线程的异步性继续显示图片
                    self.display(cat.output)

                if self.video_mode:
                    time.sleep(0.2 / self.frameRate)  # 识别时拖慢了速度，单纯播放视频时0.2改为1

    def pause_continue(self):
        if self.running.is_set():
            self.running.clear()
            self.pushButton_3.setText('继续')
        else:
            self.running.set()
            self.pushButton_3.setText('暂停')

    def stop(self):
        self.run_flag.clear()
        self.cap.release()
        if self.save:
            self.video_saver.release()
        self.pushButton.setEnabled(True)
        self.display_clear()
        self.pushButton_3.setText('暂停')
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)

    def connect_all(self):
        self.checkBox.clicked.connect(self.video_set)
        self.checkBox_2.clicked.connect(self.camera_set)
        self.checkBox_3.clicked.connect(self.save_set)
        self.pushButton.clicked.connect(self.open_file)
        self.pushButton_2.clicked.connect(self.play)
        self.pushButton_3.clicked.connect(self.pause_continue)
        self.pushButton_4.clicked.connect(self.stop)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RunGUI()
    gui.show()
    sys.exit(app.exec_())
