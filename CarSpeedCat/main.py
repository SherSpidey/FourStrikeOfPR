import os
import sys

import cv2
import numpy as np

from speedCat import SpeedCat

from gui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# 最新版opencv与pyqt存在环境变量的冲突，需要删除这些环境变量（仅限于服务器等远程桌面）
for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


class RunGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.filename = None
        self.cap = None
        self.frameRate = None
        self.passing_first = None
        self.passing_second = None
        self.Timer = QTimer()
        self.cat = SpeedCat()

        self.connect_all()
        self.button_init()
        self.coil_init()
        self.display_clear()

    # 显示清空
    def display_clear(self):
        img = np.ones((820, 460, 3), dtype='uint8') * 180
        img = QImage(img.data, img.shape[0], img.shape[1], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    # 显示图片（彩色）
    def display(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape != (460, 820, 3):
            img = cv2.resize(img, (820, 460), interpolation=cv2.INTER_AREA)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    # 读取当前帧时间
    def get_seconds(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # 获得车辆速度
    @staticmethod
    def get_velocity(start, end, length=3.5):
        velocity = length / (end - start) * 3.6
        return velocity

    # 时分秒转化
    @staticmethod
    def get_time(seconds):
        hours = 0
        minutes = 0
        milli = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        if seconds > 60:
            minutes = seconds // 60
            seconds = seconds % 60
        if minutes > 60:
            hours = minutes // 60
            minutes = minutes % 60
        time = str(hours).zfill(2) + ':' + str(minutes) \
            .zfill(2) + ':' + str(seconds).zfill(2) + '.' + str(milli).zfill(3)
        return time

    # 预测撞线时间
    @staticmethod
    def get_prediction(velocity, length=4.5):
        return np.round(length / velocity * 3.6, 3)

    # 只有在能够发挥作用时，按钮才可被按下
    def button_init(self):
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)

    # 线圈储值初始化
    def coil_init(self):
        self.passing_first = []
        self.passing_second = []

    # 视频播放初始化
    def play_init(self):
        self.textBrowser.clear()
        self.coil_init()
        self.cap = cv2.VideoCapture(self.filename)
        self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        # 定时器设置中断播放视频
        self.Timer.start(int(1 / self.frameRate * 1000))
        self.pushButton_3.setEnabled(True)

    def open_file(self):
        self.stop()
        self.filename, _ = QFileDialog.getOpenFileName(None,
                                                       '选择文件', '',
                                                       'Video Files(*.mp4 *.avi);;All Files(*.*)')
        if self.filename != '':
            self.pushButton_2.setEnabled(True)

    # 进入中断播放视频
    def play(self):
        success, frame = self.cap.read()
        if success:
            # 获取当前帧时间
            seconds = self.get_seconds()
            time = self.get_time(seconds)
            # 缩放图片
            img = cv2.resize(frame, (820, 460), None, cv2.INTER_AREA)
            # 检测器更新
            self.cat.update(img)
            # 是否有车辆经过第一个线圈
            if self.cat.first_change():
                self.passing_first.append(seconds)
                self.textBrowser.append('在' + time + '经过第一个线圈')
            # 是否有车辆经过第二个线圈
            if self.cat.second_change():
                # 车辆匹配，已经有车辆经过第一个线圈了，防干扰
                if len(self.passing_first) != 0:
                    velocity = self.get_velocity(self.passing_first[0], seconds)
                    speed = str(np.round(velocity, 2)) + ' Km/h'
                    self.textBrowser.append('在' + time + '经过第二个线圈')
                    self.textBrowser.append('此时时速为：' + speed)
                    predict = self.get_prediction(velocity)
                    self.textBrowser.append('预计在' + str(predict) + 's 后到达预测线')
                    self.passing_second.append(seconds)
                    # 匹配完成，删除信息
                    del (self.passing_first[0])
            # 是否有车辆经过第三个线圈
            if self.cat.final_change():
                # 车辆匹配
                if len(self.passing_second) != 0:
                    self.textBrowser.append('在' + time + '经过预测线')
                    del (self.passing_second[0])

            # 防止停止后由于异步残留余像
            if self.cap.isOpened():
                self.display(img)

    # 停止视频
    def stop(self):
        if self.cap is not None:
            self.cap.release()
        self.display_clear()
        self.button_init()

    # 按键功能，中断映射
    def connect_all(self):
        self.Timer.timeout.connect(self.play)
        self.pushButton.clicked.connect(self.open_file)
        self.pushButton_2.clicked.connect(self.play_init)
        self.pushButton_3.clicked.connect(self.stop)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RunGUI()
    gui.show()
    sys.exit(app.exec_())
