import os
import cv2
import numpy as np
import qimage2ndarray as q2n
import tensorflow as tf
from PyQt5.Qt import QWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, QLabel, QSpinBox, QCheckBox,QMessageBox

from CNN.cnn import CNN
from GUI.PaintBorad import PainBoard
import Bayes.Bayes as bys
import Fisher.Fisher as fsh
import Perception.Pt as pt
import pylab

class MainWidget(QWidget):
    def __del__(self):
        try:
            if self.train_mode:
                if self.train_mode_way=='cnn':
                    self.saver.save(self.sess, os.path.join("CNN", "model", "Model-1"))
                    self.sess.close()
                #if self.train_mode_way=='fisher':
                    #fsh.train('Fisher')
        except:
            return

    # 初始化，直接super利用父类
    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.nowtrain = 0
        self.train_mode =False #  True##用于迁移学习
        self.train_mode_way = 'Pt'#'bayes'#'cnn'# #判断是何种学习方式
        self.__initCNN__()
        if self.train_mode == True:
            self.model.lr = 1e-6
        self.__initWidget__()

    def __getImg__(self,x,y):           #截取图像，然后进行二值化处理
        img = self.__paintBoard__.getImg()
        img = q2n.rgb_view(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = 1 - img / 255
        cut = np.argwhere(img == 1)
        cut = cut.transpose(1, 0)
        xmax = cut[0].max()
        xmin = cut[0].min()
        ymax = cut[1].max()
        ymin = cut[1].min()
        img = cv2.resize(img[xmin:xmax, ymin:ymax], (x,y))
        return  bys.binalize(img)

    def __initWidget__(self, title="数字识别"):

        # 窗口大小
        self.setFixedSize(640, 480)
        # 窗口标题
        self.setWindowTitle(title)
        self.__paintBoard__ = PainBoard(self)
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)
        main_layout.addWidget(self.__paintBoard__)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(20, 20, 20, 20)

        self.__clearbut__ = QPushButton("清空画板")
        self.__clearbut__.setParent(self)  # 设置父对象为本界面
        self.__clearbut__.clicked.connect(self.__paintBoard__.clear)  # 将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__clearbut__)

        self.__erasemode__ = QCheckBox("   橡皮擦")
        self.__erasemode__.setParent(self)
        self.__erasemode__.clicked.connect(self.__oneraseclick__)
        sub_layout.addWidget(self.__erasemode__)

        self.__painttn__ = QLabel(self)
        self.__painttn__.setText("      画笔粗细")
        self.__painttn__.setFixedHeight(20)
        sub_layout.addWidget(self.__painttn__)

        self.__painttnlist__ = QSpinBox(self)
        self.__painttnlist__.setMaximum(40)
        self.__painttnlist__.setMinimum(20)
        self.__painttnlist__.setValue(30)
        self.__painttnlist__.setSingleStep(2)
        self.__painttnlist__.valueChanged.connect(self.__onpensizechange__)
        sub_layout.addWidget(self.__painttnlist__)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        if self.train_mode == True:
            self.__painttn2__ = QLabel(self)
            self.__painttn2__.setText(" 当前训练数字")
            self.__painttn2__.setFixedHeight(20)
            sub_layout.addWidget(self.__painttn2__)

            self.__trainlist__ = QSpinBox(self)
            self.__trainlist__.setMaximum(9)
            self.__trainlist__.setMinimum(0)
            self.__trainlist__.setValue(0)
            self.__trainlist__.setSingleStep(1)
            self.__trainlist__.valueChanged.connect(self.__trainchange__)
            sub_layout.addWidget(self.__trainlist__)

        self.__runbut4__ = QPushButton("感知机分类")
        self.__runbut4__.setParent(self)  # 设置父对象为本界面
        self.__runbut4__.clicked.connect(self.__run4__)  # 将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__runbut4__)

        self.__runbut3__ = QPushButton("Fisher分类")
        self.__runbut3__.setParent(self)  # 设置父对象为本界面
        self.__runbut3__.clicked.connect(self.__run3__)  # 将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__runbut3__)

        self.__runbut2__ = QPushButton("Bayes分类")
        self.__runbut2__.setParent(self)  # 设置父对象为本界面
        self.__runbut2__.clicked.connect(self.__run2__)  # 将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__runbut2__)

        self.__runbut1__ = QPushButton("卷积网络")
        self.__runbut1__.setParent(self)  # 设置父对象为本界面
        self.__runbut1__.clicked.connect(self.__run1__)  # 将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__runbut1__)

        main_layout.addLayout(sub_layout)

    def __initCNN__(self):
        self.model = CNN()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Restore pretrained weights
        self.saver.restore(self.sess, os.path.join("CNN", "model", "Model"))

    def __oneraseclick__(self):
        if self.__erasemode__.isChecked():
            self.__paintBoard__.erase = True
        else:
            self.__paintBoard__.erase = False

    def __onpensizechange__(self):
        self.__paintBoard__.changeThick(self.__painttnlist__.value())

    def __trainchange__(self):
        self.nowtrain = self.__trainlist__.value()

    def __run1__(self):
        if not self.train_mode:
            img = self.__paintBoard__.getImg()
            img = q2n.rgb_view(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            img = 1 - img / 255
            img = img.reshape(1, 784)
            y = self.sess.run(self.model.out, feed_dict={self.model.x: img, self.model.rate: 0})
            #print(np.argmax(y))
            QMessageBox.about(self, "识别结果", "输入数字为"+str(np.argmax(y))+" !")
        elif self.train_mode_way == 'cnn':
            img = self.__paintBoard__.getImg()
            img = q2n.rgb_view(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            img = 1 - img / 255
            img = img.reshape(1, 784)
            train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            train[self.nowtrain] = 1
            train = np.array(train)
            train = train[np.newaxis, :]
            self.sess.run(self.model.train_op, feed_dict={self.model.x: img, self.model.y: train, self.model.rate: 0})
            self.__paintBoard__.clear()

    def __run2__(self):
        img = self.__getImg__(50,70)
        #pylab.imshow(img)
        #pylab.show()
        #print(len(np.argwhere(img == 1))+len(np.argwhere(img == 0)))
        if not self.train_mode:
            result=bys.test_no_G(img,os.path.join('Bayes','model.txt'),os.path.join('Bayes','num.txt'))
            #result = bys.test(img, 'Bayes')
            QMessageBox.about(self, "识别结果", "输入数字为" + str(result) + " !")

        elif self.train_mode_way == 'bayes':
            bys.train(img,self.nowtrain,os.path.join('Bayes','model.txt'),os.path.join('Bayes','num.txt'))
            #bys.data_save(img, self.nowtrain,'Bayes')
            self.__paintBoard__.clear()

    def __run3__(self):
        img =self.__getImg__(10,14)
        if not self.train_mode:
            result=fsh.test(img,'Fisher')
            QMessageBox.about(self, "识别结果", "输入数字为" + str(result) + " !")
        elif self.train_mode_way == 'fisher':
            fsh.save_data(img,self.nowtrain,'Fisher')
            #fsh.train('Fisher')
            self.__paintBoard__.clear()

    def __run4__(self):
        img =self.__getImg__(10,14)
        if not self.train_mode:
            result=pt.test(img,'Perception')
            QMessageBox.about(self, "识别结果", "输入数字为" + str(result) + " !")
        elif self.train_mode_way == 'Pt':
            fsh.save_data(img,self.nowtrain,'Perception')
            self.__paintBoard__.clear()


