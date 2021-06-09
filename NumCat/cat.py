import math
import cv2
import numpy as np


class Cat(object):
    def __init__(self, img=None, file_name='./test.jpg'):
        if img is None:
            self.pic_mode = True
            self.img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.img = cv2.resize(self.img, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
            self.output = self.img.copy()
        else:
            self.img = img
            self.pic_mode = False
            self.output = self.img.copy()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGBA2GRAY)
        self.__getImgPurged__()
        self.__catRects__()

    def __getImgPurged__(self):
        _, self.origin = cv2.threshold(self.img, 100, 255, cv2.THRESH_BINARY)
        self.origin = cv2.GaussianBlur(self.origin, (5, 5), 0)
        _, self.img = cv2.threshold(self.img, 100, 255, cv2.THRESH_BINARY_INV)
        m_kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.dilate(self.img, m_kernel, iterations=1)
        self.origin = 1 - self.origin / 255

    def __catRects__(self):
        rects = []
        contours, hierarchy = cv2.findContours(self.img,
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 500 > h > 40 and 400 > w > 10:
                rect = np.array([x, y, w, h])
                rects.append(rect)
        self.Rects = np.array(rects)

    def __getRects__(self):
        return self.Rects

    def __catNums__(self):  # 从图片的截取数字，大小与mnist相同
        imgs = []
        for rect in self.Rects:
            out_box = np.zeros((28, 28))
            img = self.origin[rect[1]:rect[1] + rect[3],
                  rect[0]:rect[0] + rect[2]]
            if rect[2] < rect[3]:
                inner_box = np.zeros((rect[3], rect[3]))
                inner_box[:, int(math.floor((rect[3] - rect[2]) / 2)):
                             int(math.floor((rect[3] - rect[2]) / 2) + rect[2])] = img
            else:
                inner_box = np.zeros((rect[2], rect[2]))
                inner_box[int(math.floor((rect[2] - rect[3]) / 2)):
                          int(math.floor((rect[2] - rect[3]) / 2) + rect[3]), :] = img
            img = cv2.resize(inner_box, (20, 20), interpolation=cv2.INTER_AREA)
            out_box[4:24, 4:24] = img
            imgs.append(out_box)
        if len(imgs) != 0:
            imgs.append(imgs[-1])
        return np.array(imgs)

    def __putNums__(self, predictions):  # 加工原始图片
        # k=np.max(self.Rects[:,2])/55          #字体大小粗细缩放系数，已弃用
        for (rect, prediction) in zip(self.Rects, predictions[0:-1]):
            x, y, w, h = rect
            self.output = cv2.putText(self.output, str(prediction.item()), (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),
                                      int(5), cv2.LINE_AA, False)

    def __showPic__(self):  # 单个图片结果展示
        cv2.imshow('Result', self.output)
        if self.pic_mode:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
