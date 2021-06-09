import cv2
import numpy as np


class Coil(object):
    def __init__(self, coil):
        self.im = None
        self.im_gray = None
        self.coil = coil
        self.black_value = 0
        self.cur_value = 0
        self.passing = False

    # 更新输入图片
    def update_image(self, src):
        # 应用输入图片
        self.im = src
        # 灰度值转化
        self.im_gray = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY)

    # 绘出线圈
    def draw(self, color=(0, 0, 255)):
        for p in range(len(self.coil)):
            start = self.coil[p]
            # 回到起点
            if p == len(self.coil) - 1:
                end = self.coil[0]
            else:
                end = self.coil[p + 1]
            # 画出对应直线
            cv2.line(self.im, start, end, color=color)

    # 计算先后线圈灰度值
    def update_value(self):
        # 生成转化矩阵
        # 计算矩形高度
        h = (self.coil[1][1] + self.coil[2][1] - self.coil[0][1] - self.coil[3][1]) // 2
        # 计算矩形宽度
        w = (self.coil[2][0] + self.coil[3][0] - self.coil[0][0] - self.coil[1][0]) // 2
        # 左上，左下，右上，右下的位置，与初始位置不太相同
        box_1 = np.float32([self.coil[0], self.coil[1], self.coil[3], self.coil[2]])
        box_2 = np.float32([(0, 0), (0, h), (w, 0), (w, h)])
        transform_mat = cv2.getPerspectiveTransform(box_1, box_2)
        # 得到转化后的线圈图像
        coil = cv2.warpPerspective(self.im_gray, transform_mat, (w, h))
        value = np.sum(coil) / w / h
        # 更新灰度值
        if self.black_value == 0:
            self.black_value = value
        self.cur_value = value

    # 检测是否有车辆通过
    def detect_passing(self):
        diff = self.black_value - self.cur_value
        # 阈值判断
        if diff > 28 and not self.passing:
            self.passing = True
            return True
        elif diff < 5:
            self.passing = False
        return False


class SpeedCat(object):
    def __init__(self):
        # 初始设定线圈位置，左上，左下，右下，右上
        self.coil_1_points = [(198, 223), (202, 227), (296, 224), (288, 220)]
        self.coil_2_points = [(221, 244), (225, 250), (362, 243), (348, 238)]
        self.destination_points = [(298, 310), (305, 316), (536, 297), (526, 291)]
        self.coil_1 = None
        self.coil_2 = None
        self.destination = None

        self.coils_init()

    # 线圈初始化，生成对应对象
    def coils_init(self):
        self.coil_1 = Coil(self.coil_1_points)
        self.coil_2 = Coil(self.coil_2_points)
        self.destination = Coil(self.destination_points)

    def draw_coils(self):
        # 第一、二个线圈画红色，减速带画绿色
        self.coil_1.draw()
        self.coil_2.draw()
        self.destination.draw(color=(0, 255, 0))

    # 更新图片和灰度值
    def update(self, src):
        # 更新当前处理帧
        self.coil_1.update_image(src)
        self.coil_2.update_image(src)
        self.destination.update_image(src)
        # 更新当前帧灰度值
        self.coil_1.update_value()
        self.coil_2.update_value()
        self.destination.update_value()

        # 当前帧画出线圈
        self.draw_coils()

    def first_change(self):
        return self.coil_1.detect_passing()

    def second_change(self):
        return self.coil_2.detect_passing()

    def final_change(self):
        return self.destination.detect_passing()


if __name__ == '__main__':
    # img = cv2.imread("./test.jpg")
    # img = cv2.resize(img, (820, 460), None, cv2.INTER_AREA)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cat = SpeedCat()
    # cat.update(img)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('测试完毕！')
