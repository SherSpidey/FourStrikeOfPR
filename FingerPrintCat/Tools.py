import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import signal


class Enhancer(object):
    def __init__(self, kernel=16, mask_thresh=0.1, gradient_sigma=1,
                 block_sigma=7, smooth_sigma=7, freq_size=38, freq_h=5,
                 min_wave_len=5, max_wave_len=15, kx=0.65, ky=0.65,
                 angle=3, filter_thresh=-3, freq_thresh=2):
        self.kernel = kernel
        self.mask_thresh = mask_thresh
        self.gradient_sigma = gradient_sigma
        self.block_sigma = block_sigma
        self.smooth_sigma = smooth_sigma
        self.freq_size = freq_size
        self.freq_h = freq_h
        self.min_wae_len = min_wave_len
        self.max_wave_len = max_wave_len
        self.kx = kx
        self.ky = ky
        self.angle = angle
        self.filter_thresh = filter_thresh
        self.freq_thresh = freq_thresh

        self._mask = None
        self._normed = None
        self._oriented = None
        self._freq_mean = None
        self._freq_median = None
        self._freq = None
        self._freq_im = None
        self._binary_im = None

    def __normalization(self, src):  # Z-score标准化
        self._normed = (src - np.mean(src)) / np.std(src)

    def __ridge_segment(self, src):
        h, w = src.shape
        self.__normalization(src)

        # 确保能被整数分割
        box_h = (h // self.kernel + 1) * self.kernel
        box_w = (w // self.kernel + 1) * self.kernel

        # 均值是0， 所以填充是zeros
        box = np.zeros((box_h, box_w))
        mask = np.zeros((box_h, box_w))

        box[0:h, 0:w] = self._normed

        # 计算每一个分块的标准差
        for i in range(0, box_h, self.kernel):
            for j in range(0, box_w, self.kernel):
                kernel = box[i:i + self.kernel, j:j + self.kernel]
                mask[i:i + self.kernel, j:j + self.kernel] = np.ones((self.kernel,
                                                                      self.kernel)) * np.std(kernel)
        mask = mask[0:h, 0:w]
        # 根据区域块分离指纹与背景
        self._mask = mask > self.mask_thresh
        # 重新计算分离后的均值与标准差
        mask_mean = np.mean(self._normed[self._mask])
        mask_std = np.std(self._normed[self._mask])
        self._normed = (self._normed - mask_mean) / mask_std

    def __ridge_orientation(self):
        # 生成高斯核，计算梯度大小
        kernel_size = np.fix(6 * self.gradient_sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1

        gauss = cv2.getGaussianKernel(int(kernel_size), self.gradient_sigma)
        kernel = gauss * gauss.T

        # X与Y方向的一阶导数
        kernel_y, kernel_x = np.gradient(kernel)

        # 离散卷积操作，保持尺寸不变，实际计算过程卷积核进行了翻转
        gx = signal.convolve2d(self._normed, kernel_x, mode='same')
        gy = signal.convolve2d(self._normed, kernel_y, mode='same')
        # x的偏导与y的偏导的点乘，此时不能直接利用公式计算方向场，还需进行低通滤波
        gx_2 = np.power(gx, 2)
        gy_2 = np.power(gy, 2)
        gxy = gx * gy

        # 平滑图像
        kernel_size = np.fix(6 * self.block_sigma)
        gauss = cv2.getGaussianKernel(int(kernel_size), self.block_sigma)
        kernel = gauss * gauss.T

        gx_2 = ndimage.convolve(gx_2, kernel)
        gy_2 = ndimage.convolve(gy_2, kernel)
        gxy = 2 * ndimage.convolve(gxy, kernel)

        # 开始计算角度的解析解
        common = np.sqrt(np.power(gxy, 2) + np.power((gx_2 - gy_2), 2)) + np.finfo(float).eps  # 尾项实现尽可能的精确解

        # 计算出的角度是两倍角
        double_sin = gxy / common
        double_cos = (gx_2 - gy_2) / common

        # 继续平滑
        if self.smooth_sigma:
            kernel_size = np.fix(6 * self.smooth_sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gauss = cv2.getGaussianKernel(int(kernel_size), self.smooth_sigma)
            kernel = gauss * gauss.T
            double_sin = ndimage.convolve(double_sin, kernel)
            double_cos = ndimage.convolve(double_cos, kernel)

        self._oriented = np.arctan2(double_sin, double_cos) / 2 + np.pi / 2

    # 分块的频率场计算
    def __calculate_freq(self, im_block, orient_block):
        h, w = im_block.shape

        # 从二倍角的平均正弦余弦计算出平均的方向场块平均角度，可以有效防止杂乱问题（均值导致角度消失）
        double_sin = np.mean(np.sin(2 * orient_block))
        double_cos = np.mean(np.cos(2 * orient_block))
        theta = np.arctan2(double_sin, double_cos) / 2

        # 旋转分块方向与方向场方向竖直
        rotated_im = ndimage.rotate(im_block, theta / np.pi * 180 + 90,
                                    axes=(1, 0), reshape=False, order=3,
                                    mode='nearest')

        # 消除边缘干扰区域
        kernel_size = int(np.fix(h / np.sqrt(2)))
        offset = int(np.fix((h - kernel_size) / 2))
        rotated_im = rotated_im[offset:offset + kernel_size, offset:offset + kernel_size]

        # 对指纹进行计数，计算频率
        shadows = np.sum(rotated_im, axis=0)  # 水平计算投影
        dilation = ndimage.grey_dilation(shadows, self.freq_h,
                                         structure=np.ones(self.freq_h))

        dis = np.abs(dilation - shadows)

        max_bool = (dis < self.freq_thresh) & (shadows > np.mean(shadows))
        max_index = np.where(max_bool)

        _, nums = np.shape(max_index)
        # 如果峰数大于2，根据第一个与最后一个之间的距离以及数量计算频率
        if nums < 2:
            return np.zeros(im_block.shape)
        else:
            peaks = nums
            wave_length = (max_index[0][peaks - 1] - max_index[0][0]) / (peaks - 1)
            if self.max_wave_len > wave_length > self.min_wae_len:
                return 1 / np.double(wave_length) * np.ones(im_block.shape)
            else:
                return np.zeros(im_block.shape)

    # 计算频率场
    def __ridge_freq(self):
        h, w = self._normed.shape
        freq = np.zeros((h, w))

        # 分块计算频率场
        for i in range(0, h - self.freq_size, self.freq_size):
            for j in range(0, w - self.freq_size, self.freq_size):
                im_block = self._normed[i:i + self.freq_size, j:j + self.freq_size]
                orient_block = self._oriented[i:i + self.freq_size, j:j + self.freq_size]
                freq[i:i + self.freq_size, j:j + self.freq_size] = self.__calculate_freq(im_block, orient_block)
        # 背景指纹分离
        self._freq = freq * self._mask
        freq_1d = np.reshape(self._freq, (1, -1))

        # 实际存在频率场的区域
        index = np.where(freq_1d > 0)
        index = np.array(index)
        # 坐标第一项都是0，因为输入是形状是（1，？）
        index = index[1, :]

        valid_freq = freq_1d[0][index]
        self._freq_mean = np.mean(valid_freq)
        self._freq_median = np.median(valid_freq)
        # 生成频率场图
        self._freq = self._freq_mean * self._mask

    def __gabor_filter(self):
        img = np.double(self._normed)
        h, w = img.shape
        box = np.zeros(self._normed.shape)

        freq_1d = np.reshape(self._freq, (1, -1))
        index = np.where(freq_1d > 0)
        index = np.array(index)
        index = index[1, :]

        # 不知所以的精度步骤
        valid_freq = freq_1d[0][index]
        valid_freq = np.double(np.round(valid_freq * 100)) / 100
        uniq_freq = np.unique(valid_freq)

        sigma_x = 1 / uniq_freq[0] * self.kx
        sigma_y = 1 / uniq_freq[0] * self.ky

        kernel_size = np.int(np.round(3 * np.max([sigma_x, sigma_y])))
        x, y = np.meshgrid(np.linspace(-kernel_size, kernel_size, (2 * kernel_size + 1)),
                           np.linspace(-kernel_size, kernel_size, (2 * kernel_size + 1)))
        # gabor滤波器
        filter_org = np.exp(-((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y))) \
                     * np.cos(2 * np.pi * uniq_freq[0] * x)
        filter_y, filter_x = filter_org.shape

        angle_range = np.int(180 / self.angle)
        gabor_filter = np.zeros((angle_range, filter_y, filter_x))

        for f in range(angle_range):
            rotate_filter = ndimage.rotate(filter_org, -(f * self.angle + 90), reshape=False)
            gabor_filter[f] = rotate_filter

        max_size = kernel_size
        temp = self._freq > 0
        valid_h, valid_w = np.where(temp)

        final_temp = (valid_h > max_size) & (valid_h < h - max_size) & (valid_w > max_size) & (valid_w < w - max_size)
        final_index = np.where(final_temp)

        max_orient_index = np.round(180 / self.angle)
        orient_index = np.round(self._oriented / np.pi * 180 / self.angle)

        # 开始滤波
        for y in range(h):
            for x in range(w):
                if orient_index[y][x] < 1:
                    orient_index[y][x] += max_orient_index
                if orient_index[y][x] > max_orient_index:
                    orient_index[y][x] -= max_orient_index

        _, final = np.shape(final_index)
        for i in range(final):
            y = valid_h[final_index[0][i]]
            x = valid_w[final_index[0][i]]

            img_block = img[y - kernel_size:y + kernel_size + 1, x - kernel_size:x + kernel_size + 1]

            box[y, x] = np.sum(img_block * gabor_filter[int(orient_index[y][x]) - 1])

        self._binary_im = box >= self.filter_thresh

    # 获取方向场
    def get_orient(self):
        return self._oriented

    def enhance(self, src):
        self.__ridge_segment(src)
        self.__ridge_orientation()
        self.__ridge_freq()
        self.__gabor_filter()
        return np.array(255 * self._binary_im, dtype='uint8')


def Normalization(src, m=100, v=100):  # 标准正则化
    return np.array(m + np.sqrt(v) * (src - np.mean(src)) / np.std(src), dtype='uint8')


def normal_visualize(src):  # 正则化显示
    hist, _ = np.histogram(src, 256, [0, 255])
    plt.fill(hist)
    plt.xlim(0, 255)
    plt.ylim(0, None)
    plt.xlabel('gray contribution')
    plt.show()


# 方向场计算
def directionField(src, kernel_size=16, m=100):
    h, w = src.shape

    box_h = (h // kernel_size + 1) * kernel_size
    box_w = (w // kernel_size + 1) * kernel_size

    box = np.ones((box_h, box_w)) * m
    thetas = []

    box[0:h, 0:w] = src

    for i in range(0, box_h, kernel_size):
        for j in range(0, box_w, kernel_size):
            kernel = box[i:i + kernel_size, j:j + kernel_size]
            gx = cv2.Sobel(kernel, cv2.CV_64F, dx=1, dy=0)
            gy = cv2.Sobel(kernel, cv2.CV_64F, dx=0, dy=1)
            gxy = (2 * gx * gy).sum()

            gx2 = sum(gx * gx).sum()
            gy2 = sum(gy * gy).sum()

            theta = math.degrees(0.5 * math.atan(2 * gxy / (gx2 - gy2)))
            thetas.append(theta)


class Thinner(object):
    def __init__(self, src, mode='table'):
        self._im = src.copy()
        self._mode = mode

    # 连通函数
    def __NC8(self, b):
        d_sum = 0
        d = b.copy()
        d.append(b[1])
        for i in range(1, 5):
            d_sum = d_sum + (d[2 * i - 1] - d[2 * i - 1] * d[2 * i] * d[2 * i + 1])
        return d_sum

    # Hilditch细化
    def Hilditch(self, src, reverse=True):
        if len(src.shape) == 3:
            print('警告，细化前需要进行二值化处理！')
            return
        im = src.copy()
        if reverse:
            im = np.array((255 - im), dtype="uint8")
        h, w = im.shape
        # 一开始count为1，表示最开始要进入循环
        px, py, count = 1, 1, 1

        offset = [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]

        while count != 0:
            # 以3x3的领域遍历图片中所有的点
            count = 0
            del_list = []
            for y in range(h):
                for x in range(w):
                    b = [0 for _ in range(9)]
                    for i in range(9):
                        px = x + offset[i][0]
                        py = y + offset[i][1]
                        if w > px >= 0 and h > py >= 0:
                            if im[py, px] == 255:
                                b[i] = 1
                            elif im[py, px] == 128:
                                b[i] = -1
                    # 第一个条件
                    if b[0] != 1:
                        continue

                    # 条件二, 四周一定存在一个像素点为黑色
                    if abs(b[1]) + abs(b[3]) + abs(b[5]) + abs(b[7]) == 4:
                        continue

                    # 条件三， 不能是端点，中心点周围至少有两个以上白色像素点
                    near_sum = 0
                    for i in range(1, 9):
                        near_sum += abs(b[i])
                    if near_sum < 2:
                        continue

                    # 条件四，其实包含在条件三之中,表示不能删除端点
                    near_sum = 0
                    for i in range(1, 9):
                        if b[i] == 1:
                            near_sum += 1
                    if near_sum <= 0:
                        continue

                    # 条件五， 连通性检测
                    if self.__NC8(b) != 1:
                        continue

                    # 条件六，连通性保证？
                    near_sum = 0
                    for i in range(1, 9):
                        if b[i] != -1:
                            near_sum += 1
                        else:
                            b[i] = 0
                            if self.__NC8(b) == 1:
                                near_sum += 1
                            b[i] = -1
                    if near_sum != 8:
                        continue

                    im[y, x] = 128
                    del_list.append([y, x])  # 以空间换取时间复杂度
                    count += 1
                    # 反复遍历的过程仍存在大量的时间浪费

            # 遍历完毕，开始删除像素点
            if count != 0:
                for cord in del_list:
                    im[cord[0], cord[1]] = 0
        if reverse:
            im = np.array((255 - im), dtype="uint8")
        self._im = im

    def __scan_thin(self, src, table, vertical=True):
        h, w = src.shape
        next_pix = 1
        temp = 0
        for m in range(h if vertical else w):
            for n in range(w if vertical else h):
                # 下一点不细化
                if next_pix == 0:
                    next_pix = 1
                else:
                    if vertical:
                        try_sum = int(src[m, n - 1]) + int(src[m, n]) + int(src[m, n + 1]) if 0 < n < w - 1 else 1
                    else:
                        # 此时垂直扫描 ，交换索引大小
                        temp = m
                        m = n
                        n = temp
                        try_sum = int(src[m - 1, n]) + int(src[m, n]) + int(src[m + 1, n]) if 0 < m < h - 1 else 1
                    if src[m, n] == 0 and try_sum != 0:
                        a = [0] * 9
                        # 3x3大小扫描细化
                        for y in range(3):
                            for x in range(3):
                                if -1 < (m - 1 + y) < h and -1 < (n - 1 + x) < w and src[m - 1 + y, n - 1 + x] == 255:
                                    a[y * 3 + x] = 1
                        a_sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[
                            8] * 128
                        # 查表细化
                        src[m, n] = table[a_sum] * 255
                        if table[a_sum] == 1:
                            next_pix = 0
                    # 继续扫描前需要将索引交换回来
                    if not vertical:
                        m = temp

    def table_thin(self, src, num=10):
        table = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
                 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

        for _ in range(num):
            self.__scan_thin(src, table)
            self.__scan_thin(src, table, vertical=False)

    def thin(self):
        if self._mode == 'table':
            self.table_thin(self._im)
        else:
            self.Hilditch(self._im)
        return np.array(self._im, dtype='uint8')


class FeaturesCat(object):
    def __init__(self, src, orientation=None):
        self._im = src.copy()
        self._features = []
        self._orient = orientation
        self._offset = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1],
                                 [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

    # 从图片中截取中心矩形
    @staticmethod
    def get_box(src, x, y, dis=1):
        return src[y - dis:y + dis + 1, x - dis:x + dis + 1]

    # 去掉相邻断点
    @staticmethod
    def remove_break(points, dis=7):
        # 特征点排序
        ranged_points = np.array(sorted(points, key=lambda x: x[0]))
        count = 1
        while count != 0:
            count = 0
            new_points = []

            temp = np.zeros((len(ranged_points), 2))
            temp[0:len(ranged_points) - 1, :] = ranged_points[1:len(ranged_points), :]

            pointers = temp - ranged_points

            for i in range(len(ranged_points)):
                # 计算距离
                distance = np.sqrt(np.power(pointers[i][0], 2) + np.power(pointers[i][1], 2))
                # 距离过近， 属于断点， 只保留一个
                if distance < dis:
                    count += 1
                    continue
                else:
                    new_points.append(i)

            new_points = np.array(new_points)

            ranged_points = ranged_points[new_points]
        return ranged_points

    # 找出分叉点的分叉方向点
    def get_fork(self, x, y):
        points = []
        for i in range(8):
            by = y + self._offset[i][0]
            bx = x + self._offset[i][1]
            if self._im[by, bx] == 0:
                points.append((bx, by))
                if len(points) == 3:
                    break
        return np.array(points)

    # 检查所找出的端点的连续长度
    def check_connect(self, x, y, dis=7):
        index = 0
        # 找出分块连续点，交换中心
        for i in range(8):
            by = y + self._offset[i][0]
            bx = x + self._offset[i][1]
            if self._im[by, bx] == 0:
                index = i
                x = bx
                y = by
                break
        # 进行连续性验证
        for _ in range(dis):
            # 获取新的分块
            block = self.get_box(self._im, x, y)
            # 获取白点数量，只能有两个点连通
            block_sum = int(np.sum(block) / 255)
            if block_sum == 6:
                for i in range(8):
                    by = y + self._offset[i][0]
                    bx = x + self._offset[i][1]
                    if self._im[by, bx] == 0 and abs(i - index) != 4:
                        index = i
                        x = bx
                        y = by
                        break
            else:
                return False
        return True

    # 检查端点是否不在边缘
    def check_inner(self, x, y):
        h, w = self._im.shape
        if np.sum(self._im[:y, x]) == 255 * y or \
                np.sum(self._im[y + 1:, x]) == 255 * (h - y - 1) or \
                np.sum(self._im[y, :x]) == 255 * x or \
                np.sum(self._im[y, x + 1:]) == 255 * (w - x - 1):
            return False
        return True

    # 计算九宫格的CrossNumber，从而确定分叉点或者端点, 不除以2方便操作
    def double_cn(self, block):
        # 中心点坐标（1,1）
        center = np.ones(2, dtype='uint8')
        cn_sum = 0
        last = center + self._offset[0]
        for i in range(1, 9):
            cord = center + self._offset[i]
            cn_sum += int(abs(int(block[cord[0], cord[1]]) - int(block[last[0], last[1]])))
            last = cord
        return int(cn_sum / 255)

    def __get_points(self):
        end_points = []
        folk_points = []
        h, w = self._im.shape
        # 外框不进行检测
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self._im[y, x] == 0:
                    block = self.get_box(self._im, x, y)
                    # 等于2初步判断为端点
                    judge = self.double_cn(block)
                    if judge == 2:
                        if self.check_connect(x, y) and self.check_inner(x, y):
                            end_points.append((x, y))
                    # 分叉点检测
                    if judge == 6:
                        folk_points.append((x, y))
        # 移除端点中的断点
        end_points = self.remove_break(end_points)

        # 从方向场获取角度
        for point in end_points:
            angle = round(self._orient[point[1], point[0]] / np.pi * 180 - 90, 2)
            self._features.append(['坐标：' + str((point[0], point[1])), '类型：端点', '方向角:' + str(angle) + '°'])
        for point in folk_points:
            angles = []
            forks = self.get_fork(point[0], point[1])
            for fork in forks:
                angle = round(self._orient[fork[1], fork[0]] / np.pi * 180 - 90, 2)
                angles.append('方向角:' + str(angle) + '°')
            angles = '(' + angles[0] + ', ' + angles[1] + ', ' + angles[2] + ')'
            self._features.append(['坐标：' + str((point[0], point[1])), '类型：分叉点', angles])

        # 进行标记
        self._im = cv2.cvtColor(self._im, cv2.COLOR_GRAY2RGB)
        for point in end_points:
            cv2.circle(self._im, (point[0], point[1]), 3, (0, 0, 255), 1)
        for point in folk_points:
            cv2.circle(self._im, (point[0], point[1]), 3, (255, 0, 0), 1)

    def cat(self):
        self.__get_points()
        return self._im

    def get_features(self):
        return self._features


if __name__ == '__main__':
    # image = cv2.imread('./examples/101_3.tif', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (200, 200))
    # enhancer = Enhancer()
    # img = enhancer.enhance(image)
    # kernel = np.ones((2, 2), np.uint8)
    # temp = 255 - img
    # temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    # img = np.array(255 - temp, np.uint8)
    # thinner = Thinner(img)
    # thin = thinner.thin()
    # feature = FeaturesCat(thin, enhancer.get_orient())
    # out = feature.cat()
    # cv2.imshow('test1', image)
    # cv2.imshow('test2', img)
    # cv2.imshow('test3', thin)
    # cv2.imshow('tset4', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('测试完毕！')
