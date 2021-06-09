import cv2
import numpy as np
# from charCat import CNN, Tools


class CAT(object):
    @staticmethod
    def is_blue(plate_gray):
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(plate_gray, kernel, iterations=1)
        return np.mean(img) < 50

    @staticmethod
    def get_test_img(image):
        h, w = image.shape
        if h > w:
            box = np.zeros((h, h), dtype='uint8')
            box[:, int((h - w) / 2):w + int((h - w) / 2)] = image
        else:
            box = np.zeros((w, w), dtype='uint8')
            box[int((w - h) / 2):h + int((w - h) / 2), :] = image
        if h > 720 or w > 720:
            image = cv2.resize(box, (720, 720), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(box, (720, 720), interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def get_mask(pred):
        mask = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 255
        return mask.astype(np.uint8)

    @staticmethod
    def get_plate(mask, image):
        plate = image
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((100, 100), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # 获取最小外接矩形
            plate = mask[y:y + h, x:x + w]
            if np.mean(plate) >= 50 and w > h > 15:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标
                box = sorted(box, key=lambda xy: xy[0])  # x较小的排前面
                box_left, box_right = box[:2], box[2:]  # 前后两个坐标分开
                box_left = sorted(box_left, key=lambda x: x[1])
                box_right = sorted(box_right, key=lambda x: x[1])
                box = np.array(box_left + box_right)  # [左上，左下，右上，右下]
                a1 = np.float32([box[0], box[1], box[2], box[3]])
                a2 = np.float32([(0, 0), (0, 100), (300, 0), (300, 100)])
                transform_mat = cv2.getPerspectiveTransform(a1, a2)  # 构成转换矩阵
                plate = cv2.warpPerspective(image, transform_mat, (300, 100))  # 进行车牌矫正
        return plate

    @staticmethod
    def vertical_cat(img):
        height, width = img.shape
        char_list = []
        white = []  # 记录每一列的白色像素总和
        black = []  # ..........黑色.......
        white_max = 0
        black_max = 0
        # 计算每一列的黑白色像素总和
        for i in range(width):
            s = 0  # 这一列白色总数
            t = 0  # 这一列黑色总数
            for j in range(height):
                if img[j][i] == 255:
                    s += 1
                if img[j][i] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)

        arg = False  # False表示白底黑字；True表示黑底白字
        if black_max > white_max:
            arg = True

        n = 1
        while n < width - 2:
            n += 1
            if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
                # 上面这些判断用来辨别是白底黑字还是黑底白字
                # 0.05这个参数请多调整，对应上面的0.95
                start = n
                end = start + 1
                for m in range(start + 1, width - 1):
                    if (black[m] if arg else white[m]) > (
                            0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
                        end = m
                        break
                n = end
                if 5 < end - start < 60:
                    cj = img[1:height, start:end]
                    char_list.append(cj)
        return char_list

    @staticmethod
    def horizon_cat(img):
        h, w = img.shape
        h_list = []
        start, end, cur_len = 0, 0, 0
        # 计算水平方向在垂直方向上的投影
        for i in range(h):
            count = 0
            for j in range(w):
                if img[i, j] == 255:
                    count += 1
            # 超过设定范围
            if count / w < 0.1 or count / w > 0.8:
                if cur_len != 0:
                    end = i - 1
                    if start != end:
                        h_list.append([start, end])
                    cur_len = 0
            # 正确投影
            if count > 0:
                if cur_len == 0:
                    start = i
                cur_len += 1

        # 循环结束后，最后一段投影可能没有结束
        if cur_len != 0:
            end = h - 1
            h_list.append([start, end])
        h_list = np.array(h_list)
        # 找出投影方向最长的部分
        max = (h_list[:, 1] - h_list[:, 0]).max()
        max_index = (h_list[:, 1] - h_list[:, 0]).argmax()
        if max / h > 0.5:
            return img[h_list[max_index][0]:h_list[max_index][1] + 1, :]
        return img

    @staticmethod
    def box_char(imgs):
        box_chars = []
        for char in imgs:
            h, w = char.shape
            box = np.zeros((h, h), dtype='uint8')
            box[:, int((h - w) / 2):int((h - w) / 2) + w] = char

            char = cv2.resize(box, (20, 20), None, interpolation=cv2.INTER_AREA)
            box_chars.append(char)
        return box_chars

    @staticmethod
    def char_cat(plate):
        flag = False
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.GaussianBlur(plate, (3, 3), 0)
        _, image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)
        img = cv2.erode(img, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)

        # 绿牌黄牌需要转化灰度值

        if not CAT.is_blue(img):
            image = 255 - image

        image = CAT.horizon_cat(image)
        image = CAT.vertical_cat(image)
        # 识别成功
        if 10 > len(image) > 1:
            img = CAT.box_char(image)
            flag = True
        # 如果识别失败，返回的是车牌原图
        return img, flag


if __name__ == '__main__':
    # 已全部调试完毕，无需运行
    pass
