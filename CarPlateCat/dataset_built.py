import os
import cv2
# import numpy as np


def delete_png(dir='./data/locate_data/',
               start=1, delete_list=None):  # 删除特定不合格图片
    if delete_list is None:
        delete_list = [58, 59, 60, 61, 62, 63]
    for i in range(len(delete_list)):  # 不破坏顺序
        delete_list[i] -= i
    for i in delete_list:
        for j in range(start, len(os.listdir(dir)) + start):
            if j == i:
                os.remove(dir + str(i) + '.png')
                continue
            if j > i:
                os.rename(dir + str(j) + '.png', dir + str(j - 1) + '.png')


def rename_all(dir='./data/locate_data/raw/', i=1):  # 目录下文件按索引重命名
    for new, name in enumerate(os.listdir(dir)):
        os.rename(dir + name, dir + str(new + i) + '.png')


def get_model_image(dir='./data/locate_data/raw/',
                    dir_tp='./data/locate_data/train/'):  # 图片裁剪720X720
    for i, name in enumerate(os.listdir(dir)):
        img = cv2.imread(dir + name)
        img = img[110:830, :, :]
        cv2.imwrite(dir_tp + name, img)


def match_and_delete(label_dir='./data/locate_data/label/',
                     train_dir='./data/locate_data/train/',
                     out='./data/locate_data/temp/'):
    for new, name in enumerate(os.listdir(label_dir)):  # 数据与标签匹配排序
        img_name = name.split('.')[0] + '.png'
        os.rename(label_dir + name, out + 'label/' + str(new + 1) + '.json')
        os.rename(train_dir + img_name, out + 'train/' + str(new + 1) + '.png')


def dataset_generate(root='./data/locate_data/'):  # json文件转化为mask标注与训练图片
    json_dir = root + 'label/'
    data_save = root + 'temp/'
    for json in os.listdir(json_dir):
        pair = json.split('.')[0]
        save_dir = data_save + pair + '_json'
        execute = 'labelme_json_to_dataset ' + json_dir + json + ' -o ' + save_dir
        os.system(execute)
        mask = cv2.imread(save_dir + 'label.png', cv2.IMREAD_GRAYSCALE)
        mask = mask / (mask.max()) * 255
        cv2.imwrite(root + 'mask/' + pair + '.png', mask)
        img = cv2.imread(save_dir + 'img.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(root + 'train_g/' + pair + '.png', img)  # 变为灰度图像加快训练速度
        os.rename(save_dir + 'img.png', root + 'train/' + pair + '.png')  # 转移图片


if __name__ == '__main__':
    # 已全部调试完毕，无需运行
    pass
