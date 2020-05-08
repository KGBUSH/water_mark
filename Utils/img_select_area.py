# -*- coding: utf-8 -*-
"""
吴圆周给的营业执照很多是手机截屏，上下有些多余的部分需要切掉
e.g. /Users/apple/PycharmProjects/wb_classifier/data/water_mark/small_samples/11.jpg
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm


def run(input_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # config
    black = 20  # 设置小于20就是黑色了

    fn_list = os.listdir(input_dir)
    fn_list = [os.path.join(input_dir, fn) for fn in fn_list]
    for fn_path in tqdm(fn_list):

        # load img
        # path = '/Users/apple/PycharmProjects/wb_classifier/data/water_mark/small_samples/9_1.jpg'
        path = fn_path
        img_cv2 = cv2.imread(path, cv2.IMREAD_COLOR)

        # 原图
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        plt.imshow(img_cv2)
        plt.show()
        a0 = img_cv2[:, :, 0]
        a1 = img_cv2[:, :, 1]
        a2 = img_cv2[:, :, 2]

        vertical_line = img_cv2[:, img_cv2.shape[1] // 2, 0]
        vertical_line = img_cv2[:, 0, 0]
        middle_cut = img_cv2.shape[0] // 2
        up_boundary = None
        down_boundary = None
        for i in range(middle_cut, -1, -1):
            if vertical_line[i] < black and vertical_line[i - 1] < black:
                up_boundary = i
                break

        for i in range(middle_cut, img_cv2.shape[0] - 1):
            if vertical_line[i] < black and vertical_line[i + 1] < black:
                down_boundary = i
                break

        # 颜色分布
        # plt.scatter(np.arange(vertical_line.shape[0]), vertical_line)
        # plt.show()

        # 切割无效区域之后的图片
        # plt.imshow(img_cv2[up_boundary:down_boundary, :, :])
        # plt.show()

        img_save_path = os.path.join(out_dir,
                                     fn_path.split('/')[-1])
        cv2.imwrite(img_save_path, img_cv2[up_boundary:down_boundary, :, :])


if __name__ == '__main__':
    run(input_dir='/Users/apple/PycharmProjects/water_mark/data/处理图片/处理图片',
        out_dir='/Users/apple/PycharmProjects/water_mark/data/处理图片/cut无效区域')
