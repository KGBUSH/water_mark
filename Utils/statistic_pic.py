# -*- coding: utf-8 -*-

"""
统计图片

"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def size(img_dir):
    fn_list = os.listdir(img_dir)
    results = np.zeros((fn_list.__len__(), 3))
    for idx, fn in enumerate(fn_list):
        fn_path = os.path.join(img_dir, fn)
        img_cv2 = cv2.imread(fn_path)
        h, w = img_cv2.shape[0:2]
        results[idx] = h, w, h/float(w)
    plt.hist(results[:, 0], 20)
    plt.title('h distribution')
    plt.show()
    plt.hist(results[:, 1], 20)
    plt.title('w distribution')
    plt.show()
    plt.hist(results[:, 2], 20)
    plt.title('h/w distribution')
    plt.show()


if __name__ == '__main__':
    imgs_dir = '/Users/apple/PycharmProjects/water_mark/data/test/0228_smallsamples_positive/at2020-02-28'
    size(imgs_dir)
