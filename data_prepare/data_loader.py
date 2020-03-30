# -*- coding: utf-8 -*-
import cv2
import os
import random
import numpy as np
import sys
sys.path.append('../')

import config
from bak.yt_original import cls_dict

def image_preprocess_by_normality(img_cv2):
    '''
    mean_BGR:  58.420655528964474 62.79212985074819 108.6818206309374
    std:  62.55949780042519
    '''
    mean = (58.42, 62.79, 108.68)
    std = 62.56
    img_data = np.asarray(img_cv2, dtype=np.float32)
    # img_data = img_data - mean
    img_data = img_data / 255.0 #std
    img_data = img_data.astype(np.float32)
    return img_data


def preprocess(img_path):
    img_cv2 = cv2.imread(img_path)
    # img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_NEAREST)
    img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_LINEAR_EXACT)
    # normalization
    img_data = image_preprocess_by_normality(img_cv2)  # [0, 1]
    return img_data


class DataGenerator:
    def __init__(self,
                 img_dir,
                 label_dir,
                 img_w,
                 img_h,
                 img_ch,
                 batch_size):

        self.img_h = img_h
        self.img_w = img_w
        self.img_ch = img_ch
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.batch_size = batch_size

        self.img_fn_list = []  # 这里装的是img的绝对路径
        self.n = 0  # number of images
        self.indexes = []
        self.cur_index = 0
        self.imgs = []
        self.texts = []

    ## samples
    def build_data(self):
        print("DataGenerator, build data ...")
        self.img_fn_list = os.listdir(self.img_dir)
        self.n = len(self.img_fn_list)
        print("sample size of current generator: ", self.n)
        self.indexes = list(range(self.n))
        random.shuffle(self.indexes)

    def next_sample(self):  ## index max -> 0
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        try:
            # load one image and its label
            img_idx = self.indexes[self.cur_index]
            img_fn = self.img_fn_list[img_idx]
            img_path = os.path.join(self.img_dir, img_fn)
            img_data = preprocess(img_path)
            # parse label
            label_path = os.path.join(self.label_dir, img_fn.split('.')[0] + '.txt')
            with open(label_path, 'r') as f:  # 之前yutao用newline py3
                lines = f.readlines()
                label_data = cls_dict.label_num_dict['BG']
                for line in lines:
                    line = line.strip()
                    # print('line: ', line)
                    if 'PL' in line:
                        level = line.split(',')[1].strip()
                        label_data = cls_dict.label_num_dict['{}_{}'.format('PL', level)]
                        break
                # print('class: ', label_data)
        except Exception as e:
            img_data, label_data = self.next_sample()
            # print(e)
            # exit(1)
        return img_data, label_data

    def next_batch(self):  ## batch size
        while True:
            X_data = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_ch], dtype=np.float32)
            Y_data = np.zeros([self.batch_size], dtype=np.int)

            for i in range(self.batch_size):
                img_data, label_data = self.next_sample()
                X_data[i] = img_data.astype(np.float)  # 这个astype没必要写了，上面有
                Y_data[i] = label_data

            # dict
            inputs = {
                'images': X_data,  # (bs, h, w, 1)
            }
            outputs = {
                'labels': Y_data,  # (bs)
            }
            yield (inputs, outputs)


def test_preprocess():
    img_path = '/data/data/train_20/images/2_frontage_9.jpg'

    input_mask, output_mask = preprocess(img_path)
    hot_map = cv2.applyColorMap(np.uint8(output_mask * 255.0), cv2.COLORMAP_JET)
    cv2.imshow('mask', hot_map)
    cv2.waitKey(0)


def load_test():
    # img_dir = '../data/cls_crop_result_on_real'
    img_dir = config.image_dir
    label_dir = config.label_dir
    train_data = DataGenerator(img_dir=img_dir,
                               label_dir=label_dir,
                               img_h=config.img_h,
                               img_w=config.img_w,
                               img_ch=config.img_ch,
                               batch_size=2)
    train_data.build_data()

    inputs, outputs = train_data.next_batch().__next__()
    images = inputs['images']
    labels = outputs['labels']
    print(images.shape)
    print(labels)
    # print(labels)
    hot_map = cv2.applyColorMap(np.uint8(images[0] * 100), cv2.COLORMAP_JET)
    cv2.imshow('mask', hot_map)
    cv2.waitKey(0)


if __name__ == "__main__":
    load_test()
