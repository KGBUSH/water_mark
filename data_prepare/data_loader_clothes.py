# -*- coding: utf-8 -*-
"""
工服自拍
正样本和"没有穿着装备"的负样本
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd
import numpy as np
import sys
from collections import Counter

sys.path.append('../')
import config


def image_preprocess_by_normality(img_cv2):
    '''
    mean_BGR:  58.420655528964474 62.79212985074819 108.6818206309374
    std:  62.55949780042519
    '''
    mean = (58.42, 62.79, 108.68)
    std = 62.56
    img_data = np.asarray(img_cv2, dtype=np.float32)
    # img_data = img_data - mean
    img_data = img_data / 255.0  # std
    img_data = img_data.astype(np.float32)
    return img_data


def preprocess(img_path):
    img_cv2 = cv2.imread(img_path)
    # img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_NEAREST)
    img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_LINEAR_EXACT)
    # normalization
    img_data = image_preprocess_by_normality(img_cv2)  # [0, 1]
    return img_data


def preprocess2(img_path):
    """
    using cv2.INTER_LINEAR
    因为线上的Opencv的版本是老版本
    """
    img_cv2 = cv2.imread(img_path)
    # img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_NEAREST)
    img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_LINEAR)
    # normalization
    img_data = image_preprocess_by_normality(img_cv2)  # [0, 1]
    return img_data


def preprocess_pad(img_path):
    """
    长边保持到config中的img_w，或者img_h（img_w == img_h）
    短边等比缩放，然后padding
    """
    try:
        img_cv0 = cv2.imread(img_path)
        # 1. 确定图片的长边和短边，然后把长边resize到224，保持纵横比的情况下resize短边
        assert config.img_w == config.img_h
        h, w, _ = img_cv0.shape
        m = max(w, h)
        ratio = 1.0 * config.img_w / m
        new_w, new_h = int(ratio * w), int(ratio * h)
        assert new_w > 0 and new_h > 0
        # 使用cv2.resize时，参数输入是 宽×高×通道
        # img_cv1 = cv2.resize(img_cv0, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR_EXACT)
        img_cv1 = cv2.resize(img_cv0, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 2. 把图片进行填充，填充到256 256
        W, H = config.img_w, config.img_w
        top = (H - new_h) // 2
        bottom = (H - new_h) // 2
        if top + bottom + new_h < H:
            bottom += 1
        left = (W - new_w) // 2
        right = (W - new_w) // 2
        if left + right + new_w < W:
            right += 1
        pad_image = cv2.copyMakeBorder(img_cv1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if pad_image.shape[0] != config.img_h or pad_image.shape[1] != config.img_w:
            print('!!!image padding error:', pad_image.shape)
        return pad_image
    except:
        raise cv2.error


class DataGeneratorClothes(object):
    def __init__(self,
                 data_dir,
                 img_w,
                 img_h,
                 img_ch,
                 batch_size):

        self.img_h = img_h
        self.img_w = img_w
        self.img_ch = img_ch
        self.data_dir = data_dir
        self.batch_size = batch_size
        print('batch_size={}, img_h={}, img_w={}, img_ch={}'.format(batch_size, img_h, img_w, img_ch))

        self.img_fn_list = []  # 这里装的是img的绝对路径
        self.df_info = []
        self.dict_info = {}
        self.n = 0  # number of images， 这个后面很快就赋值为样本总数了
        self.indexes = []
        self.cur_index = 0
        self.imgs = []
        self.texts = []

    ## samples
    def build_data(self):
        print("DataGenerator, build data ...")
        for r, d, f in os.walk(self.data_dir):
            for each_file in f:
                if '.csv' in each_file:
                    df = pd.read_csv(os.path.join(r, each_file), encoding='gb2312')
                    # df['validation_status'][df['validation_status'] == 2] = 0  # 把原来是2的，全部改成0
                    df.loc[df.validation_status == 2, 'validation_status'] = 0
                    self.df_info.append(df)
                else:
                    self.img_fn_list.append(os.path.join(r, each_file))
        self.df_info = pd.concat(self.df_info, axis=0, ignore_index=True)
        self.df_info = self.df_info.loc[:, ['id', 'validation_status']]
        for i in range(self.df_info.shape[0]):
            self.dict_info.update({self.df_info.loc[i, 'id']: self.df_info.loc[i, 'validation_status']})

        self.n = len(self.img_fn_list)
        # print("sample size of current generator: ", self.n)
        print("sample size of current generator={}, labels={} ".format(self.dict_info.__len__(),
                                                                       Counter(self.dict_info.values())
                                                                       ))
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
            img_path = self.img_fn_list[img_idx]
            # img_data = preprocess(img_path)
            img_data = preprocess2(img_path)
            # img_data = preprocess_pad(img_path)
            # parse label
            img_id = int(img_path.split('/')[-1].split('.')[0])  # id is int
            # label_data = self.df_info[self.df_info['id'] == img_id]['validation_status']
            label_data = self.dict_info[img_id]

        except (ValueError, KeyError, cv2.error) as e:
            print('!!!! wrong image:', img_path)
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


class DataGeneratorClothes2(DataGeneratorClothes):
    """
    读取文件夹中多个类别的图片
    e.g.
    clothes_vali2_0101to1225_PhotoNotRight_at12272020-01-02/
    clothes_vali2_0101to1225_PhotoNotRight_at1227.csv
    clothes_vali2_0101to1225_reasonDontWearEquipment_at1227_at2020-01-02/
    clothes_vali2_0101to1225_reasonDontWearEquipment_at1227__.csv
    vali1_1101to1218_at2020-01-02/
    vali1_1101to1218_at2020-01-02__.csv
    """

    def __init__(self,
                 data_dir,
                 img_w,
                 img_h,
                 img_ch,
                 batch_size):
        super(DataGeneratorClothes2, self).__init__(data_dir,
                                                    img_w,
                                                    img_h,
                                                    img_ch,
                                                    batch_size)

    def build_data(self):
        """
        重写
        :return:
        """
        print("DataGenerator, build data ...")
        for r, d, f in os.walk(self.data_dir):
            for each_file in f:
                if '.csv' in each_file:
                    df = pd.read_csv(os.path.join(r, each_file), encoding='gb2312')
                    df['validation_status'][df['validation_status'] == 2] = 0  # 把原来是2的，全部改成0
                    self.df_info.append(df)
                else:
                    self.img_fn_list.append(os.path.join(r, each_file))
        self.df_info = pd.concat(self.df_info, axis=0, ignore_index=True)
        self.df_info = self.df_info.loc[:, ['id', 'validation_status']]
        for i in range(self.df_info.shape[0]):
            self.dict_info.update({self.df_info.loc[i, 'id']: self.df_info.loc[i, 'validation_status']})

        self.n = len(self.img_fn_list)
        print("sample size of current generator: ", self.n)
        self.indexes = list(range(self.n))
        random.shuffle(self.indexes)


class DataGenerator_PN(object):
    """
    用于mask
    后面也改成可以用于logo
    """

    def __init__(self,
                 data_dir_p,
                 data_dir_n,
                 img_w,
                 img_h,
                 img_ch,
                 batch_size):

        self.img_h = img_h
        self.img_w = img_w
        self.img_ch = img_ch
        self.data_dir_p = data_dir_p
        self.data_dir_n = data_dir_n
        self.batch_size = batch_size
        print('data_dir_p={};    data_dir_n={}'.format(self.data_dir_p, self.data_dir_n))
        print('batch_size={}, img_h={}, img_w={}, img_ch={}'.format(batch_size, img_h, img_w, img_ch))

        self.img_fn_list_positive = []  # 这里装的是img的绝对路径
        self.n_positive = 0  # number of images， 这个后面很快就赋值为样本总数了
        self.indexes_positive = []
        self.cur_index_positive = 0

        self.img_fn_list_negative = []  # 这里装的是img的绝对路径
        self.n_negative = 0  # number of images， 这个后面很快就赋值为样本总数了
        self.indexes_negative = []
        self.cur_index_negative = 0

    # samples
    def build_positive_data(self):
        print("DataGenerator, build data ...")
        for r, d, f in os.walk(self.data_dir_p):
            for each_file in f:
                if '.jpg' in each_file or '.png' in each_file:
                    self.img_fn_list_positive.append(os.path.join(r, each_file))

        self.n_positive = len(self.img_fn_list_positive)
        print("positive sample size of current generator: ", self.n_positive)
        self.indexes_positive = list(range(self.n_positive))
        random.shuffle(self.indexes_positive)

    def build_negative_data(self):
        print("DataGenerator, build data ...")
        for r, d, f in os.walk(self.data_dir_n):
            for each_file in f:
                if '.jpg' in each_file or '.png' in each_file:
                    self.img_fn_list_negative.append(os.path.join(r, each_file))

        self.n_negative = len(self.img_fn_list_negative)
        print("negative sample size of current generator: ", self.n_negative)
        self.indexes_negative = list(range(self.n_negative))
        random.shuffle(self.indexes_negative)

    def next_positive_sample(self):  ## index max -> 0
        self.cur_index_positive += 1
        if self.cur_index_positive >= self.n_positive:
            self.cur_index_positive = 0
            random.shuffle(self.indexes_positive)
        try:
            # load one image and its label
            img_idx = self.indexes_positive[self.cur_index_positive]
            img_path = self.img_fn_list_positive[img_idx]
            # img_data = preprocess2(img_path)
            img_data = preprocess_pad(img_path)
            # parse label
            label_data = 1

        except (ValueError, KeyError, cv2.error) as e:
            print('!!!! wrong image:', img_path)
            img_data, label_data = self.next_positive_sample()
            # print(e)
            # exit(1)
        return img_data, label_data

    def next_negative_sample(self):  ## index max -> 0
        self.cur_index_negative += 1
        if self.cur_index_negative >= self.n_negative:
            self.cur_index_negative = 0
            random.shuffle(self.indexes_negative)
        try:
            # load one image and its label
            img_idx = self.indexes_negative[self.cur_index_negative]
            img_path = self.img_fn_list_negative[img_idx]
            # img_data = preprocess2(img_path)
            img_data = preprocess_pad(img_path)
            # parse label
            label_data = 0

        except (ValueError, KeyError, cv2.error) as e:
            print('!!!! wrong image:', img_path)
            img_data, label_data = self.next_negative_sample()
            # print(e)
            # exit(1)
        return img_data, label_data

    def next_batch(self):  ## batch size
        while True:
            X_data = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_ch], dtype=np.float32)
            Y_data = np.zeros([self.batch_size], dtype=np.int)

            for i in range(self.batch_size / 2):
                img_data, label_data = self.next_positive_sample()
                X_data[i] = img_data.astype(np.float)  # 这个astype没必要写了，上面有
                Y_data[i] = label_data

            for i in range(self.batch_size / 2, self.batch_size):
                img_data, label_data = self.next_negative_sample()
                X_data[i] = img_data.astype(np.float)  # 这个astype没必要写了，上面有
                Y_data[i] = label_data
            # dict
            inputs = {
                'images': X_data,  # (bs, h, w, 1)
            }
            outputs = {
                'labels': Y_data,  # (bs)
            }
            yield inputs, outputs


if __name__ == "__main__":
    # load_test()

    # Data_Gen = DataGeneratorClothes2(data_dir=config.clothes_data_dir,
    #                                  img_h=config.img_h,
    #                                  img_w=config.img_w,
    #                                  img_ch=config.img_ch,
    #                                  batch_size=config.batch_size
    #                                  )
    # Data_Gen.build_data()

    path = '/Users/apple/PycharmProjects/water_mark/data/图片(1)/2.jpg'
    path = '/Users/apple/PycharmProjects/water_mark/data/图片(1)/3.jpg'
    preprocess_pad(img_path=path)
