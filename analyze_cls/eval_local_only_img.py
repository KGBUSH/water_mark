# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import shutil
import cv2

from model_design.model_design_nn2 import *
import config
from data_prepare import data_loader_clothes


def main():
    # 1. load model
    config.is_training = False
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.img_h, config.img_w, config.img_ch])
    net_out = cls_net(images, is_training=False, is_clothes=True)  # shape=(?, 2)
    scores = tf.keras.layers.Softmax()(net_out)  # shape=(?, 2)

    # 2. load checkpoint
    # 2.1 model地址
    model_save_dir = os.path.join(config.proj_root_path, 'run_output/watermark_run_output0330')
    print("config.model_save_dir", model_save_dir)

    # 2.2 load
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)
    ckpt_path = tf.train.latest_checkpoint(model_save_dir)
    print('latest_checkpoint_path: ', ckpt_path)
    if ckpt_path is not None:
        saver.restore(sess, ckpt_path)
    else:
        print('！！！！！！ckpt not exists, task over!')
        exit(0)
        # saver.restore(sess, '/Users/apple/PycharmProjects/wb_classifier/run_output_from210/model.ckpt-9001')

    # 3. load evaluation data
    image_eval_dir = '/Users/apple/PycharmProjects/water_mark/data/处理图片/cut无效区域'
    image_eval_dir = '/Users/apple/PycharmProjects/water_mark/data/tmp_some_samples'
    print('test data path : ', image_eval_dir)
    df_info = []
    img_fn_list = []
    dict_info = {}  # to save label
    for r, d, f in os.walk(image_eval_dir):
        for each_file in f:
            if '.csv' in each_file:
                # df = pd.read_csv(os.path.join(r, each_file), encoding='gb2312')
                # df.loc[df.validation_status == 2, 'validation_status'] = 0
                # df_info.append(df)
                pass
            elif '.jpg' in each_file or '.png' in each_file:
                img_fn_list.append(os.path.join(r, each_file))
            else:
                pass

    # 4. evaluation
    total_cnt = 0
    with sess.as_default():
        correct_cnt = 0
        err_cnt = 0
        for idx, img_fn in enumerate(img_fn_list):  # 一张图一张图的遍历
            # if idx > 10:
            #     break
            # print(idx, '----------------- img_fn: %s' % img_fn)
            try:
                img_data = data_loader_clothes.preprocess_pad(img_fn)
                # print('!! padding')
                # img_data = data_loader_clothes.preprocess_pad(img_fn)
                _images = np.expand_dims(img_data, axis=0)
                # import IPython; IPython.embed()
                # print('_images shape: ', images.shape)
            except Exception as e:
                print('### ERROR: fail to read img: ', img_fn)
                print(e)
                continue

            _scores = sess.run(scores, feed_dict={images: _images})
            print(img_fn.split('/')[-1], end="\t")
            print('sccore: ', _scores)
            total_cnt += 1
            if _scores[0, 0] < 0.5:
                correct_cnt += 1
            # plt.imshow(_images[0, :, :, ::-1])
            # plt.title(str(_scores))
            # plt.show()
    print('total=', total_cnt, 'correct=', correct_cnt)


if __name__ == '__main__':
    main()
