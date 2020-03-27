# -*- coding: utf-8 -*-


"""
该脚本用于本地下载
下载人工认为是正样本的图片
"""


from __future__ import print_function

import os
# from tqdm import tqdm
import pandas as pd
from concurrent import futures
from requests_futures.sessions import FuturesSession
import time
import config

PROCESSES = 20
THREADS = 20


def get_background_callback(map_id, output_dir):
    def background_callback(session, response):
        if response.status_code != 200:
            print('fail: %s' % map_id)
        filename = os.path.join(output_dir, '%s.jpg' % map_id)
        with open(filename, 'wb') as f:
            f.write(response.content)

    return background_callback


def download_batch(url_dict, output_dir):
    future_list = []
    with FuturesSession(max_workers=THREADS) as session:
        i = 0
        for map_id, image_url in url_dict.items():
            if i % 100 == 0:
                print(i)
            future = session.get(
                url=image_url,
                timeout=10,
                background_callback=get_background_callback(
                    map_id=str(map_id),
                    output_dir=output_dir,
                )
            )
            future_list.append(future)
            i += 1

    futures.wait(future_list)


def download(source_filename, output_dir, output_csv):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # df = pd.read_csv(source_filename)
    df = pd.read_excel(source_filename)

    # 筛选出部分图片
    df = df.iloc[-num:, :]
    print('筛选出部分图片数量:', num, df.shape)
    df['id'] = range(df.shape[0])
    id = 'id'  # 第一列是图片id，但是列名可能不一样

    # df['id_maskType'] = df[id].astype('str') + '_' + df['口罩状态'].astype('str')
    # df['id_maskType'] = df[id].astype('str') + '_' + df['type'].astype('str')

    url_items = df[[id, 'qualify_url']].values.tolist()
    # url_items = df[['selfie_id', 'selfie_img_url']].values.tolist()

    download_batch(url_dict=dict(url_items), output_dir=output_dir)  # 多进程的包有问题 用单线程
    df.to_csv(output_csv, index=False)

    print('vali1 Finish: images number is {}, and saved csv shape is {}'.format(len(os.listdir(output_dir)),
                                                                                  df.shape))


if __name__ == '__main__':

    dir = os.path.join('/data/dengyang/watermark', '10w_pictures')
    print(config.proj_root_path)
    this_path = os.path.split(os.path.realpath(__file__))[0]

    # 正样本的csv
    date = time.strftime("%Y-%m-%d", time.localtime())
    num = 20
    source_filename = os.path.join(config.proj_root_path, 'utils_download/excel_wuyuanzhou/10w_pictures.xlsx')  # hive导出的selfie表

    # 图片下载位置，已经每张图片对应的item写到了一张csv表
    output_dir = os.path.join(dir, 'at{}/'.format(date))
    output_csv = os.path.join(dir, 'at{}__.csv'.format(date))
    print("样本的csv:", source_filename)
    print("output to :", output_dir)

    download(
        source_filename=source_filename,
        output_dir=output_dir,
        output_csv=output_csv
    )
