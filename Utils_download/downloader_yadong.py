# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import multiprocessing
import pandas as pd
from concurrent import futures
from requests_futures.sessions import FuturesSession


PROCESSES = 10
THREADS = 10


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
        for map_id, image_url in url_dict.items():
            future = session.get(
                url=image_url,
                timeout=10,
                background_callback=get_background_callback(
                    map_id=map_id,
                    output_dir=output_dir,
                )
            )
            future_list.append(future)

    futures.wait(future_list)


def download(source_filename, output_dir, fields):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(source_filename, sep='\t', encoding='utf8')
    url_items = df[[fields['key'], fields['value']]].values.tolist()

    pool = multiprocessing.Pool(processes=PROCESSES)
    offset, batch_size = 0, 1000
    while True:
        _url_items = url_items[offset:offset + batch_size]
        if not _url_items:
            break
        offset += batch_size
        pool.apply_async(download_batch, (dict(_url_items), output_dir))
    pool.close()
    pool.join()


if __name__ == '__main__':
    # source_filename = '/home/huyadong/data/clothes_detection/clothes_detection_data_20190823.csv'
    # output_dir = '/home/huyadong/data/clothes_detection/selfie_images/'
    # download(
    #     source_filename=source_filename,
    #     output_dir=output_dir,
    #     fields={
    #         'key': 'selfie_id',
    #         'value': 'image_url',
    #     }
    # )

    source_filename = '/home/huyadong/data/clothes_detection/clothes_detection_id_card_20190823.csv'
    output_dir = '/home/huyadong/data/clothes_detection/id_card_images/'
    download(
        source_filename=source_filename,
        output_dir=output_dir,
        fields={
            'key': 'transporter_id',
            'value': 'id_card_image_url',
        }
    )
