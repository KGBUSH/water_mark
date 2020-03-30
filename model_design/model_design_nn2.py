# -*- coding: utf-8 -*-
import tensorflow as tf
from model_design import darknet_base
import config


def cls_net(inputs, is_training=True, is_clothes=True):
    with tf.variable_scope('CNN_Module'):
        darknet53_out = darknet_base.darknet_53(inputs=inputs, is_training=is_training)  # shape=(?, 16,12,1024)
        print('darknet53_base_out: ', darknet53_out.get_shape().as_list())

    with tf.variable_scope('FC_Module'):
        avg_pool_size = darknet53_out.get_shape().as_list()[1:3]  # shape=(16,12)

        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=avg_pool_size,  # (8, 8),
                                                    padding='valid'
                                                    )(darknet53_out)
        print('avg_pool: ', avg_pool.get_shape().as_list())  # [None, 1, 1, 1024]
        avg_pool_shape = avg_pool.get_shape().as_list()  # 全连接reshape用

        seq = tf.reshape(tensor=avg_pool, shape=[-1, avg_pool_shape[-1]], name='fm2seq')  # shape=(?, 1024)
        print('seq: ', seq.get_shape().as_list())

        fc1 = tf.keras.layers.Dense(units=256)(seq)

        if is_clothes:
            fc2 = tf.keras.layers.Dense(units=2)(fc1)  # 工服自拍目前是2分类
        else:
            fc2 = tf.keras.layers.Dense(units=len(config.cls_num_dict))(fc1)

        # net_out = tf.nn.softmax(fc2)  # 在tf.nn.sparse_softmax_cross_entropy_with_logits做
        return fc2


class ClothesNet:
    """
    仿 grad-cam.tensorflow project's vgg.py
    """

    def __init__(self, inputs):
        self.parameters = []
        self.layers = {}
        self.inputs = inputs

    def cls_net(self, is_training=True, is_clothes=False):
        with tf.variable_scope('CNN_Module'):
            darknet53_out = darknet_base.darknet_53(inputs=self.inputs, is_training=is_training)  # shape=(?,16,12,1024)
            self.layers['darknet53_out'] = darknet53_out
            print('darknet53_base_out: ', darknet53_out.get_shape().as_list())

        with tf.variable_scope('FC_Module'):
            avg_pool_size = darknet53_out.get_shape().as_list()[1:3]  # shape=(16,12)

            avg_pool = tf.keras.layers.AveragePooling2D(pool_size=avg_pool_size,  # (8, 8),
                                                        padding='valid'
                                                        )(darknet53_out)
            print('avg_pool: ', avg_pool.get_shape().as_list())  # [None, 1, 1, 1024]

            seq = tf.reshape(tensor=avg_pool, shape=[-1, 1024], name='fm2seq')  # shape=(?, 1024)
            print('seq: ', seq.get_shape().as_list())

            fc1 = tf.keras.layers.Dense(units=256)(seq)

            if is_clothes:
                fc2 = tf.keras.layers.Dense(units=2)(fc1)  # 目前是2分类
                self.layers['fc2'] = fc2
            else:
                fc2 = tf.keras.layers.Dense(units=len(config.cls_num_dict))(fc1)

            # net_out = tf.nn.softmax(fc2)
            return fc2


def main():
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, config.img_h, config.img_w, config.img_ch), name='inputs')
    net_out = cls_net(inputs, is_training=True)


if __name__ == '__main__':
    main()
