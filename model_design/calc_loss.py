import tensorflow as tf
import numpy as np
import os
import cv2

import sys

sys.path.append('../')

import config


def build_loss(logits, labels):
    """
    :param logits: # shape=(?, n_class)
    :param labels: # shape=(?, )
    :return:
    """
    logits = tf.expand_dims(input=logits, axis=-2)
    labels = tf.expand_dims(input=labels, axis=-1)
    print('logits: ', logits.get_shape().as_list())  # ('logits: ', [None, 1, 2])
    print('labels: ', labels.get_shape().as_list())  # ('labels: ', [None, 1])
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    )
    print('ce_loss: ', cross_entropy_loss.get_shape().as_list())  # ('ce_loss: ', [None, 1])
    loss = tf.reduce_mean(cross_entropy_loss)
    return loss


def focal_loss_softmax(logits, labels, gamma=2):
    """
    Computer focal loss for multi classification
    When gamma = 0, focal loss is equivalent to categorical cross-entropy,
    and as gamma is increased the effect of the modulating factor is likewise increased
    (gamma = 2 works best in experiments).

    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(logits, axis=-1)  # [batch_size,num_classes]
    # To avoid divided by zero
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)

    labels = tf.one_hot(labels, depth=y_pred.shape[1])  # shape=(5,4)
    loss = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    loss = tf.reduce_sum(loss)
    return loss


if __name__ == '__main__':
    logits = tf.placeholder(dtype=tf.float32, shape=[3, 2])
    labels = tf.placeholder(dtype=tf.int32, shape=[3])

    loss = build_loss(logits, labels)

    #
    # print('loss: ', loss.get_shape().as_list())
    # loss = tf.reduce_sum(loss) * 2e-6

    print('loss: ', loss.get_shape().as_list())
