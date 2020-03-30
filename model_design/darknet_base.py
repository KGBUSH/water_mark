# coding: utf-8

import tensorflow as tf
from model_design import refix_batch_norm


def darknet_53(inputs, is_training=True):
    def conv2d(inputs, filters, kernel_size, strides, name=''):
        def _fixed_padding(inputs, kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
            return padded_inputs

        if strides > 1:
            inputs = _fixed_padding(inputs, kernel_size)

        conv = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=('same' if strides == 1 else 'valid'),
                                      name=name
                                      )(inputs)
        bn = refix_batch_norm.my_batch_norm(inputs=conv, is_training=is_training)

        act = tf.keras.activations.relu(x=bn, alpha=0.1)
        return act

    def res_block(inputs, name):
        n_channel = inputs.get_shape().as_list()[-1]
        res_block_shortcut = inputs
        res_block_conv1 = conv2d(inputs=inputs,
                                 filters=int(n_channel / 2),
                                 kernel_size=1,
                                 strides=1,
                                 name='%s_conv1' % name)
        res_block_conv2 = conv2d(inputs=res_block_conv1,
                                 filters=n_channel,
                                 kernel_size=1,
                                 strides=1,
                                 name='%s_conv2' % name)
        res_block_out = tf.add(res_block_conv2, res_block_shortcut)
        return res_block_out

    # ------------------- part1 ------------------
    conv1_1 = conv2d(inputs=inputs,
                     filters=32,
                     kernel_size=3,
                     strides=1,
                     name='conv1_1')  # (?, 256, 192, 32)  这个注释对应的256*192的img size

    conv1_2 = conv2d(inputs=conv1_1,
                     filters=64,
                     kernel_size=3,
                     strides=2,
                     name='conv1_2')  # Tensor("CNN_Module/LeakyRelu_1:0", shape=(?, 128, 96, 64), dtype=float32)

    res_block1_1 = res_block(inputs=conv1_2,
                             name='res_block1_1')

    # ----------------- part2 ------------------
    conv2_1 = conv2d(inputs=res_block1_1,
                     filters=128,
                     kernel_size=3,
                     strides=2,
                     name='conv2_1')

    block = conv2_1
    for block_idx in range(2):
        block = res_block(inputs=block,
                          name='res_block2_%d' % (block_idx + 1))

    # ----------------- part3 ------------------
    conv3_1 = conv2d(inputs=block,
                     filters=256,
                     kernel_size=3,
                     strides=2,
                     name='conv3_1')

    block = conv3_1
    for block_idx in range(8):
        block = res_block(inputs=block,
                          name='res_block3_%d' % (block_idx + 1))

    route1 = block

    # ----------------- part4 -----------------
    conv4_1 = conv2d(inputs=block,
                     filters=512,
                     kernel_size=3,
                     strides=2,
                     name='conv4_1')  # Tensor("CNN_Module/LeakyRelu_26:0", shape=(?, 16, 12, 512), dtype=float32)

    block = conv4_1
    for block_idx in range(8):
        block = res_block(inputs=block,
                          name='res_block4_%d' % (block_idx + 1))
    route2 = block

    # ----------------- part5 -----------------
    conv5_1 = conv2d(inputs=block,
                     filters=1024,
                     kernel_size=3,
                     strides=2,
                     name='conv5_1')  # Tensor("CNN_Module/LeakyRelu_43:0", shape=(?, 8, 6, 1024), dtype=float32)

    block = conv5_1
    for block_idx in range(4):
        block = res_block(inputs=block,
                          name='res_block5_%d' % (block_idx + 1))

    return block

#
# def yolo_block(inputs, filters):
#     net = conv2d(inputs=inputs, filters=filters * 1, kernel_size=1, strides=1)
#     net = conv2d(inputs=net, filters=filters * 2, kernel_size=3, strides=1)
#     net = conv2d(inputs=net, filters=filters * 1, kernel_size=1, strides=1)
#     net = conv2d(inputs=net, filters=filters * 2, kernel_size=3, strides=1)
#     net = conv2d(inputs=net, filters=filters * 1, kernel_size=1, strides=1)
#     route = net
#     net = conv2d(inputs=net, filters=filters * 2, kernel_size=3, strides=1)
#     return route, net
#



def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    # tf.image.resize_bilinear(images=inputs, size=(new_height, new_width), align_corners=True, name='upsampling')
    return inputs


def main():
    inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    darknet53_out = darknet_53(inputs=inputs)
    print(darknet53_out.get_shape().as_list())



if __name__ == '__main__':
    main()
