# -*- coding: utf-8 -*-
#
# import sys
#
# reload(sys)
# sys.setdefaultencoding('utf8')

"""
读取batch从正负样本文件夹分别读取，目的是为了在batch中调整正负样本的比例
加入validation，以及vali的tensorboard


注：用的时候注意下面这些设置
batchsize,  16
图片大小，512
图片resize cv2.INTER_LINEAR,
迭代次数，10万
save次数，一万 ->5000
损失函数，CE->FL
python环境（opencv）   3.3
learning rate：10-4

model_save_dir = os.path.join(proj_root_path, 'run_output/logo_run_output0221_FL/')
summary_save_dir = os.path.join(proj_root_path, 'train_summary/logo_train_summary0221_FL/')

"""

import tensorflow as tf
import os

from data_prepare import data_loader_clothes
from model_design import model_design_nn2 as model_design_nn
from model_design import calc_loss
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def calc_accuracy(labels, logits, name):
    with tf.variable_scope(name):
        elem_compare_res = tf.cast(tf.equal(labels, tf.argmax(logits, axis=-1, output_type=tf.int32)), dtype=tf.float32)
        acc = tf.reduce_mean(elem_compare_res)
    return acc


def main():
    # init directory
    if not os.path.exists(config.model_save_dir):
        os.mkdir(config.model_save_dir)
    if not os.path.exists(config.summary_save_dir):
        os.mkdir(config.summary_save_dir)

    # data_load
    Data_Gen = data_loader_clothes.DataGenerator_PN(data_dir_p=config.mask_data_dir,
                                                    data_dir_n=config.clothes_data_dir,
                                                    img_h=config.img_h,
                                                    img_w=config.img_w,
                                                    img_ch=config.img_ch,
                                                    batch_size=config.batch_size
                                                    )
    Data_Gen_val = data_loader_clothes.DataGenerator_PN(data_dir_p=config.mask_data_eval_dir,
                                                        data_dir_n=config.image_clothes_eval_dir,
                                                        img_h=config.img_h,
                                                        img_w=config.img_w,
                                                        img_ch=config.img_ch,
                                                        batch_size=config.batch_size
                                                        )
    Data_Gen.build_positive_data()
    Data_Gen.build_negative_data()
    Data_Gen_val.build_positive_data()
    Data_Gen_val.build_negative_data()
    generator = Data_Gen.next_batch()
    generator_val = Data_Gen_val.next_batch()
    print('!!config.mask_data_dir', config.mask_data_dir)
    print('!!config.clothes_data_dir', config.clothes_data_dir)
    print('!!config.mask_data_eval_dir', config.mask_data_eval_dir)
    print('!!config.image_clothes_eval_dir', config.image_clothes_eval_dir)

    # define train model
    is_training = tf.placeholder(dtype=tf.bool)
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.img_h, config.img_w, config.img_ch])
    labels = tf.placeholder(dtype=tf.int32, shape=[None])  # shape=(?, )
    logits = model_design_nn.cls_net(inputs=images, is_training=is_training, is_clothes=True)  # shape=(?, 5)

    # cls loss
    cls_loss = calc_loss.build_loss(logits=logits, labels=labels)  # shape=()  # 交叉熵
    # cls_loss = calc_loss.focal_loss_softmax(logits=logits, labels=labels, gamma=config.focal_loss_gamma)

    # l2_loss
    l2_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
    for scope_name in ['CNN_Module', 'FC_Module']:
        module_train_vars = tf.trainable_variables(scope=scope_name)
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(var) for var in module_train_vars])
        l2_loss += regularization_cost * config.l2_loss_lambda
    loss = cls_loss + l2_loss

    total_loss = tf.summary.scalar("Loss/0_total_loss", loss)
    cls_loss = tf.summary.scalar("Loss/1_cls_loss", cls_loss)
    l2_loss = tf.summary.scalar("Loss/2_l2_loss", l2_loss)

    acc = calc_accuracy(labels=labels, logits=logits, name='acc')
    train_acc_summary = tf.summary.scalar("Metrics/train_acc", acc)
    vali_acc_summary = tf.summary.scalar("Metrics/vali_acc", acc)

    # summary_op
    # summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.summary_save_dir)  # train_summary文件夹
    # test_writer = tf.summary.FileWriter(config.summary_save_dir)  # train_summary文件夹

    # train_op
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):  # 该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行。
        train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss=loss,
                                                                                       global_step=global_step)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)  # 保存3个模型
    ckpt_path = tf.train.latest_checkpoint(config.model_save_dir)  # 来自动获取最后一次保存的模型
    print('latest_checkpoint_path: ', ckpt_path)
    if ckpt_path is not None:
        saver.restore(sess, ckpt_path)
        prev_step = int(ckpt_path.split('-')[-1])
    else:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        prev_step = -1

    # train
    with sess.as_default():
        count_train = 0
        count_vali = 0
        for i in range(config.train_steps):
            if (i + 1) % 20 == 0:
                # validation
                _is_training = False
                _inputs, _outputs = next(generator_val)  # .__next__()
                _img_tensor = _inputs['images']
                _label_tensor = _outputs['labels']
                # print(_ant_tensor.shape)
                _loss, _vali_acc_summary = sess.run([loss, vali_acc_summary],  # 去掉了 train_op
                                                    feed_dict={
                                                        images: _img_tensor,
                                                        labels: _label_tensor,
                                                        is_training: _is_training
                                                    })
                print('val_step: ', prev_step + 1 + i, 'loss: ', _loss)
                writer.add_summary(_vali_acc_summary, prev_step + 1 + i)
                count_vali += 1
                writer.flush()
            else:
                _is_training = True
                _inputs, _outputs = next(generator)  # .__next__()
                _img_tensor = _inputs['images']
                _label_tensor = _outputs['labels']
                # print(_ant_tensor.shape)
                _loss, _, _total_loss, _cls_loss, _l2_loss, _train_acc_summary = sess.run([loss, train_op, total_loss,
                                                                                           cls_loss, l2_loss,
                                                                                           train_acc_summary],
                                                                                          feed_dict={
                                                                                              images: _img_tensor,
                                                                                              labels: _label_tensor,
                                                                                              is_training: _is_training
                                                                                          })
                print('step: ', prev_step + 1 + i, 'loss: ', _loss)
                writer.add_summary(_total_loss, prev_step + 1 + i)
                writer.add_summary(_cls_loss, prev_step + 1 + i)
                writer.add_summary(_l2_loss, prev_step + 1 + i)
                writer.add_summary(_train_acc_summary, prev_step + 1 + i)
                count_train += 1
                writer.flush()

                if i % config.save_n_iters == 0:  # 存model
                    saver.save(sess=sess,
                               save_path=os.path.join(config.model_save_dir, 'model.ckpt'),
                               global_step=global_step)


if __name__ == '__main__':
    main()
