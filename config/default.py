# -*- coding: utf-8 -*-
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


import os

# 210 docker位置
docker_project_path = '/code/projects/wb_classifier'
proj_root_path = docker_project_path

# clothes
clothes_data_dir = os.path.join(proj_root_path, 'data/samples_at0108')  # clothes的数据, 正负样本各8万，19年全年的数据
model_save_dir = os.path.join(proj_root_path, 'run_output/logo_run_output0221_FL/')
summary_save_dir = os.path.join(proj_root_path, 'train_summary/logo_train_summary0221_FL/')

# mask
mask_model_save_dir = os.path.join(proj_root_path, 'run_output/run_output0209_for_CE/')
mask_summary_save_dir = os.path.join(proj_root_path, 'train_summary/train_summary0209_for_CE/')


# training
batch_size = 16
train_steps = 100000
save_n_iters = 5000


# loss
focal_loss_gamma = 1






########################## clothes 验证数据 ####################


# 0102的所有数据
image_clothes_eval_dir = os.path.join(proj_root_path, 'data/samples_all0102_at0109')
# 0102的所有数据，正负样本分开存放, 70526正样本，2200负样本
image_clothes_eval_dir = os.path.join(proj_root_path, 'data/samples_1226to0110_vali1and2_split')

# 把正负样本分开，目前目的是validation->tensorboard的时候均匀一点
image_clothes_eval_dir_vali1 = os.path.join(proj_root_path, 'data/samples_1226to0110_vali1and2_split/vali1')
image_clothes_eval_dir_vali2 = os.path.join(proj_root_path, 'data/samples_1226to0110_vali1and2_split/vali2')
###############################################################




########################## mask 数据 ####################
# 训练
mask_data_dir = os.path.join(proj_root_path, 'data/dada_mask/samples_at0207_1')
mask_data_dir = os.path.join(proj_root_path, 'data/dada_mask/samples_at0208_forTrain')

# vali
mask_data_eval_dir = os.path.join(proj_root_path, 'data/dada_mask/samples_at0208_forVali')
###############################################################