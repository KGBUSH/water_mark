# -*- coding: utf-8 -*-
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


import os

# 210 docker位置
docker_project_path = '/code/projects/water_mark'
proj_root_path = docker_project_path

# 除了京东到家以外
model_save_dir = os.path.join(proj_root_path, 'run_output/watermark_run_output0330/')
summary_save_dir = os.path.join(proj_root_path, 'train_summary/watermark_train_summary0330/')

# training
batch_size = 4
train_steps = 20000
save_n_iters = 5000

# loss
focal_loss_gamma = 1

########################## only京东到家 数据 ####################
# 训练
jd_data_dir = '/data/dengyang/watermark/5k_bg_pictures/at2020-03-27'  # 5000

# vali
jd_data_eval_dir = '/data/dengyang/watermark/2k_bg_pictures/at2020-03-27'  # 2000

########################## 有京东到家又有美团 自己造的假数据 ####################

# 训练
add_meituan_to_jd_data_dir = '/data/dengyang/watermark/water_make_train_data0329/images'  # 6000

# 验证
add_meituan_to_jd_data_eval_dir = '/data/dengyang/watermark/water_make_vali_data/images'  # 实际统计出来是2992
