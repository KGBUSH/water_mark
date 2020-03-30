# -*- coding: utf-8 -*-


from .default import *

try:
    from .local import *
except ImportError:
    pass

"""
logo: 标签里面1 是穿工服； 2或者0是没穿
"""

img_h = 1024
# img_h = 256
img_w = int(img_h / 1.0)
img_ch = 3


learning_rate = 1e-4
l2_loss_lambda = 1e-5  # should be very small in under fitting period

# log
LOG_PATH = os.path.join(proj_root_path, 'data/log/')


print('## project_root_path: ', proj_root_path)
print('## model_save_dir: ', model_save_dir)
print('## summary_save_dir: ', summary_save_dir)
print('## clothes_data_dir: ', clothes_data_dir)
print('## batchSize =', batch_size, 'learning_rate = ', learning_rate)



