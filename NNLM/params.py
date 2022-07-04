import os
import torch


# 模型相关参数
n_step = 5  # number of steps, n-1 in paper
n_hidden = 2  # number of hidden size, h in paper
m = 2  # embedding size, m in paper


# 训练相关参数
epochs = 200
batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 数据集路径
train_path = ''
val_path = ''
test_path = ''
