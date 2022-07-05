import os
import sys
from pathlib import Path
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 模型相关参数
n_step = 5  # number of steps, n-1 in paper
n_hidden = 5  # number of hidden size, h in paper
emb_size = 128  # embedding size, m in paper

# 训练相关参数
epochs = 200
batch_size = 512
lr = 0.001

save_epoch = 50  # 每多少轮保存一次模型
models_path = ROOT / 'weigths'  # 模型保存路径
vocab_path = ROOT / 'vocab.json'  # 单词表保存路径
test_model = ROOT / 'weights' / 'weights-50.ckpt'  # 测试时使用模型

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集路径
train_path = 'data/train.txt'
valid_path = 'data/valid.txt'
test_path = 'data/test.txt'
# train_path = '/content/drive/MyDrive/data_nlp/data/train.txt'
# valid_path = '/content/drive/MyDrive/data_nlp/data/valid.txt'
# test_path = '/content/drive/MyDrive/data_nlp/data/test.txt'