import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

import warnings
warnings.filterwarnings('ignore')
from time import time
import random

from models import mf
from models.mf import MF
from util.loaddata import Data
import multiprocessing
import heapq
import util.metrics as metrics

def set_seed(seed=1024):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

def get_device():
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    return device

if __name__ == '__main__':

    set_seed()
    device = get_device()

    filepath = 'Data/movielens-1m'
    data_generator = Data(path=filepath,mode=1)  # mode=1 设置为测试集模式
    mf.data_generator = data_generator   # batch_size=args.batch_size
    print("read complete")

    model = MF(data_generator.n_users, data_generator.n_items,
                embedding_size = 10, l2_reg_embedding=0.00001, device=device)
    model.fit(learning_rate=0.001, batch_size=2000, epochs=50, verbose=5, early_stop=False)

