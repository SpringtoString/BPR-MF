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
from trainers import basetrainer
from trainers.basetrainer import BaseTrainer
from models.mf import MF
from util.loaddata import Data
from util.utils import set_seed , get_device
import multiprocessing
import heapq
import util.metrics as metrics


if __name__ == '__main__':

    set_seed()
    device = get_device(use_cuda=True, gpu_id=0)

    filepath = 'Data/movielens-1m'
    data_generator = Data(path=filepath,mode=1)  # mode=1 设置为测试集模式
    basetrainer.data_generator = data_generator           # batch_size=args.batch_size
    print("read complete")

    model = MF(data_generator.n_users, data_generator.n_items,embedding_size = 10, l2_reg_embedding=0.025)
    trainer = BaseTrainer(model, lr=0.001, batch_size=1024, epochs=100, verbose=5, save_round=100, early_stop=True, device=device)
    trainer.fit()

