import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed=1024):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

def get_device(use_cuda=True, gpu_id=0):
    device = 'cpu'
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:%d'%(gpu_id)
    return device

class EarlyStopManager(object):

    def __init__(self, patience):
        self.patience = patience
        self.count = 0
        self.max_metric = 0

    def step(self, metric):

        if metric > self.max_metric:
            self.max_metric = metric
            self.count = 0
            return False
        else:
            self.count = self.count + 1
            if self.count > self.patience:
                return True
            return False