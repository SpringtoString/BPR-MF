# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from collections import OrderedDict, namedtuple, defaultdict
import random
import multiprocessing
import heapq
import time
import sys
from prettytable import PrettyTable
sys.path.append('../')
import util.metrics as metrics


class MF(nn.Module):

    def __init__(self, n_user, n_item,
                 embedding_size=4, l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 seed=1024,):

        super(MF, self).__init__()
        self.model_name = 'mf'
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_size = embedding_size
        self.l2_reg_embedding = l2_reg_embedding
        self.embedding_dict = nn.ModuleDict({
            'user_emb': self.create_embedding_matrix(n_user, embedding_size),
            'item_emb': self.create_embedding_matrix(n_item, embedding_size),
        })

    def forward(self, input_dict):
        '''

        :param input_dict:
        :return:   rui, ruj
        '''
        users, pos_items, neg_items = input_dict['users'], input_dict['pos_items'], input_dict['neg_items']

        user_vector = self.embedding_dict['user_emb'](users)
        pos_items_vector = self.embedding_dict['item_emb'](pos_items)
        neg_items_vector = self.embedding_dict['item_emb'](neg_items)

        rui = torch.sum(torch.mul(user_vector, pos_items_vector), dim=-1, keepdim=True)
        ruj = torch.sum(torch.mul(user_vector, neg_items_vector), dim=-1, keepdim=True)

        emb_loss = torch.norm(user_vector) ** 2 + \
                   torch.norm(pos_items_vector) ** 2 + torch.norm(neg_items_vector) ** 2

        return rui, ruj, emb_loss

    def rating(self, user_batch, all_item):
        user_vector = self.embedding_dict['user_emb'](user_batch)
        pos_items_vector = self.embedding_dict['item_emb'](all_item)
        return torch.mm(user_vector, pos_items_vector.t())

    def create_embedding_matrix(self, vocabulary_size, embedding_size, init_std=0.0001, sparse=False,):
        embedding = nn.Embedding(vocabulary_size, embedding_size, sparse=sparse)
        nn.init.normal_(embedding.weight, mean=0, std=init_std)
        return embedding