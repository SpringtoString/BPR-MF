import numpy as np
import pandas as pd
import random as rd
import scipy.sparse as sp
from time import time
from collections import defaultdict


class Data(object):
    def __init__(self, path, batch_size=256, mode=0):
        '''

        :param path: 文件路径
        :param batch_size: sample的样本数
        :param mode: {0:验证,1:测试}
        '''
        self.path = path
        self.batch_size = batch_size
        train_file = path + '/train.txt'

        if mode==0:
            test_file = path + '/validation.txt'
        else:
            test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items= 0, 0
        self.n_train, self.n_test = 0, 0


        ''' 读取 user,item 交互记录 '''
        self.train_items, self.test_set = {}, {}  # userid to itemset

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

                    self.train_items[uid] = items

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    try:
                        l = l.strip('\n').split(' ')
                        uid = int(l[0])
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
                    self.test_set[uid] = items

        self.n_items += 1  # user、item都是从0开始编号，所以要 +1
        self.n_users += 1
        self.print_statistics()

        self.exist_users = [i for i in range(self.n_users)]
        self.all_item = [i for i in range(self.n_items)]

    def sample(self,batch_size=256):
        '''
        :return: [user id], [postiveitem id] [negativeitem id]
        '''
        if  batch_size <= self.n_users:
            users = rd.sample(self.exist_users, batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items


    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

