from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import collections
import os
import json
import re
import collections
import datetime
import pickle
import random
from tqdm import trange
import time

class data_batcher(object):
    def __init__(self, data_idx_list, data_idx_list_test, dic, seq_max, target_size):
        self.data_idx_list = data_idx_list
        self.data_idx_list_test = data_idx_list_test
        self.dic = dic
        self.voc_size = len(dic)
        self.seq_max = seq_max
        self.target_size = target_size
        
        self.train_size = len(self.data_idx_list)
        self.test_size = len(self.data_idx_list_test)
    
    
    def get_train_batch_rand(self, size):
        np.random.seed(seed=int(time.time()))
        assert size <= len(self.data_idx_list)

        data_x1 = np.zeros((size, self.seq_max), dtype=np.int)
        len_x1 = np.zeros(size, dtype=np.int)
        data_y = np.zeros(size, dtype=np.int)

        index = np.random.choice(range(len(self.data_idx_list)), size, replace=False)
        for a in range(len(index)):
            idx = index[a]

            s1 = self.data_idx_list[idx][0]
            label = self.data_idx_list[idx][1]

            x1 = s1 + [self.dic["<PAD>"]] * (self.seq_max - len(s1))
            y = label

            assert len(x1) == self.seq_max
            assert max(x1) < self.voc_size
            assert 0 <= y <= self.target_size

            data_x1[a] = x1
            data_y[a] = y
            len_x1[a] = len(s1)
        return data_x1, data_y, len_x1

    def get_test_batch_rand(self, size):
        np.random.seed(seed=int(time.time()))
        assert size <= len(self.data_idx_list_test)

        data_x1 = np.zeros((size, self.seq_max), dtype=np.int)
        data_y = np.zeros(size, dtype=np.int)
        len_x1 = np.zeros(size, dtype=np.int)

        index = np.random.choice(range(len(self.data_idx_list_test)), size, replace=False)
        for a in range(len(index)):
            idx = index[a]

            s1 = self.data_idx_list_test[idx][0]
            label = self.data_idx_list_test[idx][1]

            x1 = s1 + [self.dic["<PAD>"]] * (self.seq_max - len(s1))
            y = label

            assert len(x1) == self.seq_max
            assert max(x1) < self.voc_size
            assert 0 <= y <= self.target_size

            data_x1[a] = x1
            data_y[a] = y
            len_x1[a] = len(s1)
        return data_x1, data_y, len_x1
    
    
    def get_train_batch_step(self, start, size):
        assert start+size <= len(self.data_idx_list)

        data_x1 = np.zeros((size, self.seq_max), dtype=np.int)
        data_y = np.zeros(size, dtype=np.int)
        len_x1 = np.zeros(size, dtype=np.int)

        for a in range(size):
            s1 = self.data_idx_list[start+a][0]
            label = self.data_idx_list[start+a][1]

            x1 = s1 + [self.dic["<PAD>"]] * (self.seq_max - len(s1))
            y = label

            assert len(x1) == self.seq_max
            assert max(x1) < self.voc_size
            assert 0 <= y <= self.target_size

            data_x1[a] = x1
            data_y[a] = y
            len_x1[a] = len(s1)
        return data_x1, data_y, len_x1


    def get_test_batch_step(self, start, size):
        assert start+size <= len(self.data_idx_list_test)

        data_x1 = np.zeros((size, self.seq_max), dtype=np.int)
        data_y = np.zeros(size, dtype=np.int)
        len_x1 = np.zeros(size, dtype=np.int)

        for a in range(size):
            s1 = self.data_idx_list_test[start+a][0]
            label = self.data_idx_list_test[start+a][1]

            x1 = s1 + [self.dic["<PAD>"]] * (self.seq_max - len(s1))
            y = label

            assert len(x1) == self.seq_max
            assert max(x1) < self.voc_size
            assert 0 <= y <= self.target_size

            data_x1[a] = x1
            data_y[a] = y
            len_x1[a] = len(s1)
        return data_x1, data_y, len_x1
        
        