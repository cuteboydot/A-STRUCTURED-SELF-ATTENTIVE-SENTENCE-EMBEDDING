from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import GRUCell
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
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from batcher import data_batcher
from model_rnn_self_att import rnn_self_att_model

file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_idx_list = "./data/data_idx_list.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"
dir_summary = "./model/summary/"

np.random.seed(0)

pre_trained = 0

print("-"*70)
print("SELF ATTENTIVE MODEL TRAINER..")
print("-"*70)
print()

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(now)
print("Load vocabulary from model file...")

with open(file_data_idx_list, 'rb') as handle:
    data_idx_list = pickle.load(handle)
with open(file_data_idx_list_test, 'rb') as handle:
    data_idx_list_test = pickle.load(handle)
with open(file_rdic, 'rb') as handle:
    rdic = pickle.load(handle)
with open(file_dic, 'rb') as handle:
    dic = pickle.load(handle)
with open(file_max_len, 'rb') as handle:
    max_len = pickle.load(handle)

SIZE_VOC = len(dic)
print("voc_size = %d" % SIZE_VOC)

SIZE_SENTENCE_MAX = max_len
print("max_sentence_len = %d" % SIZE_SENTENCE_MAX)
print()

print("dataset for train = %d" % len(data_idx_list))
print("dataset for test = %d" % len(data_idx_list_test))
SIZE_TRAIN_DATA = len(data_idx_list)
SIZE_TEST_DATA = len(data_idx_list_test)
print()

BATCHS = 200
BATCHS_TEST = SIZE_TEST_DATA
EPOCHS = 100
STEPS = int(len(data_idx_list) / BATCHS)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(now)
print("Train start!!")
print()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:    
    batcher = data_batcher(data_idx_list, data_idx_list_test, dic, SIZE_SENTENCE_MAX, 2)
    model = rnn_self_att_model(voc_size= SIZE_VOC, 
                                 target_size= 2,
                                 input_len_max= SIZE_SENTENCE_MAX, 
                                 lr= 0.001,
                                 dev= "/gpu:1", 
                                 sess= sess,
                                 makedir= True)
    
    loop_step = 0
    for epoch in range(EPOCHS):
        for step in range(STEPS):
            data_x1, data_y, len_x1 = batcher.get_train_batch_rand(BATCHS)

            writer = False
            if loop_step % 50 == 0:
                writer = True

            results = model.batch_train(BATCHS, data_x1, data_y, len_x1, writer)
            batch_pred = results[0]
            batch_loss = results[1]
            batch_acc = results[2]
            batch_att = results[3]
            batch_att_tot = results[4]
            g_step = results[5]
            batch_lr = results[6]

            if loop_step % 100 == 0:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("epoch[%03d] glob_step[%06d] - batch_loss:%.5f, batch_acc:%.4f, lr=%.7f  (%s)" %
                      (epoch, g_step, batch_loss, batch_acc, batch_lr, now))

            if loop_step % 100 == 0:
                data_x1, data_y, len_x1 = batcher.get_test_batch_rand(BATCHS_TEST)

                results = model.batch_test(BATCHS_TEST, data_x1, data_y, len_x1, writer)
                batch_pred = results[0]
                batch_loss = results[1]
                batch_acc = results[2]
                batch_att = results[3]
                batch_att_tot = results[4]
                g_step = results[5]
                batch_lr = results[6]

                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("epoch[%03d] glob_step[%06d] - test_loss: %.5f, test_acc: %.4f, lr=%.7f  (%s)" %
                      (epoch, g_step, batch_loss, batch_acc, batch_lr, now))
                
            loop_step += 1

        model.save_model()
        
print()
print("Train finished!!")
print()