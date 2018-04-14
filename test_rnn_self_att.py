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

from batcher import data_batcher
from model_rnn_self_att import rnn_self_att_model

file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_idx_list = "./data/data_idx_list.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"
file_word2vec = "./data/word2vec.bin"
dir_summary = "./model/summary/"

np.random.seed(0)

print("-"*70)
print("SELF ATTENTIVE MODEL TESTER..")
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

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(now)
print("Test start!!")
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
                                 lr= 0.000125,
                                 dev= "/gpu:1", 
                                 sess= sess,
                                 makedir= False)
    model.load_model(tf.train.latest_checkpoint("./model/rnn_self_att/2017-12-20 11:19/checkpoints/"))
    
    BATCHS_TEST = 10
    data_x1, data_y, len_x1 = batcher.get_test_batch_step(500, BATCHS_TEST)

    results = model.batch_test(BATCHS_TEST, data_x1, data_y, len_x1, False)
    batch_pred = results[0]
    batch_loss = results[1]
    batch_acc = results[2]
    batch_att = results[3]
    batch_att_tot = results[4]
    g_step = results[5]
    batch_lr = results[6]
    
    """"""
    # visualize..
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import rcParams, rc
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    path = '/Library/Fonts/AppleGothic.ttf'
    prop =  fm.FontProperties(fname=path)
    path = fm.findfont(prop, directory=path)
    print(prop.get_name())
    rc('font', family=prop.get_name())
    rc('text', usetex='false')
    rcParams['font.family'] = prop.get_name()
    rcParams.update({'font.size': 14})

    print()
    
    for a in range(BATCHS_TEST):
        att_tot = batch_att_tot[a].T
        att = np.reshape(batch_att[a], [1, -1])
        
        sentence = [rdic[w] for w in data_x1[a] if w != 0]
        print(sentence)
        print("target:%d, Predict:%d" % (data_y[a], batch_pred[a]))
        
        print("attention_total_rows")
        plt.clf()
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        im = ax.imshow(att_tot[:, :len(sentence)], cmap="YlOrBr")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        
        y_label = []
        for b in range(att_tot.shape[0]):
            y_label.append("row %d" % b)
        ax.set_xticks(range(len(sentence)))
        ax.set_xticklabels(sentence, fontsize=14, rotation=90, fontproperties=prop)
        ax.set_yticks(range(att_tot.shape[0]))
        ax.set_yticklabels(y_label, fontsize=14, rotation=0, fontproperties=prop)

        ax.grid()
        #plt.show()
        plt.savefig(os.path.join("./img", "att_tot_%d.png"%a), bbox_inches="tight")
        
        print("attention_final")
        plt.clf()
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        im = ax.imshow(att[:, :len(sentence)], cmap="YlOrBr")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        
        ax.set_xticks(range(len(sentence)))
        ax.set_xticklabels(sentence, fontsize=14, rotation=90, fontproperties=prop)
        ax.set_yticks(range(att.shape[0]))
        ax.set_yticklabels(["prob "], fontsize=14, rotation=0, fontproperties=prop)

        ax.grid()
        plt.savefig(os.path.join("./img", "att_final_%d.png"%a), bbox_inches="tight")
        
        print("~" * 70)

print()
print("Test finished!!")
print()