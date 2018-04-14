from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import collections
import re
import collections
import pickle
import random

file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_idx_list = "./data/data_idx_list.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

max = 0
map = np.zeros((300), np.int32)
data_list0 = []
data_list1 = []
word_list = []

with open("./data/rt-polarity.neg", encoding="latin-1") as f:
    lines = f.readlines()

    for line in lines:
        line = clean_str(line)

        word_list.append(line)

        words = line.split()
        data_list0.append([words, 0])

        map[len(words)] += 1
        if max < len(words):
            max = len(words)

with open("./data/rt-polarity.pos", encoding="latin-1") as f:
    lines = f.readlines()

    for line in lines:
        line = clean_str(line)

        word_list.append(line)

        words = line.split()
        data_list1.append([words, 1])

        map[len(words)] += 1
        if max < len(words):
            max = len(words)

#for zzz in range(map.shape[0]):
#    if map[zzz] > 0:
#        print("[%03d] count : %d" % (zzz, map[zzz]))

print("data_list_neg size : %d" % (len(data_list0)))
print("data_list_neg example")
for a in range(5):
    print(data_list0[a])
print()

print("data_list_pos size : %d" % (len(data_list1)))
print("data_list_pos example")
for a in range(5):
    print(data_list1[a])
print()

total_words = " ".join(word_list).split()
count = collections.Counter(total_words).most_common()

symbols = ["<PAD>", "<UNK>"]
rdic = symbols + [i[0] for i in count if i[1] > 0]     # word list order by count desc
dic = {w: i for i, w in enumerate(rdic)}                      # dic {word:count} order by count desc
voc_size = len(dic)
print("voc_size size = %d" % voc_size)
print("data_list example")
print(rdic[:20])
SIZE_VOC = len(dic)

max_len = max
print("sentence max len = %d" % max_len)
print()

data_idx_list0 = []
for words, label in data_list0:
    words_idx = []

    for word in words:
        idx = -1
        if word in dic:
            idx = dic[word]
        else:
            idx = dic["<UNK>"]
        assert 0 <= idx < SIZE_VOC
        words_idx.append(idx)

    data_idx_list0.append([words_idx, label])

data_idx_list1 = []
for words, label in data_list1:
    words_idx = []

    for word in words:
        idx = -1
        if word in dic:
            idx = dic[word]
        else:
            idx = dic["<UNK>"]
        assert 0 <= idx < SIZE_VOC
        words_idx.append(idx)

    data_idx_list1.append([words_idx, label])

random.shuffle(data_idx_list0)
random.shuffle(data_idx_list1)

SIZE_TEST_DATA = int(len(data_idx_list0) * 0.1)
data_idx_list_test = data_idx_list0[:SIZE_TEST_DATA] + data_idx_list1[:SIZE_TEST_DATA]
data_idx_list = data_idx_list0[SIZE_TEST_DATA:] + data_idx_list1[SIZE_TEST_DATA:]
print("dataset for train = %d" % len(data_idx_list))
print("dataset for test = %d" % len(data_idx_list_test))
print()

# save dictionary
with open(file_data_idx_list, 'wb') as handle:
    pickle.dump(data_idx_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_data_idx_list_test, 'wb') as handle:
    pickle.dump(data_idx_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_dic, 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_rdic, 'wb') as handle:
    pickle.dump(rdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_max_len, 'wb') as handle:
    pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("dictionary files saved..")
print()