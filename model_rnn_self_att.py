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


class rnn_self_att_model(object):
    def __init__(self, voc_size, target_size, input_len_max, lr, dev, sess, makedir=True):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Create rnn_self_att_model class...")
        print()

        self.voc_size = voc_size
        self.target_size = target_size
        self.input_len_max = input_len_max
        self.lr = lr
        self.sess = sess
        self.dev = dev
        self.makedir = makedir

        self._build_graph()
        self.sess.run(tf.global_variables_initializer())


    def _build_graph(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Build Graph...")
        print()

        self.xavier_init = tf.contrib.layers.xavier_initializer()

        self.embed_dim = 100
        self.state_dim = 100
        self.bi_state_dim = self.state_dim * 2
        self.attend_dim = 250
        self.feat_dim = self.bi_state_dim
        self.fc_dim = 150

        print("embed_dim : %d" % self.embed_dim)
        print("state_dim : %d" % self.state_dim)
        print("bi_state_dim : %d" % self.bi_state_dim)
        print("attend_dim : %d" % self.attend_dim)
        print("feat_dim : %d" % self.feat_dim)
        print("fc_dim : %d" % self.fc_dim)
        print()

        with tf.device(self.dev):
            with tf.variable_scope("input_placeholders"):
                self.enc_input = tf.placeholder(tf.int32, shape=[None, None], name="enc_input")
                self.enc_seq_len = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len")
                self.targets = tf.placeholder(tf.int32, shape=[None, ], name="targets")
                self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            with tf.variable_scope("words_embedding"):
                self.embeddings = tf.get_variable("embeddings", [self.voc_size, self.embed_dim], initializer=self.xavier_init)
                self.embed_in = tf.nn.embedding_lookup(self.embeddings, self.enc_input, name="embed_in")

                self.pad_mask = tf.sequence_mask(self.enc_seq_len, self.input_len_max, dtype=tf.float32, name="pad_mask1")

            with tf.variable_scope("rnn_encoder_layer"):
                self.output_enc, self.state_enc = bi_rnn(GRUCell(self.state_dim), GRUCell(self.state_dim),
                                                         inputs=self.embed_in, sequence_length=self.enc_seq_len, dtype=tf.float32)

                self.state_enc = tf.concat([self.state_enc[0], self.state_enc[1]], axis=1, name="state_enc1")
                assert self.state_enc.get_shape()[1] == self.bi_state_dim

                self.output_enc = tf.concat(self.output_enc, axis=2)  # [batch, max_eng, state*2]
                self.output_enc = tf.nn.dropout(self.output_enc, keep_prob=self.keep_prob, name="output_enc1")
                print("output_enc.get_shape() : %s" % (self.output_enc.get_shape()))
                assert self.output_enc.get_shape()[2] == self.bi_state_dim

            with tf.variable_scope("attention_layer"):
                self.rows = 30
                self.W_s1 = tf.get_variable("W_s1", [1, 1, self.feat_dim, self.attend_dim], initializer=self.xavier_init)
                self.bias_s1 = tf.get_variable("bias_s1", [self.attend_dim])
                self.W_s2 = tf.get_variable("W_s2", [self.attend_dim, self.rows], initializer=self.xavier_init)

                self.identity = tf.reshape(tf.tile(tf.diag(tf.ones(self.rows)), [self.batch_size, 1]),
                                           [self.batch_size, self.rows, self.rows], name="identity")

                self.output_enc_ex = tf.reshape(self.output_enc, [-1, self.input_len_max, 1, self.feat_dim])
                self.context_att = tf.nn.conv2d(self.output_enc_ex, self.W_s1, strides=[1,1,1,1], padding="SAME")

                self.context_att = tf.tanh(tf.nn.bias_add(self.context_att, self.bias_s1), name="context_att")
                print("context_att.get_shape() : %s" % (self.context_att.get_shape()))

                # attention
                self.attention_tot = tf.matmul(tf.reshape(self.context_att, [-1, self.attend_dim]), self.W_s2)
                self.attention_tot = tf.reshape(self.attention_tot, [-1, self.input_len_max, self.rows])
                self.attention_tot = tf.nn.softmax(self.attention_tot, dim=1) * tf.reshape(self.pad_mask, [-1, self.input_len_max, 1])
                self.attention_tot = tf.nn.softmax(self.attention_tot, dim=1)
                print("attention_tot.get_shape() : %s" % (self.attention_tot.get_shape()))

                self.attention = tf.reduce_sum(self.attention_tot, axis=2)
                self.attention = tf.reshape(self.attention, [self.batch_size, self.input_len_max]) * self.pad_mask
                self.attention = tf.nn.softmax(self.attention)
                print("attention.get_shape() : %s" % (self.attention.get_shape()))

                self.attention_tot_T = tf.transpose(self.attention_tot, [0, 2, 1], name="attention_tot_T")
                self.AA_t = tf.matmul(self.attention_tot_T, self.attention_tot) - self.identity
                print("AA_t.get_shape() : %s" % (self.AA_t.get_shape()))

                # penalty
                self.P = tf.square(tf.norm(self.AA_t, axis=[-2, -1], ord="fro"))
                self.P = tf.reduce_mean(self.P, name="P")

                # context..
                self.context = tf.reduce_sum(self.output_enc * tf.reshape(self.attention, [-1, self.input_len_max, 1]),
                                             axis=1, name="context")
                print("context.get_shape() : %s" % (self.context.get_shape()))
                assert self.context.get_shape()[1] == self.feat_dim

            with tf.variable_scope("dense_layer"):
                self.W_out1 = tf.get_variable("W_out1", [self.feat_dim, self.fc_dim], initializer=self.xavier_init)
                self.bias_out1 = tf.get_variable("bias_out1", [self.fc_dim])
                self.W_out2 = tf.get_variable("W_out2", [self.fc_dim, self.target_size], initializer=self.xavier_init)
                self.bias_out2 = tf.get_variable("bias_out2", [self.target_size])

                self.fc = tf.nn.xw_plus_b(self.context, self.W_out1, self.bias_out1)
                self.fc = tf.tanh(self.fc)
                print("fc.get_shape() : %s" % (self.fc.get_shape()))

                self.y_hat = tf.nn.xw_plus_b(self.fc, self.W_out2, self.bias_out2, name="y_hat")
                print("y_hat.get_shape() : %s" % (self.y_hat.get_shape()))

            with tf.variable_scope("train_optimization"):
                self.train_vars = tf.trainable_variables()

                print()
                print("trainable_variables")
                for varvar in self.train_vars:
                    print(varvar)
                print()

                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.targets)
                self.loss = tf.reduce_mean(self.loss, name="loss")
                self.loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.train_vars if "bias" not in v.name]) * 0.0001
                self.loss = self.loss + self.loss_l2 + self.P

                self.predict = tf.argmax(tf.nn.softmax(self.y_hat), 1)
                self.predict = tf.cast(tf.reshape(self.predict, [self.batch_size, 1]), tf.int32, name="predict")

                self.target_label = tf.cast(tf.reshape(self.targets, [self.batch_size, 1]), tf.int32)
                self.correct = tf.equal(self.predict, self.target_label)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name="accuracy")

                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.decay_rate = tf.maximum(0.00007,
                                             tf.train.exponential_decay(self.lr, self.global_step,
                                                                        1000, 0.9, staircase=True),
                                             name="decay_rate")
                self.opt = tf.train.AdamOptimizer(learning_rate=self.decay_rate)
                self.grads_and_vars = self.opt.compute_gradients(self.loss, self.train_vars)
                self.grads_and_vars = [(tf.clip_by_norm(g, 0.5), v) for g, v in self.grads_and_vars]
                self.grads_and_vars = [(tf.add(g, tf.random_normal(tf.shape(g), stddev=0.001)), v) for g, v in self.grads_and_vars]

                self.train_op = self.opt.apply_gradients(self.grads_and_vars, global_step=self.global_step, name="train_op")


            # Summaries for loss and lr
            self.loss_summary = tf.summary.scalar("loss", self.loss)
            self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
            self.lr_summary = tf.summary.scalar("lr", self.decay_rate)

            # Output directory for models and summaries
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.out_dir = os.path.abspath(os.path.join("./model/rnn_self_att", timestamp))
            print("LOGDIR = %s" % self.out_dir)
            print()

            # Train Summaries
            self.train_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.lr_summary])
            self.train_summary_dir = os.path.join(self.out_dir, "summary", "train")
            self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)

            # Test summaries
            self.test_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.lr_summary])
            self.test_summary_dir = os.path.join(self.out_dir, "summary", "test")
            self.test_summary_writer = tf.summary.FileWriter(self.test_summary_dir, self.sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model-step")
            if self.makedir:
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    def batch_train(self, batchs, data_x, data_y, len_x, writer=False):
        feed_dict = {self.enc_input: data_x,
                     self.enc_seq_len: len_x,
                     self.targets: data_y,
                     self.batch_size: batchs,
                     self.keep_prob: 0.6}

        results = \
            self.sess.run([self.train_op, self.predict, self.loss, self.accuracy, self.attention, self.attention_tot,
                           self.global_step, self.decay_rate, self.train_summary_op],
                          feed_dict)

        if writer:
            self.train_summary_writer.add_summary(results[8], results[6])

        ret = [results[1], results[2], results[3], results[4], results[5], results[6], results[7]]
        return ret

    def batch_test(self, batchs, data_x, data_y, len_x, writer=False):
        feed_dict = {self.enc_input: data_x,
                     self.enc_seq_len: len_x,
                     self.targets: data_y,
                     self.batch_size: batchs,
                     self.keep_prob: 1.0}

        results = \
            self.sess.run([self.predict, self.loss, self.accuracy, self.attention, self.attention_tot,
                           self.global_step, self.decay_rate, self.test_summary_op],
                          feed_dict)

        if writer:
            self.test_summary_writer.add_summary(results[7], results[5])

        ret = [results[0], results[1], results[2], results[3], results[4], results[5], results[6]]
        return ret

    def save_model(self):
        current_step = tf.train.global_step(self.sess, self.global_step)
        self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)

    def load_model(self, file_model):
        print("Load model (%s)..." % file_model)
        #file_model = "./model/2017-12-20 11:19/checkpoints/"
        #self.saver.restore(self.sess, tf.train.latest_checkpoint(file_model))
        self.saver.restore(self.sess, file_model)

