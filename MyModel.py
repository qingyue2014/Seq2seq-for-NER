# coding=utf-8
# @author: cer
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys
import os
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

class Model:
    def __init__(self, embedding_size, hidden_size, vocab_size, slot2index, epoch_num, batch_size,isAttention,isCRF):

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot2index=slot2index
        self.slot_size = len(slot2index)
        self.epoch_num = epoch_num
        self.isAttention=isAttention
        self.isCRF=isCRF
        # 每句输入的实际长度，除了padding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')

    def build(self, isembedding, word_embeddings=None,is_inference=False):

        a = tf.constant(1)

        self.input_steps = tf.add(tf.reduce_max(self.encoder_inputs_actual_length, name='max_target_len'),a)

        self.encoder_inputs = tf.placeholder(tf.int32, [None,self.batch_size],
                                             name='encoder_inputs')
        self.decoder_targets = tf.placeholder(tf.int32, [self.batch_size, None],
                                              name='decoder_targets')

        ending = tf.strided_slice(self.decoder_targets, [0, 1], [self.batch_size, self.input_steps], [1, 1])

        self.decoder_outputs = tf.concat([ending,tf.fill([self.batch_size, 1], self.slot2index['<EOS>'])], 1)

        self.global_step=tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

        if isembedding==False:
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],
                                                        -0.1, 0.1), dtype=tf.float32, name="random_embedding")
        else:
            self.embeddings = tf.Variable(word_embeddings, dtype=tf.float32, trainable=False,name="pre_embedding")

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        print("encoder_inputs_embedded:", self.encoder_inputs_embedded)
        self.decoder_embedding = tf.Variable(tf.random_uniform([self.slot_size, self.embedding_size],
                                                          -0.1, 0.1), trainable=True, dtype=tf.float32,
                                        name="decoder_random_embedding")

        decoder_embeddeds = tf.nn.embedding_lookup(self.decoder_embedding, self.decoder_targets)

        #用于存储decoder解码概率
        #self.logits = tf.Variable(tf.random_uniform([self.input_steps, self.batch_size, self.slot_size]), dtype=tf.float32, name='logits')
        #print("self.logits:",self.logits)
        print("decoder_embeddeds:",decoder_embeddeds)
        # Encoder

        # 使用单个LSTM cell
        encoder_f_cell_0 = LSTMCell(self.hidden_size)
        encoder_b_cell_0 = LSTMCell(self.hidden_size)
        encoder_f_cell = DropoutWrapper(encoder_f_cell_0,output_keep_prob=0.5)
        encoder_b_cell = DropoutWrapper(encoder_b_cell_0,output_keep_prob=0.5)
        # encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embedded, perm=[1, 0, 2])
        # 下面四个变量的尺寸：T*B*D，T*B*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)

        # Decoder
        decoder_lengths = self.encoder_inputs_actual_length
        self.slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        self.slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")
        # <sos> 在词表中索引为0 0时刻decoder的输入
        sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='SOS') * 2
        sos_step_embedded = tf.nn.embedding_lookup(self.embeddings, sos_time_slice)
        # pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        # pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
        pad_step_embedded = tf.zeros([self.batch_size, self.hidden_size*2+self.embedding_size],
                                     dtype=tf.float32)
        encoder_add_embedded=tf.zeros([self.batch_size,self.hidden_size*2],dtype=tf.float32)

        def initial_fn():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):

            #self.update = tf.assign(self.logits[time], outputs)
            print("outputs:", outputs)
            #self.assign_op = tf.assign(self.logits[time], outputs, validate_shape=False)
            #print("self.logits:",self.logits)
            # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
            # print("slot output_logits: ", output_logits)
            # prediction_id = tf.argmax(output_logits, axis=1)
            # 选择logit最大的下标作为sample
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            print("prediction:",tf.shape(prediction_id))
            return prediction_id

        def next_inputs_fn1(time, outputs, state, sample_ids):
            # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
            print("sample_ids:",tf.shape(sample_ids))
            print("self.decoder_targets[time]:", decoder_embeddeds[time])
            pred_embedding = tf.nn.embedding_lookup(self.decoder_embedding, sample_ids)
            print("pre_embedding",pred_embedding)
            # 输入是h_i+o_{i-1}+c_i
            next_input=tf.concat((pred_embedding, encoder_outputs[time]), 1)
            '''next_input=tf.cond(time < self.input_steps-1,
                               lambda :tf.concat((pred_embedding, encoder_outputs[time - 1],encoder_outputs[time + 1]), 1),
                               lambda :tf.concat((pred_embedding, encoder_outputs[time - 1], encoder_add_embedded), 1))'''
            print("next_input",next_input)
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            next_state = state
            return elements_finished, next_inputs, next_state

        def next_inputs_fn2(time, outputs, state, sample_ids):
            # 训练时使用正确的标签作为每一步的输入
            print("sample_ids:",tf.shape(sample_ids))
            print("self.decoder_targets[time]:", decoder_embeddeds[time])
            #pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
            pred_embedding = tf.transpose(decoder_embeddeds, [1, 0, 2])[time]
            print("pre_embedding",pred_embedding)
            # 输入是h_i+o_{i-1}+c_i
            next_input=tf.concat((pred_embedding, encoder_outputs[time]), 1)
            '''next_input=tf.cond(time < self.input_steps-1,
                               lambda :tf.concat((pred_embedding, encoder_outputs[time - 1],encoder_outputs[time + 1]), 1),
                               lambda :tf.concat((pred_embedding, encoder_outputs[time - 1], encoder_add_embedded), 1))'''
            print("next_input",next_input)
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            next_state = state
            return elements_finished, next_inputs, next_state


        my_helper1 = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn1)

        my_helper2 = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn2)



        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    fc_layer = Dense(self.slot_size)
                    cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
                    if self.isAttention:
                        memory = tf.transpose(encoder_outputs, [1, 0, 2])
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                            num_units=self.hidden_size, memory=memory,
                            memory_sequence_length=self.encoder_inputs_actual_length)
                        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                            cell, attention_mechanism, attention_layer_size=self.hidden_size)
                        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                            attn_cell, self.slot_size, reuse=reuse
                        )
                        decoder = tf.contrib.seq2seq.BasicDecoder(
                            cell=out_cell, helper=helper,
                            initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size), output_layer=fc_layer)
                    else:
                        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper,
                            initial_state=self.encoder_final_state, output_layer=fc_layer)

                    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder, output_time_major=True,
                        impute_finished=True, maximum_iterations=self.input_steps)
                    return final_outputs
        #预测时使用
        if is_inference:
            outputs = decode(my_helper1, 'decode')
        #训练时使用
        else:
            outputs = decode(my_helper2, 'decode')

        print("outputs.rnn_output: ", outputs.rnn_output)
        print("outputs.sample_id: ", outputs.sample_id)
        # weights = tf.to_float(tf.not_equal(outputs[:, :-1], 0))
        # [batch_size,input_steps,slot_size]
        if self.isCRF:
            self.logits = tf.transpose(outputs.rnn_output, [1, 0, 2])
            print(self.logits)
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                    tag_indices=self.decoder_outputs,
                                                                    sequence_lengths=decoder_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            self.decoder_prediction = outputs.sample_id
            decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(outputs.rnn_output))
            self.decoder_targets_time_majored = tf.transpose(self.decoder_outputs, [1, 0])
            self.decoder_targets_true_length = self.decoder_targets_time_majored[:decoder_max_steps]
            # 定义mask，使padding不计入loss计算
            self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
            # 定义slot标注的损失
            loss_slot = tf.contrib.seq2seq.sequence_loss(
                outputs.rnn_output, self.decoder_targets_true_length, weights=self.mask)
            self.loss = loss_slot

        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars),global_step=self.global_step)

    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        #print(np.transpose(unziped[0], [1, 0]))
        #print(np.shape(unziped[0]), np.shape(unziped[1]),np.shape(unziped[2]))
        if mode == 'train':
            if self.isCRF:
                output_feeds = [self.train_op, self.loss, self.slot_W, self.transition_params]
            else:
                output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                            self.mask, self.slot_W]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                             self.encoder_inputs_actual_length: unziped[1],
                             self.decoder_targets: unziped[2]}
        if mode in ['test']:
            if self.isCRF:
                output_feeds = [self.logits,self.transition_params]
            else:
                output_feeds = self.decoder_prediction
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                             self.encoder_inputs_actual_length: unziped[1]}
        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results
