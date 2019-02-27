# coding=utf-8
# @author: qingyue

import numpy as np
import codecs
import os
import random
import logging, sys, argparse

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def data_pipeline(data, length=50):
    print("---------------processing corpus--------------")
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # 分割成这样[原始句子的词，标注的序列，intent]
    data = [[t.split("\t")[0].strip().split(" "), t.split("\t")[1].strip().split(" ")] for t in
            data]
    data = [[t[0], t[1]] for t in data]  # 将BOS和EOS去掉，并去掉对应标注序列中相应的标注
    seq_in, seq_out = list(zip(*data))
    sin = []
    sout = []
    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        if (i + 1) % 10000 == 0:
            print('reading corpus processing line number :{}'.format(i + 1))
        temp = seq_in[i]
        temp.append('<EOS>')
        sin.append(temp)

        temp = seq_out[i]
        temp.insert(0, '<SOS>')
        sout.append(temp)
        data = list(zip(sin, sout))
    print("---------------processing corpus finished--------------")
    return data

def load_embed_txt(embed_file, vocab, dim):
    '''
    加载已有的词向量到embedding中
    '''
    print("---------------loading the pre_embeddings--------------")
    embeddings = np.random.seed(12345)
    embeddings = np.random.uniform(-1, 1, size=(len(vocab), dim))
    with codecs.open(embed_file, 'r', encoding='utf-8') as f:
        for (num, line) in enumerate(f):
            if num == 0:
                continue
            if (num + 1) % 1000 == 0:
                print('reading embeddings processing number :{}'.format(num + 1))
            tokens = line.strip().split(" ")
            word = tokens[0]
            vector = tokens[1]
            embedding = [float(x) for x in vector.split(' ')]
            if word in vocab:
                word_id = vocab[word]
                embeddings[word_id] = np.asarray(embedding)
    print("---------------loading the pre_embeddings finished--------------")
    return embeddings


def vocab_build(data, min_count=5):
    """
    :param min_count: 罕见字出现频率的最小阈值
    :return:
    """
    print("---------------building the vocab--------------")
    word_count = {}
    # print(data)
    seq_in, seq_out = list(zip(*data))
    '''
    <PAD>: 补齐序列
    <UNK>: 未出现词
    <O>:   其他
    '''
    tag2id = {'<PAD>': 0, '<UNK>': 1, "O": 2, "<EOS>": 3}
    for sent_ in seq_in:
        for word in sent_:
            if word not in word_count:
                word_count[word] = [len(word_count) + 1, 1]
            else:
                word_count[word][1] += 1
    for tag_ in seq_out:
        for tagert in tag_:
            if tagert not in tag2id:
                tag2id[tagert] = len(tag2id)
    low_freq_words = []
    for word, [word_id, word_freq] in word_count.items():
        if word_freq < min_count:
            low_freq_words.append(word)

    # 删去罕见字
    for word in low_freq_words:
        del word_count[word]
    '''
    <PAD>: 补齐序列
    <UNK>: 未出现词
    <SOS>: decoder时0时刻的输入
    <EOS>: 序列结束的标志词
    '''
    word2id = {'<PAD>': 0, '<UNK>': 1, "<SOS>": 2, "<EOS>": 3}
    for word in word_count.keys():
        if word not in word2id:
            word2id[word] = len(word2id)
    '''with open('data1', 'wb') as fw:
        pickle.dump(word2id, fw)'''

    # 生成id2word
    id2word = {v: k for k, v in word2id.items()}

    # 生成id2tag
    id2tag = {v: k for k, v in tag2id.items()}
    print('word2id', word2id)
    print('tag2id', tag2id)
    print("---------------building the vocab finished--------------")
    return word2id, id2word, tag2id, id2tag


def process(batch):
    unziped = list(zip(*batch))
    max_length = max(unziped[1]) + 1
    padbatch = []
    for i in zip(unziped[0], unziped[1], unziped[2]):
        data = []
        source = i[0][:]
        while len(source) < max_length:
            source.append(0)
        target = i[2][:]
        while len(target) < max_length:
            target.append(0)
        data.append(source)
        data.append(i[1])
        data.append(target)
        padbatch.append(data)
    return padbatch


def getBatch(batch_size, train_data,mode):
    '''
    返回一个batch的迭代器
    '''
    if mode=="train":
        random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        padbatch = process(batch)
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield padbatch


def test_data():
    train_data = open('TEST.txt', 'r', encoding='utf-8').readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(train_data)
    word2index, index2word, slot2index, index2slot = \
        vocab_build(train_data_ed)
    # print("slot2index: ", slot2index)
    # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index)
    index_test = to_index(test_data_ed, word2index, slot2index)
    batch = next(getBatch(16, index_test))
    # print(batch)
    unziped = list(zip(*batch))
    print("word num: ", len(word2index.keys()), "slot num: ", len(slot2index.keys()))
    print(unziped[0])
    print(np.shape(unziped[0]))


def to_index(train, word2index, slot2index):
    '''
    source_data和target_data映射为对应的序号序列
    '''
    new_train = []
    for sin, sout in train:
        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                           sout))
        new_train.append([sin_ix, true_length, sout_ix])
    return new_train


if __name__ == '__main__':
    test_data()