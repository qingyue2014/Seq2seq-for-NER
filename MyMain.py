# coding=utf-8
# @author: cer
import tensorflow as tf
from Mydata import *
from MyModel import Model
import numpy as np
import os, argparse
import pickle
import time
from tensorflow.contrib.crf import viterbi_decode

parser = argparse.ArgumentParser(description='Seq2seq-Attention-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default="TEXT.txt", help='#train_data_name')
parser.add_argument('--test_data', type=str, default="TEXT.txt", help='#test_data_name')
parser.add_argument('--variables_path', type=str, default="variables.bin", help='keep_variables_name')
parser.add_argument('--CRF', type=str2bool, default=False, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--Attention', type=str2bool, default=False, help='use attention at encoder. if False, not attention')
parser.add_argument('--embed_size', type=int, default=64, help='input_embed_size')
parser.add_argument('--hidden_size', type=int, default=100, help='#dim of hidden state')
parser.add_argument('--batch_size', type=int, default=16, help='#batch number')
parser.add_argument('--epoch_num', type=int, default=20, help='epoch times')
parser.add_argument('--number_cuda', type=str, default="0", help='#number of cuda_visible_devices')
parser.add_argument('--action', type=str, default="train", help='#train/predict')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.number_cuda
embedding_size = args.embed_size #pre_embedding=false时有效
hidden_size = args.hidden_size
batch_size = args.batch_size
epoch_num =args.epoch_num
train_data_path=os.path.join('.', 'data_path',args.train_data)
test_data_path=os.path.join('.', 'data_path',args.test_data)
variables_path=os.path.join('.', 'variables_path',args.variables_path)
isCRF=args.CRF
isAttention=args.Attention
action=args.action
model_path='checkpoints/'
pre_embedding=False  #是否使用预训练的embedding
pre_embedding_path='WordData/Daily.seg.300d.vector' #pre_embedding_path==true时有效
pre_embedding_dim=300
output_path='output.txt' #输出测试结果路径 格式：word true_tag pred_tag
eval_result_file='result.txt' # 评估结果路径
curtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def train(sess,model,index_train,is_debug=False):
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    fw=open('log.txt','a')
    fw.write('running time:'+curtime_str+'\n')
    fw.write("parameter configure:\tisCRF:{}\tisAttention:{}\tembedding_size:{}\thidden_size:{}\tbatch_size:{}\tepoch_num:{}\ntrain_loss:\n"
	                             .format(isCRF,isAttention,embedding_size, hidden_size,batch_size,epoch_num))
    print("---------------training beginning--------------")
    for epoch in range(epoch_num):
        train_loss = 0.0
        time=0
        for i, batch in enumerate(getBatch(batch_size, index_train,"train")):
            # 执行一个batch的训练
            if isCRF:
                _, loss, slot_W, transition_params = model.step(sess, "train", batch)
            else:
                _, loss, decoder_prediction, mask, slot_W = model.step(sess, "train", batch)
            train_loss += loss
            time=i
        train_loss /= (time + 1)
        fw.write(str(train_loss)+"\n")
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))
        if (epoch+1) % 2 == 0 :
            saver.save(sess,model_path,global_step=model.global_step)
            predict(sess, model, index_test)
            print("----saving ----")
        # 每训一个epoch，测试一次
    print("---------------training finished--------------")
    fw.close()

def predict(sess,model,index_test):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)

    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)
    else:
        print('Model not found, please train your model first')
        return
    fr = open(test_data_path,'r',encoding='utf-8').readlines()
    fw = open(output_path, 'w', encoding='utf-8')
    for j, batch in enumerate(getBatch(batch_size, index_test,"test")):
        if isCRF:
            logits,transition_params = model.step(sess, "test", batch)
            seq_len_list=list(zip(*batch))[1]
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            for index in range(len(label_list)):
                linenum=index+j*batch_size
                input_sen=fr[linenum].replace('\n','').split('\t')[0].strip().split(' ')
                slot_truth=fr[linenum].replace('\n','').split('\t')[1].strip().split(' ')
                tag_list=[]
                for tag_index in label_list[index]:
                    tag=index2slot[tag_index]
                    tag_list.append(tag)
                s=writeResult(input_sen,slot_truth,tag_list)
                fw.write(s)
                fw.write('\n')
        else:
            decoder_prediction = model.step(sess, "test", batch)
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            for index in range(decoder_prediction.shape[0]):
                sen_len=batch[index][1]
                linenum=index+j*batch_size
                input_sen=fr[linenum].replace('\n','').split('\t')[0].strip().split(' ')
                slot_truth=fr[linenum].replace('\n','').split('\t')[1].strip().split(' ')
                slot_pred=index_seq2slot(decoder_prediction[index], index2slot)[:sen_len]
                s=writeResult(input_sen,slot_truth,slot_pred)
                fw.write(s)
                fw.write('\n')
    fw.close()
    printEvalResult(output_path)
    print("---------------prediction finished--------------")

def writeResult(input_sen,truth_tag, pred_tag):
    s=''
    for result in zip(input_sen,truth_tag,pred_tag) :
        s+=result[0]+' '+result[1]+' '+result[2]+'\n'
    return s

def printEvalResult(output_path):
    eval_perl='conlleval.pl'
    cmd='perl {} <{}> {}'.format(eval_perl, output_path, eval_result_file)
    os.system(cmd)
    with open(eval_result_file,'r',encoding='utf-8') as fr:
        for line in fr:
            line=line.replace('\n','')
            print(line)


if __name__ == '__main__':
    if not os.path.exists(variables_path):
        save_file=open(variables_path,"wb")
        train_data=open(train_data_path,'r',encoding='utf-8').readlines()
        test_data = open(test_data_path, 'r', encoding='utf-8').readlines()
        train_data_ed = data_pipeline(train_data)
        test_data_ed = data_pipeline(test_data)
        word2index, index2word, slot2index, index2slot = \
            vocab_build(train_data_ed)
        index_train = to_index(train_data_ed, word2index, slot2index)
        index_test = to_index(test_data_ed, word2index, slot2index)

        pickle.dump(word2index,save_file)
        pickle.dump(slot2index,save_file)
        pickle.dump(index2slot,save_file)
        pickle.dump(index_train,save_file)
        pickle.dump(index_test,save_file)
    else:
        print("loading from variables.bin......",variables_path)
        load_file=open(variables_path,"rb")
        word2index=pickle.load(load_file)
        slot2index=pickle.load(load_file)
        index2slot=pickle.load(load_file)
        index_train=pickle.load(load_file)
        index_test=pickle.load(load_file)
    new_embedding_size=embedding_size
    word_embeddings=None
    if pre_embedding == True:
        new_embedding_size = pre_embedding_dim
        word_embeddings = load_embed_txt(pre_embedding_path, word2index, pre_embedding_dim)
    model = Model(new_embedding_size, hidden_size, len(word2index), slot2index, epoch_num, batch_size,isAttention,isCRF)
    model.build(isembedding=pre_embedding, word_embeddings=word_embeddings, is_inference=True)
    with tf.Session() as sess:
        if action=='train':
            train(sess,model,index_train)
        else:
            predict(sess,model,index_test)


