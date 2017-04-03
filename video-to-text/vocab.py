# coding: utf-8
'''
对MSR-VTT数据集的标注文本进行处理。根据描述句子的集合提取出一个字典。
'''

from __future__ import print_function

import os
import pickle
import json
import nltk
from collections import Counter
from args import anno_json_path, feat_dir, vocab_pkl_path


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, w):
        '''
        将新单词加入词汇表中
        '''
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1

    def __call__(self, w):
        '''
        返回单词对应的id
        '''
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        '''
        得到词汇表中词汇的数量
        '''
        return self.nwords


def build_vocab(rawdata, threshold):
    '''
    根据标注的文本得到词汇表。频数低于threshold的单词将会被略去
    '''
    counter = Counter()
    with open(rawdata, 'r') as f:
        msr = json.load(f)
    sentences = msr['sentences']
    ncaptions = len(sentences)
    for i, row in enumerate(sentences):
        caption = row['caption']
        # 直接按照空格进行单词的切分
        # tokens = caption.lower().split(' ')
        # 使用nltk来进行单词切分
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i % 1000 == 0:
            print('[{}/{}] tokenized the captions.'.format(i, ncaptions))

    # 略去一些低频词
    words = [w for w, c in counter.items() if c >= threshold]
    # 剔除掉 '.' 这个符号，用<end>代替
    words.remove('.')
    # 开始构建词典！
    vocab = Vocabulary()
    for w in words:
        vocab.add_word(w)
    return vocab


def main():
    vocab = build_vocab(rawdata=anno_json_path, threshold=3)
    print('Vocabulary has %d words.' % len(vocab))
    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)
    with open(vocab_pkl_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('Save vocabulary to %s' % vocab_pkl_path)


if __name__ == '__main__':
    main()
