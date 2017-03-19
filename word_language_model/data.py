import os
import re
import torch


class Dictionary(object):

    def __init__(self):
        self.word2idx = {'<EOS>': 0}
        self.idx2word = ['<EOS']
        self.n_words = 1

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = self.n_words
            self.idx2word.append(w)
            self.n_words += 1
        return self.word2idx[w]

    def __len__(self):
        return self.n_words


class Corpus(object):

    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)

        n_tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<EOS>']
                n_tokens += len(words)
                for w in words:
                    self.dictionary.add_word(w)

        ids = torch.LongTensor(n_tokens)
        with open(path, 'r') as f:
            token = 0
            for line in f:
                words = line.split() + ['<EOS>']
                for w in words:
                    ids[token] = self.dictionary.word2idx[w]
                    token += 1
        return ids
