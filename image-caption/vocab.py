# coding: utf-8

from __future__ import print_function

import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.nwords = 0

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word[self.nwords] = w
            self.nwords += 1

    def __call__(self, w):
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    ncaptions = len(ids)

    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print('[{}/{}] tokenized the captions.'.format(i, ncaptions))

    # 略去一些低频词
    words = [w for w, c in counter.items() if c >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for w in words:
        vocab.add_word(w)

    return vocab


def main():
    vocab = build_vocab(json='./data/annotations/captions_train2014.json',
                        threshold=4)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print('Saved vocabulary file to vocab.pkl')


if __name__ == '__main__':
    main()
