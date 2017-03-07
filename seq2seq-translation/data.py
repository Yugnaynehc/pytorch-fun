"""
We do thress things in the code:
1. read the source sentence from txt file
2. normalize these sentence and build dual language pairs
3. filter some undisered pairs
4. generate input and target sequence
"""

import os
import re
import torch
import pickle
import random
import unicodedata
from lang import Lang
from torch.autograd import Variable


max_len = 10
eng_prefix = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
data_root = '../data/'


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    # add space between some symbols
    s = re.sub(r'([.?!])', r' \1', s)
    # filter out some symbols
    s = re.sub(r'[^0-9a-zA-Z\t.?!]+', r' ', s)
    return s.split('\t')


def get_pair(filepath, reverse=False):
    raw = open(filepath).read().strip().split('\n')
    raw_pairs = [normalize_string(l) for l in raw]
    pairs = []
    for p in raw_pairs:
        # filter some sentences
        if not p[0].startswith(eng_prefix):
            continue
        if re.compile(r'.*\d').match(p[0]) is not None:
            continue
        s = p[0].split(' ')
        t = p[1].split(' ')
        if len(s) > max_len or len(t) > max_len:
            continue
        if reverse:
            pairs.append([t, s])
        else:
            pairs.append([s, t])
    return pairs


def prepare_data(source_lang, target_lang, reverse=False):
    filename = '%s-%s.txt' % (source_lang, target_lang)
    pairs = get_pair(os.path.join(data_root, filename), reverse)
    if reverse:
        source_lang_dict = Lang(target_lang)
        target_lang_dict = Lang(source_lang)
    else:
        source_lang_dict = Lang(source_lang)
        target_lang_dict = Lang(target_lang)
    for p in pairs:
        source_lang_dict.add_word_list(p[0])
        target_lang_dict.add_word_list(p[1])
    return pairs, source_lang_dict, target_lang_dict


def sentence_to_idx_var(s, lang_dict, cuda=False):
    l = []
    for w in s:
        l.append(lang_dict.word2idx[w])
    l.append(lang_dict.word2idx['EOS'])
    var = Variable(torch.LongTensor(l).view(-1, 1))
    return var if not cuda else var.cuda()


def get_training_var(cuda=False):
    data = random.choice(pairs)
    source_var = sentence_to_idx_var(data[0], source_lang_dict, cuda)
    target_var = sentence_to_idx_var(data[1], target_lang_dict, cuda)
    return source_var, target_var


saved_file = os.path.join(data_root, 'saved_language_pair.pkl')
if os.path.exists(saved_file):
    with open(saved_file, 'rb') as f:
        pairs, source_lang_dict, target_lang_dict = pickle.load(f)
else:
    pairs, source_lang_dict, target_lang_dict = prepare_data('eng', 'fra', reverse=True)
    with open(saved_file, 'wb') as f:
        pickle.dump([pairs, source_lang_dict, target_lang_dict], f)


if __name__ == '__main__':
    print(source_lang_dict.idx2word)
    print(pairs[:10])
