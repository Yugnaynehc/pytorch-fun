import glob
import random
import unicodedata
import string
import torch
from torch.autograd import Variable


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if c in all_letters
        and unicodedata.category(c) != 'Mn')


def read_lines(filepath):
    names = open(filepath).read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def char_to_tensor(c):
    t = torch.zeros(1, n_letters)
    t[0][all_letters.index(c)] = 1
    return t


def name_to_tensor(name):
    l = len(name)
    t = torch.zeros(l, 1, n_letters)
    for i in range(l):
        t[i][0][all_letters.index(name[i])] = 1
    return t


def training_pair():
    language_idx = random.randint(0, n_language-1)
    language = language_list[language_idx]
    language_tensor = torch.LongTensor([language_idx])
    name_idx = random.randint(0, len(name_dict[language])-1)
    name = name_dict[language][name_idx]
    name_tensor = name_to_tensor(name)
    # print('Language: %s \t name:%s' % (language, name))
    return (Variable(language_tensor), Variable(name_tensor),
            language, name)


def language_from_output(output):
    v, i = output.topk(1)
    return language_list[i.data[0][0]]


all_letters = string.ascii_letters + ' .,:;'
name_dict = {}
language_list = []
for filepath in glob.glob('../data/names/*.txt'):
    language = filepath.split('/')[-1][:-4]
    language_list.append(language)
    name_dict[language] = read_lines(filepath)

n_language = len(language_list)
n_letters = len(all_letters)

if __name__ == '__main__':
    for _ in range(1):
        training_pair()
