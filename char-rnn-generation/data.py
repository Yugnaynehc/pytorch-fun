import random
import glob
import unicodedata
import string
import torch
from torch.autograd import Variable


def unicode_to_ascii(s):
    '''
    Convert unicode characters to ascii symbols
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if
        c in all_letters and
        unicodedata.category(c) != 'Mn')


def read_names(filepath):
    '''
    Read names for file
    '''
    names = open(filepath).read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def random_choice(d):
    '''
    Choose one example from the given array data
    '''
    return d[random.randint(0, len(d)-1)]


def generate_language_tensor(language):
    '''
    Generate the one-hot language tensor
    '''
    t = torch.zeros(1, n_language)
    t[0][language_list.index(language)] = 1
    return t


def generate_input_sequence_tensor(name):
    '''
    Generate input sequence tensor of certain name.
    For example, given a name 'Chen', we will generate a sequence
    (C, h, e, n) and convert them to one-hot vector and then merget them
    to a tensor.
    '''
    l = len(name)
    t = torch.zeros(l, 1, n_letters)
    for i in range(l):
        t[i][0][all_letters.index(name[i])] = 1
    return t


def generate_target_sequence_tensor(name):
    '''
    Generate target sequence tensor of certain name.
    For example, given a name 'Chen', we will generate a sequence
    (7, 4, 13, n_letters-1) and convert them to a tensor.
    '''
    target = [all_letters.index(c) for c in name[1:]]
    target.append(n_letters-1)  # EOS
    return torch.LongTensor(target)


def generate_training_data():
    '''
    Generate training data, which is a tuple of Variables,
    including language, input sequence, and output sequence
    '''
    lang = random_choice(language_list)
    name = random_choice(name_dict[lang])
    lang_tensor = generate_language_tensor(lang)
    input_tensor = generate_input_sequence_tensor(name)
    target_tensor = generate_target_sequence_tensor(name)
    return (Variable(lang_tensor), Variable(input_tensor),
            Variable(target_tensor))


def generate_input_sequence_tensor_sos(name):
    '''
    Generate input sequence tensor of certain name with <SOS>.
    (<SOS> is the symbol of 'starting of sentence')
    For example, given a name 'Chen', we will generate a sequence
    (<SOS>, C, h, e, n) and convert them to one-hot vector and then merget them
    to a tensor.
    '''
    l = len(name)
    t = torch.zeros(l+1, 1, n_letters_sos)
    t[0][0][n_letters_sos-2] = 1  # SOS
    for i in range(l):
        t[i+1][0][all_letters.index(name[i])] = 1
    return t


def generate_target_sequence_tensor_sos(name):
    '''
    Generate input sequence tensor of certain name with <SOS>.
    (<SOS> is the symbol of 'starting of sentence')
    For example, given a name 'Chen', we will generate a sequence
    (7, 4, 13, n_letters-2) and convert them to a tensor.
    '''
    target = [all_letters.index(c) for c in name]
    target.append(n_letters_sos-1)  # EOS
    return torch.LongTensor(target)


def generate_training_data_sos():
    '''
    Generate training data with <SOS>. Return is a tuple of Variables,
    including language, input sequence, and output sequence
    '''
    lang = random_choice(language_list)
    name = random_choice(name_dict[lang])
    lang_tensor = generate_language_tensor(lang)
    input_tensor = generate_input_sequence_tensor_sos(name)
    target_tensor = generate_target_sequence_tensor_sos(name)
    return (Variable(lang_tensor), Variable(input_tensor),
            Variable(target_tensor))


all_letters = string.ascii_letters + " .,:;'-"
language_list = []
name_dict = {}

for filepath in glob.glob('../data/names/*.txt'):
    language = filepath.split('/')[-1][:-4]
    language_list.append(language)
    name_dict[language] = read_names(filepath)

n_letters = len(all_letters) + 1
n_letters_sos = n_letters + 1
n_language = len(language_list)
generate_training_data()
