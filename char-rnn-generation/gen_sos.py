import torch
from torch.autograd import Variable
from data import random_choice, generate_language_tensor, all_letters, n_letters_sos
import sys


def generate_char_var(c):
    t = torch.zeros(1, n_letters_sos)
    t[0][all_letters.index(c)] = 1
    return Variable(t)


def sample(output):
    v, i = output.topk(1)
    index = i.data[0][0]
    return all_letters[index] if index < n_letters_sos-2 else None


def generate_name_sos(lang):
    net = torch.load('generator_sos.pt')
    max_step = 20
    lang_var = Variable(generate_language_tensor(lang))
    name = ''
    input_tensor = torch.zeros(1, n_letters_sos)
    input_tensor[0][n_letters_sos-2] = 1
    input_var = Variable(input_tensor)
    hidden = net.init_hidden()
    for _ in range(max_step):
        output, hidden = net(lang_var, input_var, hidden)
        char = sample(output)
        if char:
            name += char
        else:
            print(name)
            break
        input_var = generate_char_var(name[-1])


if __name__ == '__main__':
    generate_name_sos(sys.argv[1])
