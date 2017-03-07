import torch
from torch.autograd import Variable
from data import random_choice, generate_language_tensor, all_letters, n_letters
import sys


def generate_char_var(c):
    t = torch.zeros(1, n_letters)
    t[0][all_letters.index(c)] = 1
    return Variable(t)


def sample(output):
    v, i = output.topk(1)
    index = i.data[0][0]
    return all_letters[index] if index != n_letters-1 else None


def generate_name(lang, start_letters=None):
    net = torch.load('generator.pt')
    max_step = 20
    lang_var = Variable(generate_language_tensor(lang))
    if start_letters is None:
        start_letters = [random_choice(all_letters[-33:-7])]
    for letter in start_letters:
        name = letter
        hidden = net.init_hidden()
        for _ in range(max_step):
            input_var = generate_char_var(name[-1])
            output, hidden = net(lang_var, input_var, hidden)
            char = sample(output)
            if char:
                name += char
            else:
                print(name)
                break


if __name__ == '__main__':
    if len(sys.argv) == 3:
        generate_name(sys.argv[1], sys.argv[2])
    else:
        generate_name(sys.argv[1])
