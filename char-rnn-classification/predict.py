import sys
import torch
from torch.autograd import Variable
from data import name_to_tensor, language_list

model = torch.load('char-rnn-classification.pt')


def inference(name):
    model.init_hidden()
    name_tensor = name_to_tensor(name)
    name_var = Variable(name_tensor)

    for t in name_var:
        output = model(t)
    return output


def predict(name, n_pred=3):
    output = inference(name)
    v, i = output.topk(n_pred)
    for x in range(n_pred):
        language = language_list[i.data[0][x]]
        print('(%.2f) %s' % (v.data[0][x], language))


if __name__ == '__main__':
    predict(sys.argv[1])
