import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, n_inp, n_hid, n_out):
        super(Model, self).__init__()
        self.rnn = nn.RNN(n_inp, n_hid)
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, input):
        output, hidden = self.rnn(input, None)
        output = self.linear(output[-1])
        return F.log_softmax(output)
