import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, n_inp, n_hid, n_out):
        super(Model, self).__init__()
        self.c2o = nn.Linear(n_inp+n_hid, n_out)
        self.c2h = nn.Linear(n_inp+n_hid, n_hid)
        self.n_hid = n_hid

    def forward(self, input, hidden):
        comb = torch.cat((input, hidden), 1)
        output = self.c2o(comb)
        hidden = self.c2h(comb)
        output = F.log_softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.n_hid))
