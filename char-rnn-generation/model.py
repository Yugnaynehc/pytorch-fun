import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, n_lang, n_inp, n_hid, n_out):
        super(Model, self).__init__()
        self.n_hid = n_hid
        self.i2o = nn.Linear(n_lang+n_inp+n_hid, n_out)
        self.i2h = nn.Linear(n_lang+n_inp+n_hid, n_hid)
        self.o2o = nn.Linear(n_out+n_hid, n_out)

    def forward(self, lang, input, hidden):
        in_comb = torch.cat((lang, input, hidden), 1)
        output = self.i2o(in_comb)
        hidden = self.i2h(in_comb)
        out_comb = torch.cat((output, hidden), 1)
        output = self.o2o(out_comb)
        output = F.dropout(output, p=0.1)
        output = F.log_softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.n_hid))
