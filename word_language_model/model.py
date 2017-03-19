import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntokens, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError('invalid rnn type')
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, bias=False)
        self.decoder = nn.Linear(nhid, ntokens)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, x, hidden):
        embedded = self.encoder(x)
        output, hidden = self.rnn(embedded, hidden)
        output = output.view(-1, self.nhid)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data  # 为了得到数据类型
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
