import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ShowTellModel(nn.Module):

    def __init__(self, opt):
        super(ShowTellModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.embedding_size = opt.embedding_size
        self.rnn_type = opt.rnn_type.upper()
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.max_seq_len = opt.max_seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0  # Schedule sampling probability
        self.img_embed = nn.Linear(self.fc_feat_size, self.embedding_size)
        self.rnn = getattr(nn, self.rnn_type)(self.embedding_size,
                                              self.hidden_size, self.num_layers,
                                              bias=False, dropout=self.drop_prob_lm)
        self.word_embed = nn.Embedding(self.vocab_size + 1, self.embedding_size)
        self.logit = nn.Linear(self.hidden_size, self.vocab_size + 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.img_embed.weight.data.uniform_(-initrange, initrange)
        self.img_embed.bias.data.zero_()

        self.word_embed.weight.data.uniform_(-initrange, initrange)

        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.zero_()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_())

    def forward(self, fc_feats, att_feats, seq):
        bsz = fc_feats.size(0)
        hidden = self.init_hidden(bsz)
        outputs = []

        # seq的第一维的大小batch size, 第二维的大小是caption的最大长度
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                it = seq[:, i - 1].clone()

                if i >= 2 and seq[:, i - 1].data.sum() == 0:
                    break
                xt = self.word_embed(it)

            output, hidden = self.rnn(xt.unsqueeze(0), hidden)
            output = F.log_softmax(self.logit(output.squeeze(0)))
            outputs.append(output)
        return torch.cat([o.unsqueeze(1) for o in outputs[1:]], 1).contiguous()
