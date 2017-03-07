import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, n_inp, n_hid, n_repeats=1):
        super(Encoder, self).__init__()

        self.n_repeats = n_repeats

        self.embedding = nn.Embedding(n_inp, n_hid)
        self.gru = nn.GRU(n_hid, n_hid)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)  # 一次只放一个单词进来
        output = embedded
        for _ in range(self.n_repeats):
            output, hidden = self.gru(output, hidden)
        return output, hidden


class Decoder(nn.Module):

    def __init__(self, n_hid, n_out, n_repeats=1):
        # input和output的维数相同
        super(Decoder, self).__init__()

        self.n_repeats = n_repeats

        self.embedding = nn.Embedding(n_out, n_hid)
        self.gru = nn.GRU(n_hid, n_hid)
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        x = F.relu(embedded)
        for _ in range(self.n_repeats):
            x, hidden = self.gru(x, hidden)
        output = self.linear(x[0])  # seq_len 是1
        output = F.log_softmax(output)
        return output, hidden


class AttentionDecoder(nn.Module):

    def __init__(self, n_hid, n_out, max_len, dropout_p=0.1, n_repeats=1):
        super(AttentionDecoder, self).__init__()

        self.n_repeats = n_repeats
        self.dropout_p = dropout_p
        self.max_len = max_len

        self.embedding = nn.Embedding(n_out, n_hid)
        self.attn = nn.Linear(n_hid*2, max_len)
        self.attn_combine = nn.Linear(n_hid*2, n_hid)
        self.gru = nn.GRU(n_hid, n_hid)
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, x, hidden, encoder_outputs):
        embedded = F.dropout(self.embedding(x).view(1, -1), self.dropout_p)
        attn = self.attn(torch.cat((hidden[0], embedded), 1))
        attn_weights = F.softmax(attn)
        attn_applied = torch.mm(attn_weights, encoder_outputs)
        attn_combined = self.attn_combine(torch.cat((embedded, attn_applied), 1))

        output = attn_combined.unsqueeze(0)
        for _ in range(self.n_repeats):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights
