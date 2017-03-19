# coding: utf-8

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        '''
        加载预训练好的resent101模型，把最后一个fc层替换成embed层并训练之
        '''
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet101(pretrained=True)

        # 降低内存使用
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.init_weights()

    def init_weights(self):
        self.resnet.fc.weight.data.uniform_(-0.1, 0.1)
        self.resnet.fc.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        bsz = len(captions)
        features = torch.stack([features]*self.num_layers, 0)
        state = (features,
                 Variable(features.data.new(self.num_layers, bsz,
                                            self.hidden_size).zero_()))
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        outputs, _ = self.lstm(packed, state)
        outputs = self.linear(outputs[0])
        return outputs

    def sample(self, feature):
        feature = torch.stack([feature]*self.num_layers, 0)
        state = (feature,
                 Variable(feature.data.new(self.num_layers, 1,
                                           self.hidden_size).zero_()))
        sampled_ids = []
        input = Variable(torch.LongTensor([self.vocab.word2idx['<start>']]))
        if feature.is_cuda:
            input = input.cuda()
        input = input.unsqueeze(0)
        input = self.embed(input)
        for i in range(20):
            output, state = self.lstm(input, state)
            output = self.linear(output.view(-1, self.hidden_size))
            predicted = output.max(1)[1]
            predicted_id = predicted.data[0][0]
            sampled_ids.append(predicted_id)
            if predicted_id == self.vocab.word2idx['<end>']:
                break
            input = self.embed(predicted)
        return sampled_ids
