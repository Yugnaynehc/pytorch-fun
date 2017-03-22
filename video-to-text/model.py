# coding: utf-8

'''
用一个带有两层LSTM的模型来处理video caption任务，
视频和文本共享LSTM权重，虽然看起来不太合理，但是有效减少了模型的参数数量。
为了把视频内容送入到LSTM中，需要插入一个全连接层来做视觉特征的embedding（降维）。
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from builtins import range
from args import vgg_checkpoint
import random


class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg16()
        self.vgg.load_state_dict(torch.load(vgg_checkpoint))

        # 用一种笨方法把VGG的最后一个fc层（其之前的ReLU层要保留）剔除掉
        self.vgg.classifier = nn.Sequential(
            *(self.vgg.classifier[i] for i in range(6)))

    def forward(self, images):
        return self.vgg(images)


class DecoderRNN(nn.Module):

    def __init__(self, frame_size, img_embed_size, hidden1_size,
                 word_embed_size, hidden2_size, num_frames, num_words,
                 vocab, use_cuda=False):
        '''
        frame_size: 视频帧的特征的大小，一般是4096（VGG的倒数第二个fc层）
        img_embed_size: 视觉特征的嵌入维度
        hidden1_size: 第一层LSTM层（处理视觉特征）的隐层维度
        word_emdeb_size: 文本特征的嵌入维度
        hidden2_size: 第二层LSTM层（处理文本和视频的融合特征）的隐层维度
        num_frames: 视觉特征的序列长度，默认是60
        num_words: 文本特征的序列长度，默认是30
        第二个LSTM层的输入维度是word_embed_size + hidden1_size
        '''
        super(DecoderRNN, self).__init__()

        self.frame_size = frame_size
        self.img_embed_size = img_embed_size
        self.hidden1_size = hidden1_size
        self.word_embed_size = word_embed_size
        self.hidden2_size = hidden2_size
        self.num_frames = num_frames
        self.num_words = num_words
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.use_cuda = use_cuda

        # video_embed用来把视觉特征嵌入到低维空间
        self.video_embed = nn.Linear(frame_size, img_embed_size)
        self.video_drop = nn.Dropout(p=0.5)
        # word_embed用来把文本特征嵌入到低维空间
        self.word_embed = nn.Embedding(self.vocab_size, word_embed_size)
        self.word_drop = nn.Dropout(p=0.5)
        # lstm1_cell用来处理视觉特征
        self.lstm1_cell = nn.LSTMCell(img_embed_size, hidden1_size)
        self.lstm1_drop = nn.Dropout(p=0.5)
        # lstm2_cell用来处理视觉和文本的融合特征
        self.lstm2_cell = nn.LSTMCell(word_embed_size + hidden1_size, hidden2_size)
        self.lstm2_drop = nn.Dropout(p=0.5)
        # linear用来把lstm的最终输出映射回文本空间
        self.linear = nn.Linear(hidden2_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax()

        self._init_weights()

        if use_cuda:
            self.cuda()

    def _init_weights(self):
        self.video_embed.weight.data.uniform_(-0.1, 0.1)
        self.video_embed.bias.data.zero_()
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()

    def _init_lstm_state(self, batch_size, volatile=False):
        lstm1_hidden = Variable(torch.zeros(batch_size, self.hidden1_size),
                                volatile=volatile)
        lstm1_cell = Variable(torch.zeros(batch_size, self.hidden1_size),
                              volatile=volatile)
        lstm1_state = (lstm1_hidden, lstm1_cell)

        lstm2_hidden = Variable(torch.zeros(batch_size, self.hidden2_size),
                                volatile=volatile)
        lstm2_cell = Variable(torch.zeros(batch_size, self.hidden2_size),
                              volatile=volatile)
        lstm2_state = (lstm2_hidden, lstm2_cell)

        if self.use_cuda:
            lstm1_state = tuple(v.cuda() for v in lstm1_state)
            lstm2_state = tuple(v.cuda() for v in lstm2_state)
        return lstm1_state, lstm2_state

    def forward(self, video_feats, captions, teacher_forcing_ratio=0.5):
        '''
        传入视频帧特征和caption，返回生成的caption
        不用teacher forcing模式（LSTM的输入来自caption的ground-truth）来训练
        而是用上一步的生成结果作为下一步的输入
        UPDATED: 最后还是采用了teacher forcing，不然很难收敛
        '''
        batch_size = len(video_feats)

        v = video_feats.view(-1, self.frame_size)
        v = self.video_embed(v)
        v = self.video_drop(v)
        v = v.view(batch_size, self.num_frames, self.img_embed_size)

        # 初始化LSTM隐层
        lstm1_state, lstm2_state = self._init_lstm_state(batch_size)
        lstm1_hidden, lstm1_cell = lstm1_state
        lstm2_hidden, lstm2_cell = lstm2_state

        # 为了在python2中使用惰性的range，需要安装future包
        # sudo pip2 install future
        # Encoding 阶段！
        for i in range(self.num_frames):
            lstm1_hidden, lstm1_cell = self.lstm1_cell(v[:, i, :],
                                                       (lstm1_hidden, lstm1_cell))
            lstm1_hidden = self.lstm1_drop(lstm1_hidden)
            word_pad = Variable(torch.zeros(batch_size, self.word_embed_size),
                                requires_grad=False)
            if self.use_cuda:
                word_pad = word_pad.cuda()
            cat = torch.cat((word_pad, lstm1_hidden), 1)
            lstm2_hidden, lstm2_cell = self.lstm2_cell(cat,
                                                       (lstm2_hidden, lstm2_cell))
            lstm2_hidden = self.lstm2_drop(lstm2_hidden)
        video_pad = torch.zeros((batch_size, self.num_words, self.img_embed_size))
        video_pad = Variable(video_pad, requires_grad=False)
        if self.use_cuda:
            video_pad = video_pad.cuda()

        # Decoding 阶段！
        # 开始准备输出啦！
        outputs = []
        # 先送一个<start>标记
        word_id = self.vocab('<start>')
        word = Variable(torch.zeros(batch_size, 1).long().fill_(word_id))
        if self.use_cuda:
            word = word.cuda()
        word = self.word_embed(word)
        word = self.word_drop(word)
        word = word.squeeze(1)
        for i in range(self.num_words):
            if captions and captions[:, i].data.sum() == 0:
                # <pad>的id是0，如果所有的word id都是0，
                # 意味着所有的句子都结束了，没有必要再算了
                break
            lstm1_hidden, lstm1_cell = self.lstm1_cell(video_pad[:, i, :],
                                                       (lstm1_hidden, lstm1_cell))
            lstm1_hidden = self.lstm1_drop(lstm1_hidden)
            cat = torch.cat((word, lstm1_hidden), 1)
            lstm2_hidden, lstm2_cell = self.lstm2_cell(cat,
                                                       (lstm2_hidden, lstm2_cell))
            lstm2_hidden = self.lstm2_drop(lstm2_hidden)
            word_logits = self.log_softmax(self.linear(lstm2_hidden))
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                # teacher forcing模式
                word_id = captions[:, i]
            else:
                # 非 teacher forcing模式
                word_id = word_logits.max(1)[1]
            word = self.word_embed(word_id).squeeze(1)
            word = self.word_drop(word)
            if captions:
                # 给了caption说明是训练模式，要返回logits
                outputs.append(word_logits)
            else:
                # 否则是推断模式，直接返回单词id
                outputs.append(word_id)
        # unsqueeze(1)会把一个向量(n)拉成列向量(nx1)
        # outputs中的每一个向量都是整个batch在某个时间步的输出
        # 把它拉成列向量之后再横着拼起来，就能得到整个batch在所有时间步的输出
        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1).contiguous()
        return outputs

    def sample(self, video_feats):
        '''
        sample就是不给caption且不用teacher forcing的forward
        '''
        return self.forward(video_feats, None, teacher_forcing_ratio=0.0)
