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
from args import use_cuda, vgg_checkpoint


class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg16()
        self.vgg.load_state_dict(torch.load(vgg_checkpoint))

        # 用一种笨方法把VGG的最后一个fc层（及其之前的ReLU层）剔除掉
        self.vgg.classifier = nn.Sequential(
            *(self.vgg.classifier[i] for i in range(5)))

    def forward(self, images):
        return self.vgg(images)


class DecoderRNN(nn.Module):

    def __init__(self, frame_size, img_embed_size, hidden1_size,
                 word_embed_size, hidden2_size, num_frames, num_words, vocab):
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

        self.img_embed_size = img_embed_size
        self.hidden1_size = hidden1_size
        self.word_embed_size = word_embed_size
        self.hidden2_size = hidden2_size
        self.num_frames = num_frames
        self.num_words = num_words
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # linear1用来把视觉特征嵌入到低维空间
        self.linear1 = nn.Linear(frame_size, img_embed_size)
        # embed用来把文本特征嵌入到低维空间
        self.embed = nn.Embedding(self.vocab_size, word_embed_size)
        # lstm1_cell用来处理视觉特征
        self.lstm1_cell = nn.LSTMCell(img_embed_size, hidden1_size)
        # lstm2_cell用来处理视觉和文本的融合特征
        self.lstm2_cell = nn.LSTMCell(word_embed_size + hidden1_size, hidden2_size)
        # linear1用来把lstm的最终输出映射回文本空间
        self.linear2 = nn.Linear(hidden2_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax()

        self._init_weights()

    def _init_weights(self):
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.zero_()
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear2.bias.data.zero_()

    def _init_hidden(self, batch_size):
        hidden1 = (Variable(torch.zeros(batch_size, self.hidden1_size)),
                   Variable(torch.zeros(batch_size, self.hidden1_size)))
        hidden2 = (Variable(torch.zeros(batch_size, self.hidden2_size)),
                   Variable(torch.zeros(batch_size, self.hidden2_size)))
        if use_cuda:
            hidden1 = tuple(v.cuda() for v in hidden1)
            hidden2 = tuple(v.cuda() for v in hidden2)
        return hidden1, hidden2

    def forward(self, video_feats, captions):
        '''
        传入视频帧特征和caption，返回生成的caption
        不用teacher forcing模式（LSTM的输入来自caption的ground-truth）来训练
        而是用上一步的生成结果作为下一步的输入
        '''
        batch_size = len(captions)

        v = video_feats.view(-1, self.visual_size)
        v = self.linear1(v)
        v = v.view(batch_size, self.num_frames, self.img_embed_size)

        # 初始化encoding阶段的LSTM隐层
        hidden1, hidden2 = self._init_hidden(batch_size)

        # 为了在python2中使用惰性的range，需要安装future包
        # sudo pip2 install future
        for i in range(self.num_frames):
            hidden1 = self.lstm1_cell(v[:, i, :], hidden1)
            word_pad = Variable(torch.zeros(batch_size, self.word_embed_size),
                                requires_grad=False)
            if use_cuda:
                word_pad = word_pad.cuda()
            cat = torch.cat((word_pad, hidden1[0]), 1)
            hidden2 = self.lstm2_cell(cat, hidden2)
        video_pad = torch.zeros((batch_size, self.num_words, self.img_embed_size))
        video_pad = Variable(video_pad, requires_grad=False)
        if use_cuda:
            video_pad = video_pad.cuda()

        # 开始准备输出啦！
        outputs = []
        # 先送一个<start>标记
        word_id = self.vocab('<start>')
        word = Variable(torch.zeros(batch_size, 1).long().fill_(word_id))
        if use_cuda:
            word = word.cuda()
        word = self.embed(word)
        word = word.squeeze(1)
        for i in range(self.num_words):
            if captions[:, i].data.sum() == 0:
                # <pad>的id是0，如果所有的word id都是0，
                # 意味着所有的句子都结束了，没有必要再算了
                break
            hidden1 = self.lstm1_cell(video_pad[:, i, :], hidden1)
            cat = torch.cat((word, hidden1[0]), 1)
            hidden2 = self.lstm2_cell(cat, hidden2)
            word_logits = self.log_softmax(self.linear2(hidden2[0]))
            word_id = word_logits.max(1)[1]
            word = self.embed(word_id).squeeze(1)
            outputs.append(word_logits)
        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1).contiguous()
        return outputs

    def sample(video_feats):
        # 因为训练的时候没有采用teacher forcing
        # 所以forward和sample是类似的
        pass
