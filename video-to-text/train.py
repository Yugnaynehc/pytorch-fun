# coding: utf-8

from builtins import range
import pickle
from utils import decode_tokens
from vocab import Vocabulary
from data import get_loader
from model import DecoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from args import vocab_pkl_path, caption_pkl_path, video_h5_path
from args import num_epochs, batch_size, learning_rate
from args import img_embed_size, word_embed_size
from args import hidden1_size, hidden2_size
from args import frame_size, num_frames, num_words
from args import use_cuda


# 加载词典
with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 初始化数据加载器
train_loader = get_loader(caption_pkl_path, video_h5_path, batch_size)
total_step = len(train_loader)

# 构建模型
decoder = DecoderRNN(frame_size, img_embed_size, hidden1_size, word_embed_size,
                     hidden2_size, num_frames, num_words, vocab)
if use_cuda:
    decoder.cuda()

# 初始化损失函数和优化器
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (videos, captions, lengths) in enumerate(train_loader):
        # 构造mini batch的Variable
        videos = Variable(videos)
        targets = Variable(captions)
        if use_cuda:
            videos = videos.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = decoder(videos, targets)
        # 把output拉直
        outputs = torch.cat([outputs[j][:lengths[j]] for j in range(batch_size)], 0)
        outputs = outputs.view(-1, vocab_size)
        # 把target拉直
        targets = torch.cat([targets[j][:lengths[j]] for j in range(batch_size)], 0)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                  (epoch, num_epochs, i, total_step, loss.data[0],
                   np.exp(loss.data[0])))
        if i % 200 == 0 and i > 0:
            torch.save(decoder, 'decoder.pth')

torch.save(decoder, 'decoder.pth')
