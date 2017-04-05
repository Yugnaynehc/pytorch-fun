# coding: utf-8

from builtins import range
import os
import pickle
from utils import decode_tokens
from vocab import Vocabulary
from data import get_loader
from model import DecoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from args import vocab_pkl_path, caption_train_pkl_path, video_h5_path
from args import num_epochs, batch_size, learning_rate
from args import img_embed_size, word_embed_size
from args import hidden1_size, hidden2_size
from args import frame_size, num_frames, num_words
from args import use_cuda, use_checkpoint
from args import decoder_pth_path, optimizer_pth_path
from args import log_environment
from tensorboard_logger import configure, log_value

configure(log_environment, flush_secs=10)


# 加载词典
with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 构建模型
decoder = DecoderRNN(frame_size, img_embed_size, hidden1_size, word_embed_size,
                     hidden2_size, num_frames, num_words, vocab)

if os.path.exists(decoder_pth_path) and use_checkpoint:
    decoder.load_state_dict(torch.load(decoder_pth_path))
if use_cuda:
    decoder.cuda()

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
if os.path.exists(optimizer_pth_path) and use_checkpoint:
    optimizer.load_state_dict(torch.load(optimizer_pth_path))

# 打印训练环境的参数设置情况
print('Learning rate: %.4f' % learning_rate)
print('Batch size: %d' % batch_size)

# 初始化数据加载器
train_loader = get_loader(caption_train_pkl_path, video_h5_path, batch_size)
total_step = len(train_loader)

# 开始训练模型
loss_count = 0
for epoch in range(num_epochs):
    for i, (videos, captions, lengths, video_ids) in enumerate(train_loader):
        # 构造mini batch的Variable
        videos = Variable(videos)
        targets = Variable(captions)
        if use_cuda:
            videos = videos.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = decoder(videos, targets)
        # 因为在一个epoch快要结束的时候，有可能采不到一个刚好的batch
        # 所以要重新计算一下batch size
        bsz = len(captions)
        # 把output压缩（剔除pad的部分）之后拉直
        outputs = torch.cat([outputs[j][:lengths[j]] for j in range(bsz)], 0)
        outputs = outputs.view(-1, vocab_size)
        # 把target压缩（剔除pad的部分）之后拉直
        targets = torch.cat([targets[j][:lengths[j]] for j in range(bsz)], 0)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        log_value('loss', loss.data[0], epoch * total_step + i)
        loss_count += loss.data[0]
        loss.backward()
        optimizer.step()

        if i % 9 == 0 and i > 0:
            loss_count /= 10
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                  (epoch, num_epochs, i, total_step, loss_count,
                   np.exp(loss_count)))
            loss_count = 0
            tokens = decoder.sample(videos).data[0].squeeze()
            we = decode_tokens(tokens, vocab)
            gt = decode_tokens(captions[0].squeeze(), vocab)
            print('[vid:%d]' % video_ids[0])
            print('WE: %s\nGT: %s' % (we, gt))
    torch.save(decoder.state_dict(), decoder_pth_path)
    torch.save(optimizer.state_dict(), optimizer_pth_path)

torch.save(decoder.state_dict(), decoder_pth_path)
torch.save(optimizer.state_dict(), optimizer_pth_path)
