# coding: utf-8

from __future__ import print_function

import torchvision.transforms as T
import pickle
from data import get_loader
from vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"


def decode_tokens(tokens, vocab):
    words = []
    cnt = 0
    for token in tokens:
        word = vocab.idx2word[token]
        words.append(word)
        cnt += len(word)
        if cnt > 47:
            words.append('\n')
            cnt = 0
    caption = ' '.join(words)
    return caption


# 超参数设置
num_epochs = 5
batch_size = 200
embed_size = 356
hidden_size = 768
crop_size = 224
num_layers = 1
learning_rate = 1e-3
train_image_path = './data/train2014resized/'
train_json_path = './data/annotations/captions_train2014.json'

# 图像预处理
transform = T.Compose([
    T.RandomCrop(crop_size),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


# 加载词典
with open('./vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 初始化数据加载器
train_loader = get_loader(train_image_path, train_json_path, vocab, transform,
                          batch_size=batch_size, shuffle=True, num_workers=2)
total_step = len(train_loader)

# 构建模型
encoder = EncoderCNN(hidden_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab, num_layers)
encoder.cuda()
decoder.cuda()

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)


plt.ion()
# 训练模型
for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):
        # 构造 mini batch
        images = Variable(images).cuda()
        inputs = Variable(captions[:, :-1]).cuda()
        targets = Variable(captions[:, 1:]).cuda()
        # 减1是因为在算loss的时候不考虑开头的<start>
        lengths = [l - 1 for l in lengths]
        targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]

        optimizer.zero_grad()
        features = encoder(images)
        outputs = decoder(features, inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            tokens = decoder.sample(features[0].unsqueeze(0))
            we = decode_tokens(tokens, vocab)
            gt = decode_tokens(captions[0][1:], vocab)
            plt.imshow(images[0].cpu().data.numpy().transpose((1, 2, 0)))
            plt.suptitle('WE: %s\nGT: %s' % (we, gt), fontsize=10,
                         x=0.1, horizontalalignment='left')
            plt.axis('off')
            plt.pause(0.1)
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                  (epoch, num_epochs, i, total_step, loss.data[0],
                   np.exp(loss.data[0])))
        if i % 250 == 0 and i > 0:
            torch.save(decoder, 'decoder.pth')
            torch.save(encoder, 'encoder.pth')


torch.save(decoder, 'decoder.pth')
torch.save(encoder, 'encoder.pth')
