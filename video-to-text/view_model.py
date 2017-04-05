# coding: utf-8


from utils import make_dot
import pickle
from model import DecoderRNN
from vocab import Vocabulary
from args import vocab_pkl_path
from args import img_embed_size, word_embed_size
from args import hidden1_size, hidden2_size
from args import frame_size, num_frames, num_words
import torch
from torch.autograd import Variable

# 加载词典
with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)


# 构建模型
decoder = DecoderRNN(frame_size, img_embed_size, hidden1_size, word_embed_size,
                     hidden2_size, num_frames, num_words, vocab)
videos = Variable(torch.FloatTensor(10, num_frames, frame_size))
outputs = decoder(videos, None)
graph = make_dot(outputs)
graph.render('model.gv', view=False, cleanup=True)
