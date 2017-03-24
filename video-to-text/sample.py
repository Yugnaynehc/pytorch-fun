# coding: utf-8

import os
import cv2
import pickle
import torch
from torch.autograd import Variable
from vocab import Vocabulary
from model import EncoderCNN
from args import video_root, vocab_pkl_path
from args import frame_sample_rate, num_frames, frame_size
from args import decoder_pth_path
from utils import preprocess_frame, decode_tokens
import numpy as np
import random


with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)

# 载入预训练模型
encoder = EncoderCNN()
decoder = torch.load(decoder_pth_path)
encoder.eval()
encoder.cuda()
decoder.eval()
decoder.cuda()

# 载入视频
videos = os.listdir(video_root)
video = random.choice(videos)
video_path = os.path.join(video_root, video)
print(video_path)
try:
    cap = cv2.VideoCapture(video_path)
except:
    print('Can not open %s.' % video_path)
    pass

frame_count = 0
frame_list = []

count = 1
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    count += 1
    if count % frame_sample_rate == 0:
        frame_list.append(frame)
        frame_count += 1
        count = 1

frame_list = np.array(frame_list)
if frame_count > num_frames:
    # frame_indices = np.linspace(0, frame_count,
    #                             num=num_frames, endpoint=False).astype(int)
    # 直接截断
    frame_list = frame_list[:num_frames]
    frame_count = num_frames

# 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
cropped_frame_list = np.array([preprocess_frame(x)
                               for x in frame_list]).transpose((0, 3, 1, 2))
cropped_frame_list = Variable(torch.from_numpy(cropped_frame_list),
                              volatile=True).cuda()

# 视频特征的shape是num_frames x 4096
# 如果帧的数量小于num_frames，则剩余的部分用0补足
feats = np.zeros((num_frames, frame_size), dtype='float32')
feats[:frame_count, :] = encoder(cropped_frame_list).data.cpu().numpy()
videos = Variable(torch.from_numpy(feats)).cuda().unsqueeze(0)

# 对视频内容进行解码得到自然语言描述
tokens = decoder.sample(videos).data[0].squeeze()
print(decode_tokens(tokens, vocab))

# 播放视频

cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    cv2.imshow('video', frame)
    if cv2.waitKey(20) == 27:
        break
