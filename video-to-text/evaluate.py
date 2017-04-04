# coding: utf-8
'''
在测试集上生成描述结果
'''

import os
import pickle
from utils import decode_tokens
import torch
from torch.autograd import Variable
from vocab import Vocabulary
from data import get_loader
from model import DecoderRNN
from args import vocab_pkl_path, video_h5_path
from args import decoder_pth_path
from args import frame_size, num_frames, num_words
from args import img_embed_size, word_embed_size
from args import hidden1_size, hidden2_size
from args import caption_val_pkl_path, val_range
from args import use_cuda
from args import reference_txt_path, predict_txt_path


with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)

# 载入预训练模型
decoder = DecoderRNN(frame_size, img_embed_size, hidden1_size, word_embed_size,
                     hidden2_size, num_frames, num_words, vocab)
decoder.load_state_dict(torch.load(decoder_pth_path))
decoder.cuda()
decoder.eval()

# 载入测试数据集
# 只能开一个进程! 不然会出现结果不稳定的灵异现象,估计是因为多进程没有同步导致数据读乱了
test_loader = get_loader(caption_val_pkl_path, video_h5_path, 100, num_workers=1)
total_step = len(test_loader)

# reference_txt = codecs.open(reference_txt_path, 'w')

result = {}
processed_count = 0
total_count = val_range[1] - val_range[0]

for i, (videos, captions, lengths, video_ids) in enumerate(test_loader):
    # 过滤一下已经计算过的视频
    selected_idx = []
    selected_video_ids = []
    for j, vid in enumerate(video_ids):
        if vid not in result:
            selected_idx.append(j)
            selected_video_ids.append(vid)
            result[vid] = None  # 占位,防止在一个batch里面一个视频被重复计算
    count = len(selected_idx)
    if count == 0:
        # 如果整个batch的视频都被计算过,就跳过这个batch
        continue
    else:
        processed_count += count
    selected_idx = torch.LongTensor(selected_idx).contiguous()

    # 构造mini batch的Variable
    videos = videos[selected_idx].contiguous()
    videos = Variable(videos)

    if use_cuda:
        videos = videos.cuda()

    outputs = decoder.sample(videos).data.squeeze(2)
    for (tokens, vid) in zip(outputs, selected_video_ids):
        s = decode_tokens(tokens, vocab)
        result[vid] = s

    # captions = captions.squeeze()
    # for tokens, vid in zip(captions, video_ids):
    #     try:
    #         s = decode_tokens(tokens, vocab)
    #         reference_txt.write('%d\t%s\n' % (vid + 1, s))  # MSVD数据集的index从1开始
    #     except Exception as e:
    #         print(s)
    #         # reference_txt.write('%d\t%s\n' % (vid, s[:-1]))
    print('Processed %d/%d' % (processed_count, total_count))

# reference_txt.close()

predict_txt = open(predict_txt_path, 'w')
for vid, s in result.items():
    predict_txt.write('%d\t%s\n' % (vid + 1, s))  # MSVD数据集的index从1开始
predict_txt.close()
