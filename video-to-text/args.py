# coding: utf-8

'''
这里存放一些参数
'''
import os

# 训练相关的超参数
num_epochs = 1
batch_size = 32
learning_rate = 1e-2
use_cuda = True

# 模型相关的超参数
img_embed_size = 500
word_embed_size = 500
hidden1_size = 512  # 第一个LSTM层的隐层单元数目
hidden2_size = 512  # 第二个KSTM层的隐层单元数目

# frame_shape = (512, 7, 7)  # 视屏帧特征的形状
frame_size = 4096  # 视频特征的维度
num_frames = 60  # 图像序列的规定长度
num_words = 30  # 文本序列的规定长度


# 数据相关的参数
video_root = './raw/TrainValVideo/'
anno_json_path = './raw/train_val_videodatainfo.json'

feat_save_path = './feats'
vocab_pkl_path = os.path.join(feat_save_path, 'vocab.pkl')
caption_pkl_path = os.path.join(feat_save_path, 'captions.pkl')

video_h5_path = os.path.join(feat_save_path, 'videos.h5')
video_h5_dataset = 'feats'


# checkpoint相关的超参数
vgg_checkpoint = './models/vgg16-00b39a1b.pth'
