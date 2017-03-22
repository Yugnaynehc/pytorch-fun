# coding: utf-8

'''
这里存放一些参数
'''
import os

# 训练相关的超参数
num_epochs = 1
batch_size = 32
learning_rate = 1e-3
use_cuda = True

# 模型相关的超参数
img_embed_size = 500
word_embed_size = 500
hidden1_size = 1000  # 第一个LSTM层的隐层单元数目
hidden2_size = 1000  # 第二个KSTM层的隐层单元数目

# frame_shape = (512, 7, 7)  # 视屏帧特征的形状
frame_size = 4096  # 视频特征的维度
num_frames = 60  # 图像序列的规定长度
num_words = 30  # 文本序列的规定长度


# 数据相关的参数
# 提供两个数据集：MSR-VTT和MSVD
msrvtt_video_root = './MSR-VTT/TrainValVideo/'
msrvtt_anno_json_path = './MSR-VTT/train_val_videodatainfo.json'
msrvtt_video_sort_lambda = lambda x: int(x[5:-4])

msvd_video_root = './MSVD/youtube_videos'
msvd_csv_path = './MSVD/MSR Video Description Corpus.csv'
msvd_video_name2id_map = './MSVD/youtube_mapping.txt'
msvd_anno_json_path = './MSVD/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成
msvd_video_sort_lambda = lambda x: int(x[3:-4])

dataset = {
    'msr-vtt': [msrvtt_video_root, msrvtt_anno_json_path, msrvtt_video_sort_lambda],
    'msvd': [msvd_video_root, msvd_anno_json_path, msvd_video_sort_lambda]
}

# 用video_root和anno_json_path这两个变量来切换所使用的数据集
# video_sort_lambda用来对视频按照名称进行排序
ds = 'msvd'
video_root, anno_json_path, video_sort_lambda = dataset[ds]

feat_save_path = './feats'
vocab_pkl_path = os.path.join(feat_save_path, ds + '_vocab.pkl')
caption_pkl_path = os.path.join(feat_save_path, ds + '_captions.pkl')

video_h5_path = os.path.join(feat_save_path, ds + '_videos.h5')
video_h5_dataset = 'feats'


# checkpoint相关的超参数
vgg_checkpoint = './models/vgg16-00b39a1b.pth'
decoder_pth_path = ds + '_decoder.pth'
