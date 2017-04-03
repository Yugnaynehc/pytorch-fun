# coding: utf-8

'''
这里存放一些参数
'''
import os

# 训练相关的超参数
num_epochs = 100
batch_size = 200
learning_rate = 3e-4
use_cuda = True

# 模型相关的超参数
img_embed_size = 1000
word_embed_size = 500
hidden1_size = 1000  # 第一个LSTM层的隐层单元数目
hidden2_size = 1000  # 第二个KSTM层的隐层单元数目

# frame_shape = (512, 7, 7)  # 视频帧特征的形状
frame_size = 4096  # 视频特征的维度
frame_sample_rate = 10  # 视频帧的采样率
num_frames = 20  # 图像序列的规定长度
num_words = 30  # 文本序列的规定长度


# 数据相关的参数
# 提供两个数据集：MSR-VTT和MSVD
msrvtt_video_root = './MSR-VTT/TrainValVideo/'
msrvtt_anno_json_path = './MSR-VTT/train_val_videodatainfo.json'
msrvtt_split_json_path = './MSR-VTT/split.json'  # 自己生成一个数据集划分的json文件（build_msrvtt_split.py）
msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
msrvtt_train_range = None
msrvtt_val_range = None
msrvtt_test_range = None

msvd_video_root = './MSVD/youtube_videos'
msvd_csv_path = './MSVD/MSR Video Description Corpus.csv'
msvd_video_name2id_map = './MSVD/youtube_mapping.txt'
msvd_anno_json_path = './MSVD/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成（build_msvd_annotation.py）
msvd_split_json_path = './MSVD/split.json'  # 自己生成一个数据集划分的json文件（build_msvd_split.py）
msvd_video_sort_lambda = lambda x: int(x[3:-4])
msvd_train_range = (0, 1200)
msvd_val_range = (1200, 1300)
msvd_test_range = (1300, 1970)


dataset = {
    'msr-vtt': [msrvtt_video_root, msrvtt_video_sort_lambda,
                msrvtt_anno_json_path, msrvtt_split_json_path,
                msrvtt_train_range, msrvtt_val_range, msrvtt_test_range],
    'msvd': [msvd_video_root, msvd_video_sort_lambda,
             msvd_anno_json_path, msvd_split_json_path,
             msvd_train_range, msvd_val_range, msvd_test_range]
}

# 用video_root和anno_json_path这两个变量来切换所使用的数据集
# video_sort_lambda用来对视频按照名称进行排序
ds = 'msvd'
video_root, video_sort_lambda, anno_json_path, split_json_path, \
    train_range, val_range, test_range = dataset[ds]

feat_dir = 'feats'

vocab_pkl_path = os.path.join(feat_dir, ds + '_vocab.pkl')
caption_pkl_path = os.path.join(feat_dir, ds + '_captions.pkl')
caption_pkl_base = os.path.join(feat_dir, ds + '_captions')
caption_train_pkl_path = caption_pkl_base + '_train.pkl'
caption_val_pkl_path = caption_pkl_base + '_val.pkl'
caption_test_pkl_path = caption_pkl_base + '_test.pkl'

video_h5_path = os.path.join(feat_dir, ds + '_videos.h5')
video_h5_dataset = 'feats'


# 结果评估相关的参数
result_dir = 'results'
reference_txt_path = os.path.join(result_dir, 'references.txt')
predict_txt_path = os.path.join(result_dir, 'predictions.txt')


# checkpoint相关的超参数
vgg_checkpoint = './models/vgg16-00b39a1b.pth'  # 从caffe转换而来
# vgg_checkpoint = './models/vgg16-397923af.pth'  # 直接用pytorch训练的模型
decoder_pth_path = os.path.join(result_dir, ds + '_decoder.pth')
