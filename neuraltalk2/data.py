from random import seed
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms as trn
from misc.resnet_utils import AttnResnet
import argparse
import json
import h5py
import os
import re
import numpy as np
import skimage.io
import time

resnet = models.resnet101()
attn_resnet = AttnResnet(resnet)

resnet.load_state_dict(torch.load('checkpoints/resnet101-5d3b4d8f.pth'))
attn_resnet.cuda()
attn_resnet.eval()


preprocess_fn = trn.Compose([
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def build_vocab(img_list, args):
    '''
    寻找出现频数小于阈值的单词，构建一个不含这些单词的词汇表
    '''
    # 计算各个单词出现的次数
    counts = {}
    for img in img_list:
        for s in img['sentences']:
            for w in s['tokens']:
                # 剔除带数字的词
                if re.compile(r'.*\d').match(w) is not None:
                    continue
                counts[w] = counts.get(w, 0) + 1
    # 按照单词出现的次数降序排序
    cw = sorted([(c, w) for w, c in counts.items()], reverse=True)
    # print('top words and their counts:')
    # print('\n'.join(map(str, cw[:20])))

    # 筛掉出现频数小于阈值的单词
    vocab = [w for c, w in cw if c >= args.word_count_threshold]
    if cw[-1][0] < args.word_count_threshold:
        vocab.append('<UNK>')
    return vocab


def encode_captions(imgs, args, wtoi):
    '''
    将每一条caption转换成对应的单词id串
    '''
    max_length = args.max_length
    # 数据集中的样本（图片）数，一个样本对应一张图片和多条caption
    nsamples = len(imgs)
    # 数据集中包含的caption总条数
    ncaptions_all = sum(len(img['sentences']) for img in imgs)

    all_captions = []
    # sample_start_idx和sample_end_idx用来记录每张图片的caption
    # 在all_captions列表的起止位置
    sample_start_idx = np.zeros(nsamples, dtype=np.uint32)
    sample_end_idx = np.zeros(nsamples, dtype=np.uint32)
    # 记录每一条caption的长度
    caption_length = np.zeros(ncaptions_all, dtype=np.uint32)

    # 记录当前处理的caption的条数
    caption_count = 0
    # 记录当前sample所对应的第一条caption在all_captions列表的起始位置
    sample_cursor = 0

    unk_id = wtoi['<UNK>']
    for sample_count, img in enumerate(imgs):
        sents = img['sentences']
        # 当前sample包含的caption数量
        ncaptions = len(sents)
        sample_captions = np.zeros((ncaptions, max_length), dtype=np.uint32)
        sample_start_idx[sample_count] = sample_cursor

        for i, s in enumerate(sents):
            tokens = s['tokens']
            l = min(len(tokens), max_length)
            caption_length[caption_count] = l
            caption_count += 1
            for j, w in enumerate(tokens[:l]):
                sample_captions[i][j] = wtoi.get(w, unk_id)

        all_captions.append(sample_captions)
        sample_end_idx[sample_count] = sample_cursor + ncaptions - 1
        sample_cursor += ncaptions

    all_captions = np.concatenate(all_captions, axis=0)
    assert all_captions.shape[0] == ncaptions_all, 'lengths don\'t match? that\'s weird'
    assert np.all(caption_length > 0), 'error: some caption had no words?'

    return all_captions, sample_start_idx, sample_end_idx, caption_length


def since(start):
    now = time.time()
    elapsed = now - start
    m = elapsed // 60
    s = elapsed - 60 * m
    return '%d:%d' % (m, s)


def main(args):
    # 建立存放处理好的数据的文件夹
    output_root = args.output_root
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    # 让h5和json文件放到该文件夹中
    output_h5 = os.path.join(output_root, args.output_h5)
    output_json = os.path.join(output_root, args.output_json)

    # 载入Karpathy处理好的caption数据
    raw_data = json.load(open(args.input_json, 'r'))
    img_list = raw_data['images']
    seed(123)

    # 创建一个词汇表
    vocab = build_vocab(img_list, args)
    idx2word = {i + 1: w for i, w in enumerate(vocab)}  # 起始索引是1, 因为0用来表示<EOS>
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}

    # 把每一条caption用单词id串来表示
    all_captions, sample_start_idx, sample_end_idx, caption_length = encode_captions(img_list, args, word2idx)

    # 创建输出的h5文件
    nimages = len(img_list)
    # f_lb记录标签信息
    f_lb = h5py.File(output_h5 + '_label.h5', 'w')
    f_lb.create_dataset('all_captions', dtype='uint32', data=all_captions)
    f_lb.create_dataset('sample_start_idx', dtype='uint32', data=sample_start_idx)
    f_lb.create_dataset('sample_end_idx', dtype='uint32', data=sample_end_idx)
    f_lb.create_dataset('caption_length', dtype='uint32', data=caption_length)
    f_lb.close()

    # f_fc记录图像特征
    f_fc = h5py.File(output_h5 + '_fc.h5', 'w')
    dataset_fc = f_fc.create_dataset('fc', (nimages, 2048), dtype='float32')

    # f_att记录attention信息
    f_att = h5py.File(output_h5 + '_att.h5', 'w')
    dataset_att = f_att.create_dataset('att', (nimages, 14, 14, 2048), dtype='float32')

    # 初始化输出的json文件的内容
    out_json = {}
    out_json['idx2word'] = idx2word
    out_json['images'] = []

    start = time.time()
    for i, img in enumerate(img_list):
        # 读取一些元数据
        split = img['split']
        filepath = img['filepath']
        filename = img['filename']

        # 载入图片
        I = skimage.io.imread(os.path.join(args.images_root, filepath, filename))

        # 处理一下灰度图片
        if len(I.shape) == 2:
            # 以下两行的作用是把一副单通道的图片复制成三通道图片
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = Variable(preprocess_fn(I).unsqueeze(0), volatile=True)  # 放到一个mini batch中
        tmp_fc, tmp_att = attn_resnet(I)

        # 将得到的图像特征写入h5数据库中
        dataset_fc[i] = tmp_fc.data.cpu().float().numpy()
        dataset_att[i] = tmp_fc.data.cpu().float().numpy()

        if i % 1000 == 999:
            print('processing %d/%d\t(%.2f%% done)\t%s' %
                  (i + 1, nimages, 100 * (i + 1) / nimages, since(start)))

        # 顺手将图片信息存入到json文件中
        img_json = {}
        img_json['split'] = split
        img_json['filename'] = filename
        if 'cocoid' in img:
            img_json['id'] = img['cocoid']

        out_json['images'].append(img_json)

    f_fc.close()
    f_att.close()
    print('wrote ', output_h5)

    # 保存json文件
    json.dump(out_json, open(output_json, 'w'))
    print('wrote ', output_json)
    print('good job!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-json', default='./coco/dataset_coco.json')
    parser.add_argument('--output-root', default='data')
    parser.add_argument('--output-json', default='metadata.json')
    parser.add_argument('--output-h5', default='h5data')

    parser.add_argument('--max-length', default=16, type=int)
    parser.add_argument('--images-root', default='./coco')
    parser.add_argument('--word-count-threshold', default=5, type=int)

    args = parser.parse_args()
    main(args)
