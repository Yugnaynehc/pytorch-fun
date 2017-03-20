# coding: utf-8

import os
import json
import pickle
from vocab import Vocabulary
import torch
from args import anno_json_path
from args import feat_save_path, vocab_pkl_path, caption_pkl_path
from args import num_words  # 文本序列的规定长度


def main():
    '''
    读取存储文本标注信息的json文件，
    并且将每一条caption以及它对应的video的id保存起来，
    放回caption word_id list和video_id list
    '''
    # 读取词汇字典
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    count = 0
    # 读取json文件
    with open(anno_json_path, 'r') as f:
        json_data = json.load(f)
    json_data = json_data['sentences']
    captions = []
    lengths = []
    video_ids = []
    for row in json_data:
        caption = row['caption'].lower()
        words = caption.split(' ')
        l = len(words) + 1  # 加上一个<end>
        lengths.append(l)
        if l > num_words:
            # 如果caption长度超出了规定的长度，就做截取处理
            words = words[:num_words - 1]
            count += 1
        # 把caption用word id来表示
        tokens = []
        # tokens.append(vocab('<start>'))
        for word in words:
            tokens.append(vocab(word))
        tokens.append(vocab('<end>'))
        while l < num_words:
            # 如果caption的长度少于规定的长度，就用<pad>（0）补齐
            tokens.append(vocab('<pad>'))
            l += 1
        captions.append(torch.LongTensor(tokens))
        video_ids.append(int(row['video_id'][5:]))

    # 统计一下有多少的caption长度过长
    print('There are %.3f%% too long captions' % (100 * float(count) / len(captions)))

    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)
    with open(caption_pkl_path, 'wb') as f:
        pickle.dump([captions, lengths, video_ids], f)
    print('Save captions to %s.' % caption_pkl_path)


if __name__ == '__main__':
    main()
