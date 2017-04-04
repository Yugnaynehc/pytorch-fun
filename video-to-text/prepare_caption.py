# coding: utf-8
'''
做两件事:
1. 对数据集进行train val test的划分
2. 把caption转换成token index表示然后存到picke中
'''

import json
import nltk
import pickle
from vocab import Vocabulary
import torch
from args import anno_json_path, split_json_path, vocab_pkl_path
from args import caption_train_pkl_path, caption_val_pkl_path, caption_test_pkl_path
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

    # 读取并初始化split信息
    with open(split_json_path, 'r') as f:
        split_json = json.load(f)
    split_dict = {int(i): s for i, s in split_json['split'].items()}
    # 初始化数据存储字典
    captions = {'train': [], 'val': [], 'test': []}
    lengths = {'train': [], 'val': [], 'test': []}
    video_ids = {'train': [], 'val': [], 'test': []}

    count = 0
    # 读取json文件
    with open(anno_json_path, 'r') as f:
        anno_json = json.load(f)
    anno_data = anno_json['sentences']

    for row in anno_data:
        caption = row['caption'].lower()
        video_id = int(row['video_id'][5:])
        split = split_dict[video_id]
        words = nltk.tokenize.word_tokenize(caption)
        l = len(words) + 1  # 加上一个<end>
        lengths[split].append(l)
        if l > num_words:
            # 如果caption长度超出了规定的长度，就做截取处理
            words = words[:num_words - 1]  # 最后要留一个位置给<end>
            count += 1
        # 把caption用word id来表示
        tokens = []
        for word in words:
            tokens.append(vocab(word))
        tokens.append(vocab('<end>'))
        while l < num_words:
            # 如果caption的长度少于规定的长度，就用<pad>（0）补齐
            tokens.append(vocab('<pad>'))
            l += 1
        captions[split].append(torch.LongTensor(tokens))
        video_ids[split].append(video_id)

    # 统计一下有多少的caption长度过长
    print('There are %.3f%% too long captions' % (100 * float(count) / len(anno_data)))

    # 分别对train val test这三个划分进行存储
    with open(caption_train_pkl_path, 'wb') as f:
        pickle.dump([captions['train'], lengths['train'], video_ids['train']], f)
        print('Save %d train captions to %s.' % (len(captions['train']),
                                                 caption_train_pkl_path))
    with open(caption_val_pkl_path, 'wb') as f:
        pickle.dump([captions['val'], lengths['val'], video_ids['val']], f)
        print('Save %d val captions to %s.' % (len(captions['val']),
                                               caption_val_pkl_path))
    with open(caption_test_pkl_path, 'wb') as f:
        pickle.dump([captions['test'], lengths['test'], video_ids['test']], f)
        print('Save %d test captions to %s.' % (len(captions['test']),
                                                caption_test_pkl_path))


if __name__ == '__main__':
    main()
