# coding: utf-8

import pickle
import h5py
import torch
import torch.utils.data as data


class MsrDataset(data.Dataset):
    '''
    MSR-VTT数据集的描述类，用来加载和提供数据
    构造的时候需要以下输入：
    1. 提供文本特征的pkl文件
    2. 包含视频帧信息的h5文件
    提供文本和视频h5特征，以及根据caption的id来返回数据
    '''

    def __init__(self, cap_pkl, video_h5):
        with open(cap_pkl, 'rb') as f:
            self.captions, self.video_ids = pickle.load(f)
        h5_file = h5py.File(video_h5, 'r')
        self.video_feats = h5_file['feats']

    def __getitem__(self, index):
        '''
        返回一个训练样本对（包含视频frame特征和对应的caption）
        根据caption来找对应的video，所以要求video存储的时候是按照id升序排列的
        '''
        caption = self.captions[index]
        video_id = self.video_ids[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, caption

    def __len__(self):
        return len(self.captions)


def collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    # 根据caption的长度对数据进行排序，
    # 之所以这么做是因为之后使用RNN时要求
    # 输入要按照sequence的长度降序排列
    data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, raw_captions = zip(*data)

    # 把视频合并在一起（把4D Tensor的序列变成5D Tensor）
    videos = torch.stack(videos, 0)

    # 把caption合并在一起（把1D Tensor的序列变成一个2D Tensor）
    lengths = [len(c) for c in raw_captions]
    captions = torch.zeros(len(raw_captions), max(lengths)).long()
    for i, cap in enumerate(raw_captions):
        end = lengths[i]
        captions[i, :end] = cap
    return videos, captions, lengths


def get_loader(cap_pkl, video_h5, batch_size=10, shuffle=True, num_workers=2):
    msr_vtt = MsrDataset(cap_pkl, video_h5)
    data_loader = torch.utils.data.DataLoader(dataset=msr_vtt,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    cap_pkl = './feats/captions.pkl'
    video_h5 = './feats/videos.h5'
    train_loader = get_loader(cap_pkl, video_h5)
    print(next(iter(train_loader)))
