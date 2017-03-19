# coding: utf-8

import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import os
from PIL import Image
import nltk
import torchvision.transforms as T


class CocoDataset(data.Dataset):

    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        '''
        返回一个训练样本对（包含图片和caption）
        '''
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        sentence = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        # 把caption转换成id串
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    # 根据caption的长度对数据进行排序，
    # 之所以这么做是因为之后使用RNN时要求
    # 输入要按照sequence的长度降序排列
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, raw_captions = zip(*data)

    # 把图片合并在一起（把3D Tensor的序列变成一个4D Tensor）
    # stack是把一个个单独的数据堆叠，会生成一个新维度
    # cat是把两个数据序列合并，会改变已有的某个维度的大小
    images = torch.stack(images, 0)

    # 把caption合并在一起（把1D Tensor的序列变成一个2D Tensor）
    lengths = [len(cap) for cap in raw_captions]
    captions = torch.zeros(len(raw_captions), max(lengths)).long()
    for i, cap in enumerate(raw_captions):
        end = lengths[i]
        captions[i, :end] = cap
    return images, captions, lengths


def get_loader(root, json, vocab, transform=None, batch_size=100, shuffle=True,
               num_workers=2):
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
