# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import glob
import cv2
import numpy as np
import skimage
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import h5py


def preprocess_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def cut_vgg(net):
    '''
    用一种笨方法把VGG的最后一个fc层（及其之前的ReLU层）剔除掉
    '''
    net.classifier = nn.Sequential(
        net.classifier[0],
        net.classifier[1],
        net.classifier[2],
        net.classifier[3],
        net.classifier[4],
    )
    return net


def main():
    # 加载一下VGG16模型
    vgg = models.vgg16()
    vgg.load_state_dict(torch.load('./models/vgg16-00b39a1b.pth'))
    # 把最后一个fc层剔除掉，之后在训练的时候会引入一个新fc层，用来做图像的embedding
    vgg = cut_vgg(vgg)
    vgg.eval()
    vgg.cuda()

    # 从视频中等间隔抽取60帧
    num_frames = 60
    # 设置一下数据读取和保存的目录
    video_root = './video'
    feat_save_path = './feats'
    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)

    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root), key=lambda x: int(x[5:-4]))
    nvideos = len(videos)

    # 创建保存视频特征的hdf5文件
    h5 = h5py.File(feat_save_path + 'videos.h5', 'w')
    dataset_feats = h5.create_dataset('feats', (nvideos, num_frames, 512, 7, 7), dtype='float32')

    for i, video in enumerate(videos):
        print(video)

        video_path = os.path.join(video_root, video)
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print('Can not open %s.' % video)
            pass

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame_list.append(frame)
            frame_count += 1

        frame_list = np.array(frame_list)
        if frame_count > num_frames:
            frame_indices = np.linspace(0, frame_count,
                                        num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]
            frame_count = num_frames

        # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
        cropped_frame_list = np.array([preprocess_frame(x)
                                       for x in frame_list]).transpose((0, 3, 1, 2))
        cropped_frame_list = Variable(torch.from_numpy(cropped_frame_list),
                                      volatile=True).cuda()

        # 视频特征的shape是num_frames x 512 x 7 x 7
        # 如果帧的数量小于num_frames，则剩余的部分用0补足
        feats = torch.LongTensor.zeros(num_frames, 512, 7, 7)
        feats[:frame_count, :] = vgg.features(cropped_frame_list).data
        dataset_feats[i] = feats


if __name__ == '__main__':
    main()
