# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import numpy as np
from utils import preprocess_frame
from model import EncoderCNN
import torch
from torch.autograd import Variable
import h5py
from args import video_root, feat_save_path, video_h5_path, video_h5_dataset
from args import video_sort_lambda
from args import num_frames, frame_size


def main():
    encoder = EncoderCNN()
    encoder.eval()
    encoder.cuda()

    # 设置一下数据读取和保存的目录
    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)

    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root), key=video_sort_lambda)
    nvideos = len(videos)

    # 创建保存视频特征的hdf5文件
    if os.path.exists(video_h5_path):
        # 如果hdf5文件已经存在，说明之前处理过，或许是没有完全处理完
        # 使用r+ (read and write)模式读取，以免覆盖掉之前保存好的数据
        h5 = h5py.File(video_h5_path, 'r+')
        dataset_feats = h5[video_h5_dataset]
    else:
        h5 = h5py.File(video_h5_path, 'w')
        dataset_feats = h5.create_dataset(video_h5_dataset,
                                          (nvideos, num_frames, frame_size),
                                          dtype='float32')

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

        # 视频特征的shape是num_frames x 4096
        # 如果帧的数量小于num_frames，则剩余的部分用0补足
        feats = np.zeros((num_frames, frame_size), dtype='float32')
        feats[:frame_count, :] = encoder(cropped_frame_list).data.cpu().numpy()
        dataset_feats[i] = feats


if __name__ == '__main__':
    main()
