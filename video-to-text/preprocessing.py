# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import numpy as np
import skimage
import torch
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


def modify_vgg(net):
    net.classifier = nn.Sequential(
        net.classifier[0],
        net.classifier[1],
        net.classifier[2],
        net.classifier[3],
        net.classifier[4],
        )
    return net


def main():
    vgg = models.vgg16()
    vgg.load_state_dict(torch.load('./models/vgg16-00b39a1b.pth'))
    vgg = modify_vgg(vgg)
    vgg.eval()
    vgg.cuda()

    num_frames = 60
    video_root = './video'
    feat_save_path = './feats'
    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)

    videos = os.listdir(video_root)
    nvideos = len(videos)
    
    h5 = h5py.File(feat_save_path + 'frames.h5', 'w')
    feats = h5.create_dataset('frames', (nvideos), dtype=)

    for video in videos:
        print(video)
        save_name = video.split('.')[0] + '.npy'
        if os.path.exists(os.path.join(feat_save_path, save_name)):
            print('Already processed...')
            continue

        video_path = os.path.join(video_root, video)
        try:
            cap = cv2.VideoCapture(video_path)
        except:
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

        cropped_frame_list = np.array([preprocess_frame(x)
                                       for x in frame_list]).transpose((0, 3, 1, 2))
        cropped_frame_list = Variable(torch.from_numpy(cropped_frame_list),
                                      volatile=True).cuda()
        feats = vgg.features(cropped_frame_list).data

        save_full_path = os.path.join(feat_save_path, save_name)
        # np.save(save_full_path, feats)


if __name__ == '__main__':
    main()
