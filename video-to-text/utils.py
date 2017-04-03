# coding: utf-8
import cv2
import numpy as np
from torch.autograd import Variable
import skimage


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]
    image = skimage.img_as_float(image).astype(np.float32)
    image -= np.array([103.939, 116.779, 123.68])
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


def decode_tokens(tokens, vocab):
    '''
    根据word id（token）列表和给定的字典来得到caption
    '''
    words = []
    for token in tokens:
        if token == vocab('<end>'):
            # words.append('.')
            break
        word = vocab.idx2word[token]
        words.append(word)
    caption = ' '.join(words)
    return caption


def var_wrap(tensor, use_cuda):
    '''
    根据use_cuda判断是否用cuda版本的Variable
    '''
    v = Variable(tensor)
    if use_cuda:
        v = v.cuda()
    return v
