# coding: utf-8
import cv2
import numpy as np
from torch.autograd import Variable
import skimage
from graphviz import Digraph


def preprocess_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]
    image = skimage.img_as_ubyte(image).astype(np.float32)
    # 减去在ILSVRC数据集上的图像的均值（BGR格式）
    image -= np.array([103.939, 116.779, 123.68])
    # image -= np.array([104.00698793, 116.66876762, 122.67891434])
    # 把BGR的图片转换成RGB的图片，因为之后的模型（caffe预训练版）用的是RGB格式
    image = image[:, :, ::-1]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[
            :, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[
            cropping_length:resized_image.shape[0] - cropping_length]
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


def make_dot(var):
    '''
    对网络结构进行可视化表示
    来自:https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py
    '''
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + ', '.join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot
