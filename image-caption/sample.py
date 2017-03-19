# coding: utf-8

import torchvision.transforms as T
import torch
from torch.autograd import Variable
from vocab import Vocabulary
from data import get_loader
import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"


def decode_tokens(tokens, vocab):
    words = []
    cnt = 0
    for token in tokens:
        word = vocab.idx2word[token]
        words.append(word)
        cnt += len(word)
        if cnt > 52:
            words.append('\n')
            cnt = 0
    caption = ' '.join(words)
    return caption


# 超参数设置
batch_size = 10
train_image_path = './data/train2014resized/'
train_json_path = './data/annotations/captions_train2014.json'


# 图像预处理
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 加载字典
with open('./vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 初始化数据加载器
train_loader = get_loader(train_image_path, train_json_path, vocab,
                          batch_size=batch_size, shuffle=True, num_workers=2)

# 载入预训练模型
encoder = torch.load('./encoder.pth')
decoder = torch.load('./decoder.pth')
encoder.cpu()
decoder.cpu()

plt.ion()
images, captions, _ = next(iter(train_loader))
# 以下的写法出了Bug，如果batch size是1，送入网络之后会出现生成的结果几乎一样
# 猜测是resnet出了问题
# for image, caption in zip(images, captions):
#     img = Variable(transform(image), volatile=True).unsqueeze(0)
#     # 对图像进行编码
#     feature = encoder(img)
#     # 对图像进行解码
#     tokens = decoder.sample(feature)
#     we = decode_tokens(tokens, vocab)
#     gt = decode_tokens(caption[1:], vocab)
#     # 显示图片和caption
#     plt.imshow(image.cpu().numpy().transpose((1, 2, 0)))
#     plt.suptitle('we: %s\ngt: %s' % (we, gt), fontsize=10,
#                  x=0.1, horizontalalignment='left')
#     plt.axis('off')
#     plt.pause(3)

imgs = torch.zeros(images.size())
for i, img in enumerate(images):
    imgs[i] = transform(img)
imgs = Variable(imgs, volatile=True)
features = encoder(imgs)
for i, f in enumerate(features):
    # 对图像进行解码
    tokens = decoder.sample(f.unsqueeze(0))
    we = decode_tokens(tokens, vocab)
    gt = decode_tokens(captions[i][1:], vocab)
    # 显示图片和caption
    plt.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
    plt.suptitle('we: %s\ngt: %s' % (we, gt), fontsize=10,
                 x=0.1, horizontalalignment='left')
    plt.axis('off')
    plt.pause(5)
