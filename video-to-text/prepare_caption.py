import json
import pickle
from vocab import Vocabulary
import torch


def main():
    '''
    读取存储文本标注信息的json文件，
    并且将每一条caption以及它对应的video的id保存起来，
    放回caption word_id list和video_id list
    '''
    # 读取词汇字典
    with open('./vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # 读取json文件
    with open('./raw/train_val_videodatainfo.json', 'r') as f:
        json_data = json.load(f)
    json_data = json_data['sentences']
    captions = []
    video_ids = []
    for row in json_data:
        caption = row['caption'].lower()
        # 把caption用word id来表示
        tokens = []
        tokens.append(vocab('<start>'))
        for word in caption.split(' '):
            tokens.append(vocab(word))
        tokens.append(vocab('<end>'))
        captions.append(torch.LongTensor(tokens))
        video_ids.append(row['video_id'])
    with open('captions.pkl', 'wb') as f:
        pickle.dump([captions, video_ids])


if __name__ == '__main__':
    main()
