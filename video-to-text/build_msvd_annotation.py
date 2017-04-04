# coding: utf-8
'''
仿照MSR-VTT数据集的格式，为MSVD数据集生成一个包含video信息和caption标注的json文件
之所以要和MSR-VTT的格式相似，是因为所有的数据集要共用一套prepare_captions的代码
'''
import json
from args import msvd_csv_path, msvd_anno_json_path
from args import msvd_video_name2id_map
import pandas as pd


# 首先根据MSVD数据集官方提供的CSV文件确定每段视频的名字
video_data = pd.read_csv(msvd_csv_path, sep=',')
video_data = video_data[video_data['Language'] == 'English']
video_data['VideoName'] = video_data.apply(lambda row: row['VideoID'] + '_' +
                                           str(row['Start']) + '_' +
                                           str(row['End']), axis=1)
# 然后根据youtubeclips整理者提供的视频名字到视频id的映射构建一个词典
video_name2id = {}
with open(msvd_video_name2id_map, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name, vid = line.strip().split()
        # name = name[1:]  # 剔除掉视频名字开头的'-'符号
        # 提取出视频的数字id
        # 减1是因为id是从1开始的，但是之后处理的时候我们默认是0开始的
        # 因为实际上我们关系的是顺序，所以减1并不影响什么
        vid = int(vid[3:]) - 1
        # 再把vid变成video+数字id的形式
        # 不要问我为什么这么做<摊手>，因为MSR-VTT是这样的，好蠢啊...
        vid = 'video%d' % vid
        video_name2id[name] = vid

# 开始准备按照MSR-VTT的结构构造json文件
sents_anno = []
for name, desc in zip(video_data['VideoName'], video_data['Description']):
    if name not in video_name2id:
        print(name)
        continue
    # 有个坑，SKhmFSV-XB0这个视频里面有一个caption的内容是NaN
    if type(desc) == float:
        print(name, desc)
        continue
    d = {}
    # 还有很多新的坑! 有的句子带有一大堆\n或者带有\r\n
    desc = desc.replace('\n', '')
    desc = desc.replace('\r', '')
    # 有的句子有句号结尾,有的没有,甚至有的有多句.把句号以及多于一句的内容去掉
    desc = desc.split('.')[0]
    d['caption'] = desc
    d['video_id'] = video_name2id[name]
    sents_anno.append(d)


anno = {'sentences': sents_anno}
with open(msvd_anno_json_path, 'w') as f:
    json.dump(anno, f)
