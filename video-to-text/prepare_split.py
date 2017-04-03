# coding: utf-8

'''
为数据集生成train，val，test的划分。MSVD数据集可以根据Vsubhashini的划分：
train:1-1200, val:1201-1300, test:1301-1970
'''

import json
from args import split_json_path
from args import train_range, val_range, test_range


split_dict = {}

for i in range(*train_range):
    split_dict[i] = 'train'
for i in range(*val_range):
    split_dict[i] = 'val'
for i in range(*test_range):
    split_dict[i] = 'test'

split = {'split': split_dict}
with open(split_json_path, 'w') as f:
    json.dump(split, f)
