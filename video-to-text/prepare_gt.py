# coding: utf-8

'''
准备ground-truth,用来评估结果的好坏
'''
import json
from args import test_range
from args import anno_json_path, reference_txt_path


reference_txt = open(reference_txt_path, 'w')

with open(anno_json_path, 'r') as f:
    anno_json = json.load(f)
anno_data = anno_json['sentences']

selected_range = range(*test_range)
error_count = 0
for row in anno_data:
    caption = row['caption'].lower()
    video_id = int(row['video_id'][5:])
    if video_id not in selected_range:
        continue

    # 把结尾的句号以及多余的句子去掉
    # s = caption.split('.')
    # caption = s[0]
    try:
        reference_txt.write('%d\t%s\n' % (video_id, caption))
    except Exception as e:
        print(caption)
        error_count += 1

print(error_count)
reference_txt.close()
