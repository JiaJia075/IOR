# -*- coding: utf-8 -*-
# @Time    : 20-2-13 下午5:03
# @Author  : wusaifei
# @FileName: Vision_data.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# 读取数据

ann_json = 'dataset/coco/annotations/instances_train2017.json'
with open(ann_json) as f:
    ann=json.load(f)

#################################################################################################
#创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
# with open('./cat_order_coco_first_40_cats.txt', 'w') as f:
with open('./cat_order_coco_80_cats.txt', 'w') as f:
    for key, value in category_dic.items():
        f.write(f"{value},{key}\n")
cat_ratios=dict([(i['name'],[]) for i in ann['categories']])
cat_hist={}
for i in ann['annotations']:
    w = i['bbox'][2]
    h = i['bbox'][3]
    if h>0 and w>0:
        if w/h >= 1.0:
            cat_ratios[category_dic[i['category_id']]].append(w/h)
        else:
            cat_ratios[category_dic[i['category_id']]].append(-h/w)

# with open('histogram_stats.txt', 'w') as f:
output_histogram_folder='histogram_cats_first_80_cats'
for cat, ratio in cat_ratios.items():
    ratio = sorted(ratio)
    start_index = round(len(ratio) * 0.05)
    end_index = round(len(ratio) * 0.95)
    ratio = ratio[start_index:end_index]
    hist, bin_edges = np.histogram(ratio, bins=50)
    if not os.path.exists(output_histogram_folder):
        os.makedirs(output_histogram_folder)  # 使用 makedirs 来递归创建目录
        print(f"文件夹 '{output_histogram_folder}' 不存在，已创建。")
    with open(output_histogram_folder+'/data_distribution_histogram'+'_'+cat+'.txt', 'w') as f:
        for edge, frequency in zip(bin_edges[:-1], hist):
            f.write(f"{edge},{frequency}\n")
    plt.hist(ratio, bins=50, alpha=0.7, color='blue')
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_histogram_folder+'/data_distribution_histogram'+'_'+cat+'.png')




