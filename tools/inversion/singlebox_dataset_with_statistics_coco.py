# --------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Official PyTorch implementation of WACV2021 paper:
# Data-Free Knowledge Distillation for Object Detection
# A Chawla, H Yin, P Molchanov, J Alvarez
# --------------------------------------------------------

"""
Create randomized labels for COCO images
COCO labels are structured as: 
[0-79] x y w h 
where x,y,w,h are normalized to 0-1 and have 6 places after decimal and 1 place before decimal (0/1) 
e.g: 
1 0.128828 0.375258 0.249063 0.733333
0 0.476187 0.289613 0.028781 0.138099

To randomize: 
First generate width and height dimensions 
Then jitter the x/y labels
Then fix using max/min clipping
"""
import numpy as np
import random
import argparse
import os
from PIL import Image
from tqdm import tqdm
import shutil
MINDIM=0.6
MAXDIM=0.8
def check_and_rebuild_path(path):
    # 检查路径是否存在
    if os.path.exists(path):
        print(f"路径 {path} 存在，尝试删除并重建。")
        # 尝试删除，注意这里假设路径是一个目录
        try:
            # 如果目录为空，则使用os.rmdir，否则使用shutil.rmtree
            shutil.rmtree(path)
            print(f"已成功删除路径 {path}。")
        except OSError as e:
            print(f"删除路径 {path} 时出错：{e.strerror}")
            return
    else:
        print(f"路径 {path} 不存在，将创建它。")
    
    # 创建目录
    try:
        os.makedirs(path)
        print(f"路径 {path} 已成功创建。")
    except OSError as e:
        print(f"创建路径 {path} 时出错：{e.strerror}")

def populate(args):

    # folder
    check_and_rebuild_path(os.path.join(args.outdir, "images", "train2014"))
    check_and_rebuild_path(os.path.join(args.outdir, "labels", "train2014")) 

    # cat category
    cat_names = []
    cat_ids = []
    with open(args.CatOrder, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            name_id = line.strip().split(',')
            cat_names.append(name_id[0])
            cat_ids.append(name_id[1])

    with open(args.outdir+'/cat_order_coco.txt', 'w', encoding='utf-8') as file:
        for num in range(args.numClasses):
            file.write(cat_names[num]+','+cat_ids[num]+'\n')

    # 读取各个类的直方图
    cat_ratios = {}
    for cat in cat_names:
        hist_value = []
        hist_num = []
        with open(args.histogramdir+'/data_distribution_histogram_'+cat+'.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                value_num = line.split(',')
                value = float(value_num[0])
                num = int(value_num[1])
                hist_value.append(value)
                hist_num.append(num)
        num_all = sum(hist_num)
        # 把直方图中-1到1的那些值的数量给最近的不在这个范围内的值，并删除
        del_value = []#记录要删除的value值，按这个找索引删除hist_value和hist_num
        for index, value in enumerate(hist_value):
            if value>=-1 and value<=0:
                del_value.append(value)
                if hist_num[index] != 0:
                    hist_num[index-1] = hist_num[index-1]+hist_num[index]
            elif value>0 and value<1:
                del_value.append(value)
                if hist_num[index] != 0:
                    hist_num[index+1] = hist_num[index+1]+hist_num[index]
        ratios = []
        ratios_weight = []
        for index, value in enumerate(hist_value):
            if value not in del_value: 
                ratios.append(value)
                ratios_weight.append(hist_num[index])
        # 看看是否正常去除-1到1的值
        if sum(ratios_weight) != num_all:
            print(cat+'数量不一致')
            print(ratios_weight)
            print(hist_num)
        cat_ratios[cat] = [ratios, ratios_weight]

    for CatIdx in tqdm(range(args.numClasses)):
        for imgIdx in tqdm(range(args.numImages_per_class)):

            # class
            cls = cat_names[CatIdx]
            cls_id = cat_ids[CatIdx]
            [ratios, ratios_weight] = cat_ratios[cls]

            # box: w,h,x,y
            width   = MINDIM + (MAXDIM-MINDIM) * np.random.rand()
            # height  = MINDIM + (MAXDIM-MINDIM) * np.random.rand()
            selected_ratio = random.choices(ratios, ratios_weight, k=1)[0]
            if selected_ratio > 0:
                width  = MINDIM + (MAXDIM-MINDIM) * np.random.rand()
                height = width/selected_ratio
            elif selected_ratio < 0:
                height  = MINDIM + (MAXDIM-MINDIM) * np.random.rand()
                width   = -height/selected_ratio
            x       = 0.5 + (0.5-width/2.0)  * np.random.rand() * np.random.choice([1,-1])
            y       = 0.5 + (0.5-height/2.0) * np.random.rand() * np.random.choice([1,-1])
            assert x+width/2.0 <= 1.0, "overflow width, x+width/2.0={}".format(x+width/2.0)
            assert y+height/2.0<= 1.0, "overflow height, y+height/2.0={}".format(y+height/2.0)



            _label_str = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                int(cls_id), x, y, width, height
            )

            # im = Image.new(mode="RGB", size=(256,256), color=(127,127,127))
            im = Image.new(mode="RGB", size=(160,160), color=(127,127,127))
            # im = Image.new(mode="RGB", size=(320,320), color=(127,127,127))

            # save
            outfile = "coco_train2014_"+cls+"_{:012d}".format(imgIdx+1)
            im.save(os.path.join(args.outdir, "images", "train2014", outfile+".jpg"))
            with open(os.path.join(args.outdir, "labels", "train2014", outfile+".txt"), "wt") as f:
                f.write(_label_str)

    pathh = args.outdir+'/images/train2014'

    for filenames in os.walk(pathh):
        filenames = list(filenames)
        print(filenames)
        filenames_all = filenames[2]#取出文件名
        filedir = filenames[0]
        print(filedir)
        with open (args.outdir+'/img_path.txt','w') as f:
            for filename in filenames_all:
                print(filename)
                # f.write(pathh+filename+'\n')
                f.write('/'+os.path.join(pathh, filename)+'\n')
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='populate single box per image labels') 
    parser.add_argument('--numImages_per_class', type=int, default=1, help='number of images to generate') 
    parser.add_argument('--numClasses', type=int, default=60, help='number of classes')
    parser.add_argument('--outdir', default='', type=str, help='output directory') 
    parser.add_argument('--histogramdir', default='./histogram_cats_first_80_cats', type=str, help='output directory') 
    parser.add_argument('--CatOrder', default='./cat_order_coco_80_cats.txt', type=str, help='output directory') 

    args = parser.parse_args()

    populate(args)

