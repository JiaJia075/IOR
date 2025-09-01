# coding=utf-8
import os
import time
import json


def sel_cat(anno_file, sel_num):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(anno_file, 'r'))
    assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))
    num_bbox = len(dataset['annotations'])
    print('num of bbox = '+str(num_bbox))

    order_of_classes = []

    # sort by cat_ids
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])
    # select first 40 cats
    sel_cats = dataset['categories'][:sel_num]
    # selected annotations
    sel_cats_ids = [cat['id'] for cat in sel_cats]
    order_of_classes.append(sel_cats_ids)
    sel_anno = []
    sel_first_image_ids = []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_first_image_ids.append(anno['image_id'])
    sel_first_image_ids = set(sel_first_image_ids)
    # selected images
    sel_images = []
    for image in dataset['images']:
        if image['id'] in sel_first_image_ids:
            sel_images.append(image)
    # selected dataset
    sel_dataset = dict()
    sel_dataset['order_of_classes'] = order_of_classes
    sel_dataset['categories'] = sel_cats
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images
    # writing results
    fp = open(os.path.splitext(anno_file)[0] + '_ior_sel_first_40_cats.json', 'w')
    json.dump(sel_dataset, fp)

    repeat = 0
    # select last 40 cats
    sel_cats = dataset['categories'][sel_num:]
    # selected annotations
    sel_cats_ids = [cat['id'] for cat in sel_cats]
    order_of_classes.append(sel_cats_ids)
    sel_anno = []
    sel_last_image_ids = []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            if anno['image_id'] in sel_first_image_ids:
                repeat += 1
            else:
                sel_anno.append(anno)
                sel_last_image_ids.append(anno['image_id'])
    sel_last_image_ids = set(sel_last_image_ids)
    # selected images
    sel_images = []
    for image in dataset['images']:
        if image['id'] in sel_last_image_ids:
            sel_images.append(image)
    # selected dataset
    sel_dataset = dict()
    sel_dataset['order_of_classes'] = order_of_classes
    sel_dataset['categories'] = dataset['categories']
    sel_dataset['annotations'] = sel_anno
    sel_dataset['images'] = sel_images
    # writing results
    fp = open(os.path.splitext(anno_file)[0] + '_ior_sel_last_40_cats_non_cooccurence.json', 'w')
    json.dump(sel_dataset, fp)
    print('num of repeat = '+str(repeat))


if __name__ == "__main__":
    anno_file = '/data1/home/anzijia/coco/annotations/instances_train2017.json'
    sel_num = 40
    sel_cat(anno_file, sel_num)
