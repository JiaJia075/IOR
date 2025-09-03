# coding=utf-8
import os
import time
import json
import shutil
import torch
import random
import numpy as np
import argparse
from PIL import Image
def compute_overlap(a, b, thr=0.2):
    """ compute the overlap of input a and b;
        input: a and b are the box
        output(bool): the overlap > 0.2 or not
    """

    iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]) + 1
    ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    intersection = iw*ih

    # this parameter can be changes for different datasets
    if intersection/a_area > thr or intersection/b_area > thr/3:
        return intersection/b_area, True
    else:
        return intersection/b_area, False
def resize_gen_img(img_shape, gen_img, gen_cat_id):
        # Calculate mean size of current input image and box
        im_mean_size = np.mean(img_shape)
        gen_img_shape_ori = gen_img.size
        box_mean_size = np.mean(gen_img_shape_ori)
        # # Modify the box size based on mean sizes
        # if float(box_mean_size) >= float(im_mean_size*0.3) and float(box_mean_size) <= float(im_mean_size*0.7):
        #     box_scale = 1.0
        # else:
        #     box_scale = random.uniform(float(im_mean_size*0.4), float(im_mean_size*0.6)) / float(box_mean_size)
        box_scale = 1.0
        # Resize the box image
        gen_img = gen_img.resize((int(box_scale * gen_img_shape_ori[0]), int(box_scale * gen_img_shape_ori[1])))
        
        # Define ground truth boxes
        gen_boxes = [0, 0, gen_img.size[0], gen_img.size[1], gen_cat_id]
        return gen_img, np.array([gen_boxes])
def mix_up(img, img_targets, gen, alpha=2.0, beta=5.0):
    """ Mixup the input image

    Args:
        image : the original image
        targets : the original image's targets
    Returns:
        mixupped images and targets
    """
    img_array = np.array(img)
    # image.flags.writeable = True
    img_shape = img_array.shape

    # 把img_bbox单拎出来
    gts = []
    for img_target in img_targets:
        bbox = img_target['bbox']
        bbox = x1y1wh2xyxy(bbox).tolist()
        category_id = img_target['category_id']
        bbox.append(category_id)
        gts.append(bbox)

    # make sure the image has more than one targets
    # If the only target occupies 75% of the image, we abandon mixupping.
    for gt in gts:
        bbox_w = gt[2] - gt[0]
        bbox_h = gt[3] - gt[1]
        if (img_shape[1]-bbox_w)<(img_shape[1]*0.25) and (img_shape[0]-bbox_h)<(img_shape[0]*0.25):
            return img, gts

    ##### For normal mixup ######
    # lambda: Sampling from a beta distribution 
    Lambda = torch.distributions.beta.Beta(alpha, beta).sample().item()
    num_mixup = 7 # more mixup boxes but not all used
            
    mixup_count = 0
    for i in range(num_mixup):
        gen_img, gen_cat_id, gen_bbox = next(gen)
        gen_img_resize, gen_bbox_resize = resize_gen_img(img_shape, gen_img, gen_cat_id)
        gen_img_resize = np.asarray(gen_img_resize)
        _gen_bbox_resize = gen_bbox_resize.copy()

        # assign a random location
        pos_x = random.randint(0, int(img_shape[1] * 0.6))
        pos_y = random.randint(0, int(img_shape[0] * 0.4))
        new_gt = [gen_bbox_resize[0][0] + pos_x, gen_bbox_resize[0][1] + pos_y, gen_bbox_resize[0][2] + pos_x, gen_bbox_resize[0][3] + pos_y]

        restart = True
        overlap = False
        max_iter = 0
        iter = 20
        # compute the overlap with each gts in image
        while restart:
            for g in gts: 
                _, overlap = compute_overlap(g, new_gt)
                # _, overlap_g = compute_overlap(g, new_gt)     
                # overlap = overlap or overlap_g
                if max_iter >= iter:
                    # if iteration > iter, delete current choosed sample
                    restart = False
                elif max_iter < iter/2 and overlap:
                    pos_x = random.randint(0, int(img_shape[1] * 0.6))
                    pos_y = random.randint(0, int(img_shape[0] * 0.4))
                    new_gt = [gen_bbox_resize[0][0] + pos_x, gen_bbox_resize[0][1] + pos_y, gen_bbox_resize[0][2] + pos_x, gen_bbox_resize[0][3] + pos_y]
                    max_iter += 1
                    restart = True
                    break
                elif iter > max_iter >= iter/2 and overlap:
                    # if overlap is True, then change the position at right bottom
                    pos_x = random.randint(int(img_shape[1] * 0.4), img_shape[1])
                    pos_y = random.randint(int(img_shape[0] * 0.6), img_shape[0])
                    new_gt = [pos_x-(gen_bbox_resize[0][2]-gen_bbox_resize[0][0]), pos_y-(gen_bbox_resize[0][3]-gen_bbox_resize[0][1]), pos_x, pos_y]
                    max_iter += 1
                    restart = True
                    break
                else:
                    restart = False
                    # print("!!!!{2} the g {0} and new_gt is: {1}".format(g, new_gt, overlap))

        if max_iter < iter:
            a, b, c, d = 0, 0, 0, 0
            if new_gt[3] >= img_shape[0]:
                # at bottom right new gt_y is or not bigger
                a = new_gt[3] - img_shape[0]
                new_gt[3] = img_shape[0]
            if new_gt[2] >= img_shape[1]:
                # at bottom right new gt_x is or not bigger
                b = new_gt[2] - img_shape[1]
                new_gt[2] = img_shape[1]
            if new_gt[0] < 0:
                # at top left new gt_x is or not bigger
                c = -new_gt[0]
                new_gt[0] = 0
            if new_gt[1] < 0:
                # at top left new gt_y is or not bigger
                d = -new_gt[1]
                new_gt[1] = 0

            # Use the formula by the paper to weight each image
            img_change = Lambda*img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]]
            gen_img_resize = (1-Lambda)*gen_img_resize
            # img_change = 0*img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]]
            # gen_img_resize = gen_img_resize
            
            # Combine the images
            if a == 0 and b == 0:
                if c == 0 and d == 0:
                    img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[:, :]
                elif c != 0 and d == 0:
                    img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[:, c:]
                elif c == 0 and d != 0:
                    img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[d:, :]
                else:
                    img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[d:, c:]

            elif a == 0 and b != 0:
                img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[:, :-b]
            elif a != 0 and b == 0:
                img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[:-a, :]
            else:
                img_array[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img_change + gen_img_resize[:-a, :-b]

            _gen_bbox_resize[0][:-1] = [new_gt[0]+gen_bbox[0],new_gt[1]+gen_bbox[1],new_gt[0]+gen_bbox[2],new_gt[1]+gen_bbox[3]]
            if len(gts) == 0:
                gts = _gen_bbox_resize
            else:
                gts = np.insert(gts, 0, values=_gen_bbox_resize, axis=0)

        mixup_count += 1
        if mixup_count>=6:
            break

    Current_image = Image.fromarray(np.uint8(img_array))

    return Current_image, gts


def list_folders(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folders
def xywh2x1y1wh(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    # y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[2] # w
    y[3] = x[3] # h
    return y

def x1y1wh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[0] = x[0] # top left x
    y[1] = x[1] # top left y
    y[2] = x[0] + x[2] # w
    y[3] = x[1] + x[3] # h
    return y

def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    # y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

def xyxy2x1y1wh(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    # y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    y[0] = x[0] # top left x
    y[1] = x[1] # top left y
    y[2] = x[2] - x[0]  # bottom right x
    y[3] = x[3] - x[1]  # bottom right y
    return y

def clear_directory(combine_dataset_img_path):
    # 检查路径是否存在
    if os.path.exists(combine_dataset_img_path):
        # 如果存在，删除整个路径
        shutil.rmtree(combine_dataset_img_path)
        print(f"Path '{combine_dataset_img_path}' already exists. Removing and recreating.")

    # 创建新的路径
    os.makedirs(combine_dataset_img_path)
    print(f"Path '{combine_dataset_img_path}' created.")

def torch_random_cycle(gen_name_list, generate_img_path, generate_label_path, order_of_classes):
    i = 0
    # 旧类目标合成一个列表
    all_but_last = order_of_classes[:-1]
    last = order_of_classes[-1]
    merged_list = [item for sublist in all_but_last for item in sublist]
    order_of_classes = [merged_list, last]

    while True:
        print(i)
        i=i+1
        indices = torch.randperm(len(gen_name_list))
        for idx in indices:
            img_path = os.path.join(generate_img_path,gen_name_list[idx]+'.jpg')
            label_path = os.path.join(generate_label_path,gen_name_list[idx]+'.txt')
            # 读img
            img = Image.open(img_path)
            width = img.width
            height = img.height
            # 读bbox
            with open(label_path, 'r') as file:
                lines = file.readlines()
            cat_bbox = lines[0].split(' ')
            bbox = [float(cat_bbox[1]),float(cat_bbox[2]),float(cat_bbox[3]),float(cat_bbox[4])]
            for w in [0,2]:
                bbox[w] = bbox[w] * width
            for h in [1,3]:
                bbox[h] = bbox[h] * height
            bbox = xywh2xyxy(bbox).tolist()  

            bbox_cat_id = order_of_classes[0][int(cat_bbox[0])]

            # # 把目标crop出来
            # cropped_img = img.crop(bbox)
            # cropped_img.save('/data/anzijia/coco/1.jpg')  
            yield img, bbox_cat_id, bbox

def add_generate(anno_file, generate_path, img_path, combine_dataset_img_path, combine_dataset_json_path):
    clear_directory(combine_dataset_img_path)

    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(anno_file, 'r'))
    assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))

    order_of_classes = dataset['order_of_classes']

    dataset_new = dict()

    dataset_new['order_of_classes'] = order_of_classes
    dataset_new['categories'] = dataset['categories']
    dataset_new['annotations'] = []
    dataset_new['images'] = []

    # 重新编序号
    annotation_id = 0

    generate_img_path = os.path.join(generate_path, 'coco/images/train2014')
    generate_label_path = os.path.join(generate_path, 'coco/labels/train2014')
    # 获得生成图片的文件名列表
    gen_name_list = []
    gen_imgs_path_list = os.listdir(generate_img_path)
    for img_name in gen_imgs_path_list:
        gen_name, _ = os.path.splitext(img_name)
        gen_name_list.append(gen_name)
    gen = torch_random_cycle(gen_name_list, generate_img_path, generate_label_path, dataset['order_of_classes'])

    for img_info in dataset['images']:
        # 取真实图像
        img_name = os.path.join(img_path,img_info['file_name'])
        img = Image.open(img_name)
        if img.mode == 'L':
            img_mode = 'gray'
        elif img.mode == 'RGB':
            img_mode = 'RGB'
        else:
            print("图像模式为:", img.mode)

        if img_mode == 'gray':
            img = img.convert("RGB")
        # 取真实图像的标签
        img_targets = [ann for ann in dataset['annotations'] if ann['image_id'] == img_info['id']]
        # # 看bbox对不对
        # for img_target in img_targets:
        #     bbox = img_target['bbox']
        #     bbox = x1y1wh2xyxy(bbox)
        #     cropped_img = img.crop(bbox)
        #     cropped_img.save('/data/anzijia/coco/1.jpg')  
   
        mix_up_img, mix_up_targets = mix_up(img, img_targets, gen, alpha=2.0, beta=2.0)
        # 看看mixup对不对
        # for mix_up_target in mix_up_targets:
        #     bbox = mix_up_target[0:-1]
        #     cropped_img = mix_up_img.crop(bbox)
        #     cropped_img.save('/data/anzijia/coco/1.jpg')  

        # save new images
        new_img_path = os.path.join(combine_dataset_img_path, img_info['file_name'])
        if img_mode == 'gray':
            mix_up_img = mix_up_img.convert("L")
        mix_up_img.save(new_img_path)

        # img information
        width = img_info['width']
        height = img_info['height']
        img_id = img_info['id']
        file_name = img_info['file_name']
        # add img information
        img_info = {
            'id': img_id, 
            'file_name': file_name,
            'height': height,
            'width': width}
        
        dataset_new['images'].append(img_info)

        # bbox information
        for mix_up_target in mix_up_targets:
            bbox = mix_up_target[0:-1]
            bbox = xyxy2x1y1wh(bbox).tolist()
            segmentation=None
            area = bbox[2]*bbox[3]
            ignore = 0
            iscrowd = 0
            category_id = int(mix_up_target[-1])

            # add annotations information
            anootation_info = {
                'segmentation': segmentation,
                'area': area,
                'ignore': ignore,
                'iscrowd': iscrowd,
                "image_id": img_id, 
                'bbox': bbox,
                'category_id': category_id,
                'id': annotation_id
            }
            annotation_id = annotation_id + 1
            dataset_new['annotations'].append(anootation_info)

    # writing results
    fp = open(combine_dataset_json_path, 'w')
    json.dump(dataset_new, fp)
    fp.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', default='', type=str, help="ori annotation json")
    parser.add_argument('--sampled_annotation_path', default='', type=str, help="sampled annotation json")
    parser.add_argument('--img_path', default='', type=str, help="image path")
    parser.add_argument('--agumented_dataset_img_path', default='', type=str, help="agumented replay dataset json image path")
    parser.add_argument('--agumented_dataset_json_path', default='', type=str, help="agumented replay dataset json path")

    args = parser.parse_args()
    add_generate(args.anno_file, args.sampled_annotation_path, args.img_path, args.agumented_dataset_img_path, args.agumented_dataset_json_path)
