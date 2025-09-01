# dataset settings
dataset_type = 'CocoDataset'
# data_root = '../data/coco/'
# data_root = '/data1/home/anzijia/VOCdevkit/coco_for_VOC2007/'
data_root = '/data1/home/anzijia/VOCdevkit/coco/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=(512, 512)),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

#augment
# img_scale = (1000, 600)  # width, height
# train_pipeline = [
#     # dict(type='LoadImageFromFile', backend_args=backend_args),
#     # dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.5, 1.5),
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict( 
#         type='MixUp', 
#         img_scale=img_scale, 
#         ratio_range=(0.8, 1.6), 
#         pad_val=114.0), 
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='RandomFlip', prob=0.5),
#     # Resize and Pad are for the last 15 epochs when Mosaic and
#     # RandomAffine are closed by YOLOXModeSwitchHook.
#     dict(type='Resize', scale=img_scale, keep_ratio=True),
#     dict(
#         type='Pad',
#         pad_to_square=True,
#         pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='PackDetInputs')
# ]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

#augment
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
# train_dataset = dict(
#     # use MultiImageMixDataset wrapper to support mosaic and mixup
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         pipeline=[
#             dict(type='LoadImageFromFile', backend_args=backend_args),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         backend_args=backend_args),
#     pipeline=train_pipeline)
# train_dataloader = dict(
#     batch_size=8,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=train_dataset)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    classwise = True,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    classwise = True,
    backend_args=backend_args)

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
