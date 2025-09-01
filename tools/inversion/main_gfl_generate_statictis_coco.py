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


from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]='3'
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data
from torchvision import datasets, transforms
import torchvision.utils as vutils
import argparse
import numpy as np
 
import functools
import random
from PIL import Image 
from mmdet.apis import init_detector, inference_detector

from inversion_tools.deepinversion_gfl import DeepInversionClass

from inversion_tools.inversion_datasets import load_batch, LoadImagesAndLabels, hyp


from inversion_tools.inversion_utils import draw_targets, convert_to_coco

import time


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_all_seeds(seeds): 
    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[3])



def run(args, dataiter):
    # device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    net = init_detector(args.config, args.inversion_checkpoint, device='cuda:0')

    # imgs, targets, imgspaths = load_batch(args.train_txt_path, args.bs, args.resolution[0], args.shuffle)
    _, targets, imgspaths, _ = next(dataiter)
    # imgs = imgs.float()/255.0
    net.eval() 

    args.start_noise = True

    parameters = dict()
    # Data augmentation params
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["do_flip"] = args.do_flip
    parameters["jitter"] = args.jitter
    parameters["rand_brightness"] = args.rand_brightness 
    parameters["rand_contrast"]   = args.rand_contrast
    parameters["random_erase"]    = args.random_erase
    parameters["mean_var_clip"] = args.mean_var_clip
    # Other params
    parameters["resolution"] = args.resolution
    parameters["bs"] = args.bs 
    parameters["iterations"] = args.iterations
    parameters["save_every"] = args.save_every
    parameters["display_every"] = args.display_every
    parameters["beta1"] = args.beta1
    parameters["beta2"] = args.beta2
    parameters["nms_params"] = args.nms_params
    parameters["cosine_layer_decay"] = args.cosine_layer_decay
    parameters["min_layers"] = args.min_layers
    parameters["num_layers"] = args.num_layers
    parameters["p_norm"] = args.p_norm
    parameters["alpha_mean"] = args.alpha_mean
    parameters["alpha_var"]  = args.alpha_var
    parameters["alpha_ssim"] = args.alpha_ssim

    # Bounding box samper
    parameters["box_sampler"]        = args.box_sampler
    parameters["box_sampler_warmup"] = args.box_sampler_warmup
    parameters["box_sampler_conf"]   = args.box_sampler_conf
    parameters["box_sampler_overlap_iou"] = args.box_sampler_overlap_iou
    parameters["box_sampler_minarea"]= args.box_sampler_minarea
    parameters["box_sampler_maxarea"]= args.box_sampler_maxarea
    parameters["box_sampler_earlyexit"] = args.box_sampler_earlyexit

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["wd"] = args.wd
    coefficients["lr"] = args.lr
    coefficients["min_lr"] = args.min_lr
    coefficients["first_bn_coef"] = args.first_bn_coef
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["alpha_img_stats"] = args.alpha_img_stats

    network_output_function = lambda x: x[1] # When in .eval() mode, DarkNet returns (inference_output, training_output). 

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             path=args.save_path,
                                             logger_big=None,
                                             parameters=parameters,
                                             use_amp=args.fp16,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function)

    # initialize inputs
    if args.init_chkpt.endswith(".pt"):
        initchkpt = torch.load(args.init_chkpt, map_location=torch.device("cpu"))
        init = initchkpt["images"]
        imgs = initchkpt["origimages"]
        targets = initchkpt["targets"]
        imgspaths = initchkpt["imgspaths"]
        init, imgs, imgspaths = init[0:args.bs], imgs[0:args.bs], imgspaths[0:args.bs]
        targets = targets[targets[:,0]<args.bs]
        if init.shape[2] != args.resolution[0]:
            init = F.interpolate(init, size=(args.resolution[0], args.resolution[1]))
            imgs = F.interpolate(imgs, size=(args.resolution[0], args.resolution[1]))
    else:
        init = torch.randn((args.bs, 3, args.resolution[0], args.resolution[1]), dtype=torch.float)
        init = torch.clamp(init, min=0.0, max=1.0)
        init = (args.init_scale * init) + args.init_bias
        # init = (args.real_mixin_alpha)*imgs + (1.0-args.real_mixin_alpha)*init
    DeepInversionEngine.save_image(init, os.path.join(DeepInversionEngine.path, "initialization.jpg"), halfsize=True)


    generatedImages, targets = DeepInversionEngine.generate_batch(targets, init)

    # Store generatedImages in coco format
    if args.save_coco:
        if not os.path.exists(os.path.join(args.save_path, "coco", "images", "train2014")):
            os.makedirs(os.path.join(args.save_path, "coco", "images", "train2014"))
        if not os.path.exists(os.path.join(args.save_path, "coco", "labels", "train2014")):
            os.makedirs(os.path.join(args.save_path, "coco", "labels", "train2014"))
        pilImages, cocoTargets = convert_to_coco(generatedImages, targets)
        # for pilim, cocotarget, imgpath in zip(pilImages, cocoTargets, imgspaths):
        #     imgname = os.path.basename(imgpath) # get filename
        #     imgname = os.path.splitext(imgname)[0] # remove .jpg/.png extension
        #     pilim.save(os.path.join(args.save_path, "coco", "images", "train2014", imgname+".png"))
        #     with open(os.path.join(args.save_path,"coco","labels","train2014",imgname+".txt"),"wt") as f:
        #         if len(cocotarget)>0:
        #             f.write(''.join(cocotarget).rstrip('\n'))
        for batch in range(generatedImages.shape[0]):
            imgname = os.path.basename(imgspaths[batch]) 
            imgname = os.path.splitext(imgname)[0]
            img_save = generatedImages[batch]
            vutils.save_image(img_save,
                            os.path.join(args.save_path, "coco", "images", "train2014", imgname+".jpg"),normalize=True, scale_each=True)
            with open(os.path.join(args.save_path,"coco","labels","train2014",imgname+".txt"),"wt") as f:
                if len(cocoTargets[batch])>0:
                    f.write(''.join(cocoTargets[batch]).rstrip('\n'))
    # Save the args
    with open(os.path.join(args.save_path, "args.txt"), "wt") as f:
        f.write(str(args)+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    # parser.add_argument('--seeds', type=str, default="0,0,23456", help="seeds for built_in random, numpy random and torch.manual_seed")
    parser.add_argument('--shuffle', action='store_true', help='use shuffle in dataloader')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--resolution', default=160, type=int, help="image optimization resolution")
    parser.add_argument('--iterations', default=2000, type=int, help='number of iterations for DI optim')
    parser.add_argument('--bs', default=40, type=int, help='batch size')
    parser.add_argument('--jitter', default=20, type=int, help='jitter')
    parser.add_argument('--mean_var_clip', action='store_true', help='clip the optimized image to the mean/var sampled from real data')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--save_every', type=int, default=1000, help='save an image every x iterations')
    parser.add_argument('--display_every', type=int, default=100, help='display the lsses every x iterations')
    parser.add_argument("--nms_conf_thres", type=float, default=0.01, help="NMS confidence 0.01 for speed, 0.001 for max mAP (default 0.01)")
    parser.add_argument("--nms_iou_thres", type=float, default=0.05, help="NMS iou threshold default: 0.5")
    parser.add_argument('--save_path', type=str, default='./results_generated/generated_sample/first_40_cats_each_class_1', help='where to store experimental data NOT: MUST BE A FOLDER')
    parser.add_argument('--sample_path', type=str, default='./results_generated/sampled_bbox/first_40_cats_each_class_1', help='smaple path')
    parser.add_argument('--fp16', action="store_true", help="Enabled Mixed Precision Training")
    parser.add_argument('--save-coco', default=True, action="store_true", help="save generated batch in coco format")

    parser.add_argument('--do_flip', default=True, action='store_true', help='DA:apply flip for model inversion')
    parser.add_argument("--rand_brightness", default=True, action="store_true", help="DA: randomly adjust brightness during optizn")
    parser.add_argument("--rand_contrast", default=False, action="store_true", help="DA: randomly adjust contrast during optizn")
    parser.add_argument("--random_erase", default=True, action="store_true", help="DA: randomly set rectangular regions to 0 during optizn")

    parser.add_argument('--r_feature', type=float, default=0.1, help='coefficient for feature distribution regularization')
    parser.add_argument('--p_norm', type=int, default=2, help='p for the Lp norm used to calculate r_feature')
    parser.add_argument('--alpha-mean', type=float, default=1.0, help='weight for mean norm in r_feature')
    parser.add_argument('--alpha-var', type=float, default=1.0, help='weight for var norm in r_feature')
    parser.add_argument('--alpha-ssim', type=float, default=0.0, help='weight for ssim')
    parser.add_argument('--cosine_layer_decay', default=False, action='store_true', help='use cosine decay for number of layers used to calculate r_feature')
    parser.add_argument('--min_layers', type=int, default=1, help='minimum number of layers used to calculate r_feature when using cosine decay')
    parser.add_argument('--num_layers', type=int, default=-1, help='number of layers used to calculate r_feature')
    parser.add_argument('--tv_l1', type=float, default=1.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--min_lr', type=float, default=0.0, help='minimum learning rate for scheduler')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay for optimization')
    parser.add_argument('--first_bn_coef', type=float, default=0.3, help='additional regularization for the first BN in the networks, coefficient, useful if colors do not match')
    parser.add_argument('--main_loss_multiplier', type=float, default=10.0, help=' coefficient for the main loss optimization')
    parser.add_argument('--alpha_img_stats', type=float, default=0.0, help='coefficient for loss_img_stats')
    parser.add_argument("--cache_batch_stats", action="store_true", help="use real image stats instead of bnorm mean/var")
    parser.add_argument("--real_mixin_alpha", type=float, default=0.0, help="how much of real image to mix in with the random initialization")
    parser.add_argument("--init_scale", type=float, default=1.0, help="for scaling the initialization, useful to start with a 'closer to black' kinda image") 
    parser.add_argument("--init_bias", type=float, default=0.0, help="for biasing the initialization")
    parser.add_argument('--init_chkpt', type=str, default="", help="chkpt containing initialization image (will up upsampled to args.resolution)")
    parser.add_argument("--beta1", type=float, default=0.0, help="beta1 for adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.0, help="beta1 for adam optimizer")

    parser.add_argument("--box-sampler", action="store_true", help="Enable False positive (Fp) sampling")
    parser.add_argument("--box-sampler-warmup", type=int, default=1000, help="warmup iterations before we start adding predictions to targets")
    parser.add_argument("--box-sampler-conf", type=float, default=0.5, help="confidence threshold for a prediction to become targets")
    parser.add_argument("--box-sampler-overlap-iou", type=float, default=0.2, help="a prediction must be below this overlap threshold with targets to become a target") # Increasing box overlap leads to more overlapped objects appearing
    parser.add_argument("--box-sampler-minarea", type=float, default=0.0, help="new targets must be larger than this minarea")
    parser.add_argument("--box-sampler-maxarea", type=float, default=1.0, help="new targets must be smaller than this maxarea")
    parser.add_argument("--box-sampler-earlyexit", type=int, default=1000000, help='early exit at this iteration')

    #model
    parser.add_argument('--config', default='',help='test config file path')
    parser.add_argument('--inversion_checkpoint', default='',help='checkpoint file')

    args = parser.parse_args()
    args.resolution = (args.resolution, args.resolution) # int -> (height,width)
    args.nms_params = { "iou_thres":args.nms_iou_thres, "conf_thres":args.nms_conf_thres }

    args.train_txt_path = args.sample_path + '/img_path.txt'
    args.order_class_path = args.sample_path + '/cat_order_coco.txt'


    print(args)
    torch.backends.cudnn.benchmark = True
    dataset = LoadImagesAndLabels(args.train_txt_path, args.resolution[0], args.bs,
                order_class_path=args.order_class_path, augment=False, hyp=hyp, rect=False, cache_images=False,
                cache_labels=False, single_cls=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=args.bs, num_workers=4,
                    shuffle=args.shuffle, pin_memory=False,
                    collate_fn=dataset.collate_fn)
    dataiter = iter(dataloader)
    num_dataset = len(dataset)
    epochs = np.ceil(num_dataset/args.bs)
    for epoch in range(int(epochs)):
        print(epoch)
        run(args, dataiter)



if __name__ == '__main__':
    main()

