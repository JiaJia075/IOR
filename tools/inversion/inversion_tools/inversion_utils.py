import torch
import pickle
import numpy as np
import cv2
import random
import os
import torchvision
from PIL import Image

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def flip_targets(targets, horizontal=True, vertical=False):
    """horizontal and vertical flipping for `targets`."""
    assert targets.shape[1] == 6
    targets_flipped = targets.clone().detach()
    if horizontal:
        targets_flipped[:,2] = 0.5 - (targets_flipped[:,2] - 0.5)
    if vertical:
        targets_flipped[:,3] = 0.5 - (targets_flipped[:,3] - 0.5)
    return targets_flipped

def jitter_targets(targets, xshift=0, yshift=0, img_shape=(320,320)):
    """
    Apply horizontal & vertical jittering to the targets for given img_shape
    note: img_shape is in real world parameters, but targets are still between 0-1
    img_shape = (height, width)
    targets shape = [batch_idx, cls, center x, center y, w, h]
    """
    assert targets.shape[1] == 6
    targets_jittered = targets.clone().detach().cpu()
    height, width = img_shape
    xywh = targets_jittered[:,2:]
    whwh = torch.tensor([width, height, width, height], dtype=torch.float32)
    xyxy = xywh2xyxy(xywh) * whwh

    # adjust the tbox
    xyxy[:,0] += xshift
    xyxy[:,2] += xshift
    xyxy[:,1] += yshift
    xyxy[:,3] += yshift

    # Limit co-ords
    xyxy[:,0] = torch.clamp(xyxy[:,0], min=0, max=width)
    xyxy[:,2] = torch.clamp(xyxy[:,2], min=0, max=width)
    xyxy[:,1] = torch.clamp(xyxy[:,1], min=0, max=height)
    xyxy[:,3] = torch.clamp(xyxy[:,3], min=0, max=height)

    # xyxy --> xywh
    xywh = xyxy2xywh(xyxy / whwh)
    targets_jittered[:,2:] = xywh

    # remove boxes that have 0 area
    oof = (targets_jittered[:,-1] * targets_jittered[:,-2] * width * height) < 1
    # print("Jittering Dropping {} boxes".format(oof.sum()))
    # targets_jittered = targets_jittered[~oof]
    targets_jittered[oof,:] = targets.clone().detach().cpu()[oof,:]

    return targets_jittered.to(targets.device)

def random_erase_masks(inputs_shape, return_cuda=True):
    """
    return a 1/0 mask with random rectangles marked as 0.
    shape should match inputs_shape
    """
    bs = inputs_shape[0]
    height = inputs_shape[2]
    width  = inputs_shape[3]
    masks = []
    _rand_erase = torchvision.transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        value=0
    )
    for idx in range(bs):
        mask = torch.ones(3,height,width,dtype=torch.float32)
        mask = _rand_erase(mask)
        masks.append(mask)
    masks = torch.stack(masks)
    if return_cuda:
        masks = masks.cuda()
    return masks

def xyxy2xywh(x):
    # Transform box coordinates from [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right) to [x, y, w, h] 
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def draw_targets(imgs, targets, order_class_path):
    """Draw `targets` bboxes on `imgs`."""
    batch_size = len(imgs)
    # Get colors + names of classes
    # with open("/data1/home/anzijia/DIODE-yolo/models/yolo/names.pkl", "rb") as f:
    #     names = pickle.load(f)
    names = []
    with open(order_class_path, "r") as f:
        for line in f:
            cat_name = line.split(',')[0]
            names.append(cat_name)
            
    with open("/data1/home/anzijia/DIODE-yolo/models/yolo/colors.pkl", "rb") as f:
        colors = pickle.load(f)

    # Draw boxes
    imgs_with_boxes = []
    for idx in range(batch_size):
        img_np = imgs[idx].clone().detach().cpu().numpy()
        img_np = np.transpose(img_np, axes=(1,2,0))
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
        height,width,_ = img_np.shape

        targets_batch = targets[targets[:,0]==idx]
        if len(targets_batch):
            for box in targets_batch:
                cls = int(box[1].item())
                xywh = box[2:].view(1,-1)
                xyxy = xywh2xyxy(xywh)
                xyxy[:,0] *= width
                xyxy[:,1] *= height
                xyxy[:,2] *= width
                xyxy[:,3] *= height
                label="{}".format(names[cls])
                plot_one_box(xyxy[0], img_np, label=label, color=colors[cls])
        imgs_with_boxes.append(np.transpose(img_np, axes=(2,0,1)))

    imgs_with_boxes = np.array(imgs_with_boxes).astype(np.float32) / 255.0
    imgs_with_boxes = torch.from_numpy(imgs_with_boxes)

    torch.cuda.empty_cache()
    return imgs_with_boxes

def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, conf, class)
    """
    # NMS methods https://github.com/ultralytics/yolov3/issues/679 'or', 'and', 'merge', 'vision', 'vision_batch'

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = 'vision_batch'
    batched = 'batch' in method  # run once per image, all classes simultaneously
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Apply conf constraint
        pred = pred[pred[:, 4] > conf_thres]

        # Apply width-height constraint
        pred = pred[((pred[:, 2:4] > min_wh) & (pred[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Compute conf
        pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (pred[:, 5:] > conf_thres).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1)
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            pred = pred[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(pred).all():
            pred = pred[torch.isfinite(pred).all(1)]

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Sort by confidence
        if not method.startswith('vision'):
            pred = pred[pred[:, 4].argsort(descending=True)]

        # Batched NMS
        if batched:
            c = pred[:, 5] * 0 if agnostic else pred[:, 5]  # class-agnostic NMS
            boxes, scores = pred[:, :4].clone(), pred[:, 4]
            boxes += c.view(-1, 1) * max_wh
            if method == 'vision_batch':
                i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            elif method == 'fast_batch':  # FastNMS from https://github.com/dbolya/yolact
                iou = box_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
                i = iou.max(dim=0)[0] < iou_thres

            output[image_i] = pred[i]
            continue

        # All other NMS methods
        det_max = []
        cls = pred[:, -1]
        for c in cls.unique():
            dc = pred[cls == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:
                dc = dc[:500]  # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117

            if method == 'vision':
                det_max.append(dc[torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], iou_thres)])

            elif method == 'or':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > iou_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'and':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'merge':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > iou_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif method == 'soft':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    dc = dc[dc[:, 4] > conf_thres]  # https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]].clamp_(0, img_shape[1])  # clip x
    boxes[:, [1, 3]].clamp_(0, img_shape[0])  # clip y

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [len(unique_classes), tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def inference(net, imgs, targets, nms_params={"iou_thres":0.5, "conf_thres":0.01}):
    """Calculate iou metrics on network using `imgs` and corresponding `targets`."""
    imgs, targets = imgs.clone().detach().cuda(), targets.clone().detach().cuda()

    # Enable inference
    net.eval()

    # Forward + nms
    with torch.no_grad():
        output = net(imgs, mode='predict')
        # preds = net(imgs)[0] # (batchsize, bboxes, 85)
        # Apply NMS
        # Confidence threshold: 0.01 for speed, 0.001 for best mAP
        # output = non_max_suppression(preds, nms_params["conf_thres"], nms_params["iou_thres"], classes=None, agnostic=False)

    # Get colors + names of classes
    with open("./models/yolo/names.pkl", "rb") as f:
        names = pickle.load(f)
    with open("./models/yolo/colors.pkl", "rb") as f:
        colors = pickle.load(f)

    # Plot bounding boxes on each image
    imgs_with_boxes = []
    for idx, det in enumerate(output):

        img_np = imgs[idx].clone().detach().cpu().numpy()
        img_np = np.transpose(img_np, axes=(1,2,0))
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np, dtype=np.uint8)

        # Plot boundingboxes for this image
        if det is not None:
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img_np, label=label, color=colors[int(cls)])
        else:
            # print("[INFERENCE] NoneType found in Prediction skipping drawing boxes on this image idx {}".format(idx))
            pass

        imgs_with_boxes.append(np.transpose(img_np, axes=(2,0,1)))
    imgs_with_boxes = np.array(imgs_with_boxes).astype(np.float32) / 255.0
    imgs_with_boxes = torch.from_numpy(imgs_with_boxes)

    # hyperparameters
    batch_size, _, height, width = imgs.shape
    iouv = torch.tensor([0.5], dtype=torch.float32).cuda()
    niou = iouv.numel()
    whwh = torch.tensor([width, height, width, height], dtype=torch.float32).cuda()
    stats = []

    # Compute metrics per image
    for img_idx, pred in enumerate(output):
        labels = targets[targets[:, 0] == img_idx , 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist()  # target class

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Clip boxes to image bounds
        clip_coords(pred, (height, width))

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()

        detected = []  # target indices
        tcls_tensor = labels[:, 0]

        # target boxes
        tbox = xywh2xyxy(labels[:, 1:5]) * whwh

        # Per target class
        for cls in torch.unique(tcls_tensor):
            ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
            pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

            # Search for detections
            if pi.shape[0]:
                # Prediction to target ious
                ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                # Append detections
                for j in (ious > iouv[0]).nonzero():
                    d = ti[i[j]]  # detected target
                    if d not in detected:
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # all targets already located in image
                            break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    p, r, ap, f1, ap_class = ap_per_class(*stats)
    mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=net.nc)  # number of targets per class

    # save memory
    del preds
    torch.cuda.empty_cache()
    return float(mp), float(mr), float(map), float(mf1), imgs_with_boxes, output

def convert_to_coco(inputs_tensor, targets=None):
    """
    Convert an inputs_tensor (bs x 3 x height x width) to #batch-size images in PIL
    format.
    Convert targets loaded by a dataloader to plaintxt annotations in coco format
    """
    images = inputs_tensor.clone().detach().cpu()
    images = torch.clamp(images, min=0.0, max=1.0) * 255.0
    images = images.to(torch.uint8)
    images = images.numpy()
    images = np.transpose(images, axes=(0,2,3,1)) # (32, h, w, 3)
    pil_images = []
    for batch_idx in range(len(images)):
        pil_images.append(Image.fromarray(images[batch_idx]))

    coco_targets = [None] * len(images)
    if targets is not None:
        targets = targets.clone().detach().cpu()
        coco_targets = []
        for batch_idx in range(len(images)):
            imboxes = targets[targets[:,0]==batch_idx]
            boxlist = []
            for box in imboxes:
                _box_str = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(box[1].item()), box[2].item(), box[3].item(), box[4].item(), box[5].item())
                boxlist.append(_box_str)
            coco_targets.append(boxlist)

    assert len(coco_targets) == len(pil_images)

    return pil_images, coco_targets