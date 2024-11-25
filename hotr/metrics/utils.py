import torch
import numpy as np
import json


def get_iou(bb1, bb2):

    bb1[0], bb1[2] = min(bb1[0], bb1[2]), max(bb1[0], bb1[2])
    bb1[1], bb1[3] = min(bb1[1], bb1[3]), max(bb1[1], bb1[3])
    bb2[0], bb2[2] = min(bb2[0], bb2[2]), max(bb2[0], bb2[2])
    bb2[1], bb2[3] = min(bb2[1], bb2[3]), max(bb2[1], bb2[3])

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def eval_obj_bbox(obj_bbox, gt_obj_bbox):
    if not gt_obj_bbox.any() and not obj_bbox.any():
        return 1. # max
    else:
        return get_iou(obj_bbox, gt_obj_bbox)



def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

def get_AP_HO(
        hand_bboxes, hand_scores, obj_bboxes, obj_scores, confidence_scores,
        gt_hand_bboxes, gt_obj_bboxes,
        iou_thres=0.5, hand_score_thres=0., obj_score_thres=0.,
        use_07_metric=True):
    """
    Inputs:
            hand_bboxes: list of ndarray [(N, 4)] with len of num_images
            hand_scores: list of ndarray [(N,)] with len of num_images
            obj_bboxes: list of ndarray [(N, 4)] with len of num_images
            obj_scores: list of ndarray [(N,)] with len of num_images
            
            gt_hand_bboxes: list of ndarray [(M, 4)] with len of num_images
            gt_obj_bboxes: list of ndarray [(M, 4)] with len of num_images
    Output: 
            ap: average precision in float
    """
    # parse gt_bboxes to a dict
    npos = 0
    img_recs = {}
    for i, bboxs in enumerate(gt_hand_bboxes):
        npos += bboxs.shape[0]
        img_recs[i] = {'hand_bbox': bboxs, 'obj_bbox': gt_obj_bboxes[i]}

    # print('npos:', npos)
    # print('img_recs:', img_recs)

    # parse det_boxes, det_scores to an array
    hand_recs = []
    obj_recs = []
    hand_confidences = []
    obj_confidences = []
    final_confidences = []
    img_ids = []

    for i, hand_bbox in enumerate(hand_bboxes):
        hand_score = hand_scores[i]
        keep_idxs = hand_score > hand_score_thres
        hand_bbox, hand_score = hand_bbox[keep_idxs], hand_score[keep_idxs]
        obj_bbox, obj_score = obj_bboxes[i][keep_idxs], obj_scores[i][keep_idxs]
        confidence_score = confidence_scores[i][keep_idxs]

        if hand_bbox.size > 0:
            hand_recs.append(hand_bbox)
            obj_recs.append(obj_bbox)
            hand_confidences.append(hand_score)
            obj_confidences.append(obj_score)
            final_confidences.append(confidence_score)
            img_ids += [i] * hand_score.shape[0]

    # 引数のデータを準備
    args_dict = {
        "hand_recs": any_to_list(hand_recs)
    }
    # JSONファイルに保存
    with open('get_hand_recs.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    hand_recs = np.concatenate(hand_recs)
    obj_recs = np.concatenate(obj_recs)
    hand_confidences = np.concatenate(hand_confidences)
    obj_confidences = np.concatenate(obj_confidences)
    final_confidences = np.concatenate(final_confidences)
    
    # print('hand_recs:', hand_recs)
    # print('obj_recs:', obj_recs)
    # print('hand_confidences:', hand_confidences)
    # print('obj_confidences:', obj_confidences)
    # print('final_confidences:', final_confidences)
    # print('img_ids:', img_ids)

    # 引数のデータを準備
    args_dict = {
        "hand_recs": any_to_list(hand_recs)
    }
    # JSONファイルに保存
    with open('get_con_hand_recs.json', 'w') as f:
        json.dump(args_dict, f, indent=2)


    # define confidence to sort the bounding box
    confidence_to_sort = final_confidences
    # sort the det_recs by confidences
    sorted_ind = np.argsort(-confidence_to_sort)
    hand_recs = hand_recs[sorted_ind, :]
    obj_recs = obj_recs[sorted_ind, :]
    img_ids = [img_ids[x] for x in sorted_ind]

    # print('hand_recs:', hand_recs)
    # print('obj_recs:', obj_recs)
    # print('img_ids:', img_ids)

    # initialize tp/fp counting
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # for analysis
    tp_ho_counter = 0
    tp_h__counter = 0

    # evaluate each det_bbox
    for d in range(nd):
        img_id = img_ids[d]
        gt_hand_bbox = img_recs[img_id]['hand_bbox']  # (n, 4)
        gt_obj_bbox = img_recs[img_id]['obj_bbox']  # (n, 4)
        
        hand_rec = hand_recs[d]  # (4,)
        obj_rec = obj_recs[d]  # (4,)
        if gt_hand_bbox.size > 0:
            # compute ious
            ixmin = np.maximum(gt_hand_bbox[:, 0], hand_rec[0])
            iymin = np.maximum(gt_hand_bbox[:, 1], hand_rec[1])
            ixmax = np.minimum(gt_hand_bbox[:, 2], hand_rec[2])
            iymax = np.minimum(gt_hand_bbox[:, 3], hand_rec[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((hand_rec[2] - hand_rec[0] + 1.) * (hand_rec[3] - hand_rec[1] + 1.) +
                   (gt_hand_bbox[:, 2] - gt_hand_bbox[:, 0] + 1.) *
                   (gt_hand_bbox[:, 3] - gt_hand_bbox[:, 1] + 1.) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            # check hand iou and check object detection
            if ovmax > iou_thres and eval_obj_bbox(obj_rec, gt_obj_bbox[jmax]) > iou_thres:
                if gt_obj_bbox[jmax].any():
                    tp_ho_counter += 1
                else:
                    tp_h__counter += 1
                tp[d] = 1.
                img_recs[img_id]['hand_bbox'] = gt_hand_bbox[np.arange(
                    gt_hand_bbox.shape[0]) != jmax]
                img_recs[img_id]['obj_bbox'] = gt_obj_bbox[np.arange(
                    gt_obj_bbox.shape[0]) != jmax]
            else:
                fp[d] = 1
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)


    ap = voc_ap(rec, prec, use_07_metric)

    return prec, rec, ap




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_overlap(a, b):
    if type(a) == torch.Tensor:
        if len(a.shape) == 2:
            area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

            iw = torch.min(a[:, 2].unsqueeze(dim=1), b[:, 2]) - torch.max(a[:, 0].unsqueeze(dim=1), b[:, 0])
            ih = torch.min(a[:, 3].unsqueeze(dim=1), b[:, 3]) - torch.max(a[:, 1].unsqueeze(dim=1), b[:, 1])

            iw[iw<0] = 0
            ih[ih<0] = 0

            ua = torch.unsqueeze((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), dim=1) + area - iw * ih
            ua[ua < 1e-8] = 1e-8

            intersection = iw * ih

            return intersection / ua

    elif type(a) == np.ndarray:
        if len(a.shape) == 2:
            area = np.expand_dims((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), axis=0) #(1, K)

            iw = np.minimum(np.expand_dims(a[:, 2], axis=1), np.expand_dims(b[:, 2], axis=0)) \
                - np.maximum(np.expand_dims(a[:, 0], axis=1), np.expand_dims(b[:, 0], axis=0)) \
                + 1
            ih = np.minimum(np.expand_dims(a[:, 3], axis=1), np.expand_dims(b[:, 3], axis=0)) \
                - np.maximum(np.expand_dims(a[:, 1], axis=1), np.expand_dims(b[:, 1], axis=0)) \
                + 1

            iw[iw<0] = 0 # (N, K)
            ih[ih<0] = 0 # (N, K)

            intersection = iw * ih

            ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), axis=1) + area - intersection
            ua[ua < 1e-8] = 1e-8

            return intersection / ua

        elif len(a.shape) == 1:
            area = np.expand_dims((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), axis=0) #(1, K)

            iw = np.minimum(np.expand_dims([a[2]], axis=1), np.expand_dims(b[:, 2], axis=0)) \
                - np.maximum(np.expand_dims([a[0]], axis=1), np.expand_dims(b[:, 0], axis=0))
            ih = np.minimum(np.expand_dims([a[3]], axis=1), np.expand_dims(b[:, 3], axis=0)) \
                - np.maximum(np.expand_dims([a[1]], axis=1), np.expand_dims(b[:, 1], axis=0))

            iw[iw<0] = 0 # (N, K)
            ih[ih<0] = 0 # (N, K)

            ua = np.expand_dims([(a[2] - a[0] + 1) * (a[3] - a[1] + 1)], axis=1) + area - iw * ih
            ua[ua < 1e-8] = 1e-8

            intersection = iw * ih

            return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
    # Returns
            The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def any_to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif torch.is_tensor(value):  # PyTorchのテンソルの場合
        return any_to_list(value.cpu().detach().numpy())
    elif isinstance(value, list):
        return [any_to_list(item) for item in value]
    return value