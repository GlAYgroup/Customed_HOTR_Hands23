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

    # # 引数のデータを準備
    # args_dict = {
    #     "hand_recs": any_to_list(hand_recs)
    # }
    # # JSONファイルに保存
    # with open('get_hand_recs.json', 'w') as f:
    #     json.dump(args_dict, f, indent=2)

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

    # # 引数のデータを準備
    # args_dict = {
    #     "hand_recs": any_to_list(hand_recs)
    # }
    # # JSONファイルに保存
    # with open('get_con_hand_recs.json', 'w') as f:
    #     json.dump(args_dict, f, indent=2)


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

def get_AP_HOS(
        hand_bboxes, hand_scores,
        obj_bboxes, obj_scores,
        sobj_bboxes, sobj_scores,
        confidence_scores,
        gt_hand_bboxes, gt_obj_bboxes, gt_sobj_bboxes,
        iou_thres=0.5, hand_score_thres=0., obj_score_thres=0.,
        use_07_metric=True):
    """
    Inputs:
        - hand_bboxes: list of ndarray [(N, 4)] with len of num_images
        - hand_scores: list of ndarray [(N,)] with len of num_images
        - obj_bboxes: list of ndarray [(N, 4)] with len of num_images
        - obj_scores: list of ndarray [(N,)] with len of num_images
        - sobj_bboxes: list of ndarray [(N, 4)] with len of num_images
        - sobj_scores: list of ndarray [(N,)] with len of num_images
        - confidence_scores: list of ndarray [(N,)] with len of num_images

        - gt_hand_bboxes: list of ndarray [(M, 4)] with len of num_images
        - gt_obj_bboxes: list of ndarray [(M, 4)] with len of num_images
        - gt_sobj_bboxes: list of ndarray [(M, 4)] with len of num_images

    Output:
        - ap: average precision in float
    """
    # グラウンドトゥルースの準備
    npos = 0
    img_recs = {}
    for i, bboxs in enumerate(gt_hand_bboxes):
        npos += bboxs.shape[0]
        img_recs[i] = {
            'hand_bbox': bboxs,
            'obj_bbox': gt_obj_bboxes[i],
            'sobj_bbox': gt_sobj_bboxes[i],
        }

    # 予測結果の準備
    hand_recs = []
    obj_recs = []
    sobj_recs = []
    hand_confidences = []
    obj_confidences = []
    sobj_confidences = []
    final_confidences = []
    img_ids = []

    for i, hand_bbox in enumerate(hand_bboxes):
        hand_score = hand_scores[i]
        keep_idxs = hand_score > hand_score_thres
        hand_bbox, hand_score = hand_bbox[keep_idxs], hand_score[keep_idxs]
        obj_bbox, obj_score = obj_bboxes[i][keep_idxs], obj_scores[i][keep_idxs]
        sobj_bbox, sobj_score = sobj_bboxes[i][keep_idxs], sobj_scores[i][keep_idxs]
        confidence_score = confidence_scores[i][keep_idxs]

        if hand_bbox.size > 0:
            hand_recs.append(hand_bbox)
            obj_recs.append(obj_bbox)
            sobj_recs.append(sobj_bbox)
            hand_confidences.append(hand_score)
            obj_confidences.append(obj_score)
            sobj_confidences.append(sobj_score)
            final_confidences.append(confidence_score)
            img_ids += [i] * hand_score.shape[0]

    hand_recs = np.concatenate(hand_recs)
    obj_recs = np.concatenate(obj_recs)
    sobj_recs = np.concatenate(sobj_recs)
    hand_confidences = np.concatenate(hand_confidences)
    obj_confidences = np.concatenate(obj_confidences)
    sobj_confidences = np.concatenate(sobj_confidences)
    final_confidences = np.concatenate(final_confidences)

    # 信頼度スコアに基づいてソート
    confidence_to_sort = final_confidences
    sorted_ind = np.argsort(-confidence_to_sort)
    hand_recs = hand_recs[sorted_ind, :]
    obj_recs = obj_recs[sorted_ind, :]
    sobj_recs = sobj_recs[sorted_ind, :]
    img_ids = [img_ids[x] for x in sorted_ind]

    # 真陽性と偽陽性の初期化
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # 各予測結果の評価
    for d in range(nd):
        img_id = img_ids[d]
        gt_hand_bbox = img_recs[img_id]['hand_bbox']
        gt_obj_bbox = img_recs[img_id]['obj_bbox']
        gt_sobj_bbox = img_recs[img_id]['sobj_bbox']

        hand_rec = hand_recs[d]
        obj_rec = obj_recs[d]
        sobj_rec = sobj_recs[d]

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
            ovmax_hand = np.max(overlaps)
            jmax = np.argmax(overlaps)

            # Active-objectのIoU計算
            overlaps_obj = eval_obj_bbox(obj_rec, gt_obj_bbox[jmax])

            # Second-objectのIoU計算
            overlaps_sobj = eval_obj_bbox(sobj_rec, gt_sobj_bbox[jmax])

            # 真陽性の条件判定
            if (ovmax_hand > iou_thres) and (overlaps_obj > iou_thres) and (overlaps_sobj > iou_thres):
                tp[d] = 1.
                img_recs[img_id]['hand_bbox'] = gt_hand_bbox[np.arange(
                    gt_hand_bbox.shape[0]) != jmax]
                img_recs[img_id]['obj_bbox'] = gt_obj_bbox[np.arange(
                    gt_obj_bbox.shape[0]) != jmax]
                img_recs[img_id]['sobj_bbox'] = gt_sobj_bbox[np.arange(
                    gt_sobj_bbox.shape[0]) != jmax]
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # 適合率と再現率の計算
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # APの計算
    ap = voc_ap(rec, prec, use_07_metric)

    return prec, rec, ap

import numpy as np

# def get_AP_single(
#         pred_bboxes, pred_scores,
#         gt_bboxes,
#         iou_thres=0.5, score_threshold=0.0,
#         use_07_metric=True):
#     """
#     単一カテゴリの Average Precision (AP) を計算します。
    
#     Args:
#         pred_bboxes (list of np.ndarray): 各画像の予測ボックスのリスト。各要素は (N, 4) の ndarray。
#             形式: [ [ [x_min, y_min, x_max, y_max], ... ], ... ]
#         pred_scores (list of np.ndarray): 各画像の予測スコアのリスト。各要素は (N,) の ndarray。
#             形式: [ [score1, score2, ...], ... ]
#         gt_bboxes (list of np.ndarray): 各画像のグラウンドトゥルースボックスのリスト。各要素は (M, 4) の ndarray。
#             形式: [ [ [x_min, y_min, x_max, y_max], ... ], ... ]
#         iou_threshold (float, optional): 真陽性とみなす IoU の閾値。デフォルトは 0.5。
#         score_threshold (float, optional): 予測スコアの閾値。これ以下の予測は無視されます。デフォルトは 0.0。
#         use_07_metric (bool, optional): VOC 2007 の 11 点 AP を使用するかどうか。デフォルトは True。
    
#     Returns:
#         float: 計算された Average Precision (AP) 値。
#     """
#     # グラウンドトゥルースの準備
#     npos = 0
#     img_recs = {}
#     for i, bboxs in enumerate(gt_bboxes):
#         npos += bboxs.shape[0]
#         img_recs[i] = {
#             'bbox': bboxs
#         }
    
#     # 予測の収集とスコアによるフィルタリング
#     bbox_recs = []
#     bbox_confidences = []
#     img_ids = []

#     for i, bboxes in enumerate(pred_bboxes):
#         scores = pred_scores[i]
#         keep_idxs = scores > score_threshold
#         bboxes, scores = bboxes[keep_idxs], scores[keep_idxs]
#         confidence_score = scores

#         if bboxes.size > 0:
#             bbox_recs.append(bboxes)
#             bbox_confidences.append(scores)
#             img_ids += [i] * scores.shape[0]
    
#     bbox_recs = np.concatenate(bbox_recs)
#     bbox_confidences = np.concatenate(bbox_confidences)

#     # 信頼度スコアに基づいてソート
#     confidence_to_sort = bbox_confidences
#     sorted_ind = np.argsort(-confidence_to_sort)
#     bbox_recs = bbox_recs[sorted_ind, :]
#     img_ids = [img_ids[x] for x in sorted_ind]

#     # 真陽性と偽陽性の初期化
#     nd = len(img_ids)
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)

#     # 各予測結果の評価
#     for d in range(nd):
#         img_id = img_ids[d]
#         gt_bbox = img_recs[img_id]['bbox']

#         bbox_rec = bbox_recs[d]

#         if gt_bbox.size > 0:
#             # compute ious
#             ixmin = np.maximum(gt_bbox[:, 0], bbox_rec[0])
#             iymin = np.maximum(gt_bbox[:, 1], bbox_rec[1])
#             ixmax = np.minimum(gt_bbox[:, 2], bbox_rec[2])
#             iymax = np.minimum(gt_bbox[:, 3], bbox_rec[3])
#             iw = np.maximum(ixmax - ixmin + 1., 0.)
#             ih = np.maximum(iymax - iymin + 1., 0.)
#             inters = iw * ih
#             uni = ((bbox_rec[2] - bbox_rec[0] + 1.) * (bbox_rec[3] - bbox_rec[1] + 1.) +
#                    (gt_bbox[:, 2] - gt_bbox[:, 0] + 1.) *
#                    (gt_bbox[:, 3] - gt_bbox[:, 1] + 1.) - inters)
#             overlaps = inters / uni
#             ovmax = np.max(overlaps)
#             jmax = np.argmax(overlaps)

#             # 真陽性の条件判定
#             if ovmax > iou_thres:
#                 tp[d] = 1.
#                 img_recs[img_id]['bbox'] = gt_bbox[np.arange(
#                     gt_bbox.shape[0]) != jmax]
#             else:
#                 fp[d] = 1.
#         else:
#             fp[d] = 1.
    
#     # 適合率と再現率の計算
#     fp = np.cumsum(fp)
#     tp = np.cumsum(tp)
#     rec = tp / float(npos)
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

#     # AP の計算
#     ap = voc_ap(rec, prec, use_07_metric)

#     return prec, rec ,ap
    

import numpy as np

def remove_zero_boxes_and_scores(bboxes, scores):
    """
    全ての値が0のボックスと対応するスコアを削除します。

    Args:
        bboxes (np.ndarray): ボックスの配列 (N, 4)。
        scores (np.ndarray): スコアの配列 (N,)。

    Returns:
        tuple: (フィルタリング後のボックス, フィルタリング後のスコア)
    """
    # 空の配列の場合を処理
    if bboxes is None or len(bboxes) == 0:
        return np.empty((0, 4)), np.empty((0,))
    valid_idxs = np.any(bboxes != 0, axis=1)  # 全てが0でない行のインデックス
    return bboxes[valid_idxs], scores[valid_idxs]

def get_AP_single(
        pred_bboxes, pred_scores,
        gt_bboxes,
        iou_thres=0.5, score_threshold=0.0,
        use_07_metric=True):
    """
    単一カテゴリの Average Precision (AP) を計算します。

    Args:
        pred_bboxes (list of np.ndarray): 各画像の予測ボックスのリスト。
        pred_scores (list of np.ndarray): 各画像の予測スコアのリスト。
        gt_bboxes (list of np.ndarray): 各画像のグラウンドトゥルースボックスのリスト。
        iou_threshold (float, optional): IoUの閾値。デフォルトは0.5。
        score_threshold (float, optional): スコアの閾値。これ以下の予測は無視されます。デフォルトは0.0。
        use_07_metric (bool, optional): VOC 2007の11点法を使用するか。デフォルトはTrue。

    Returns:
        tuple: (precision, recall, AP) のタプル。
    """
    # グラウンドトゥルースの準備
    npos = 0
    img_recs = {}
    for i, bboxs in enumerate(gt_bboxes):
        bboxs, _ = remove_zero_boxes_and_scores(bboxs, np.zeros(bboxs.shape[0]))  # ボックスのみフィルタ
        npos += bboxs.shape[0]
        img_recs[i] = {'bbox': bboxs}
    
    # 予測の収集とスコアによるフィルタリング
    bbox_recs = []
    bbox_confidences = []
    img_ids = []

    for i, bboxes in enumerate(pred_bboxes):
        scores = pred_scores[i]

        # 無効なボックスとスコアを削除
        bboxes, scores = remove_zero_boxes_and_scores(bboxes, scores)

        # ボックスが空の場合をスキップ
        if bboxes.size == 0:
            continue

        keep_idxs = scores > score_threshold

        # スコアが条件を満たすボックスがない場合をスキップ
        if np.sum(keep_idxs) == 0:
            continue

        bboxes, scores = bboxes[keep_idxs], scores[keep_idxs]

        bbox_recs.append(bboxes)
        bbox_confidences.append(scores)
        img_ids += [i] * scores.shape[0]

    bbox_recs = np.concatenate(bbox_recs) if bbox_recs else np.array([])
    bbox_confidences = np.concatenate(bbox_confidences) if bbox_confidences else np.array([])

    if len(bbox_recs) == 0:
        # 有効な予測がない場合
        return 0.0, 0.0, 0.0

    # スコアに基づいてソート
    sorted_ind = np.argsort(-bbox_confidences)
    bbox_recs = bbox_recs[sorted_ind, :]
    img_ids = [img_ids[x] for x in sorted_ind]

    # 真陽性 (TP) と偽陽性 (FP) の初期化
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # 各予測の評価
    for d in range(nd):
        img_id = img_ids[d]
        gt_bbox = img_recs[img_id]['bbox']

        bbox_rec = bbox_recs[d]

        if gt_bbox.size > 0:
            # IoU の計算
            ixmin = np.maximum(gt_bbox[:, 0], bbox_rec[0])
            iymin = np.maximum(gt_bbox[:, 1], bbox_rec[1])
            ixmax = np.minimum(gt_bbox[:, 2], bbox_rec[2])
            iymax = np.minimum(gt_bbox[:, 3], bbox_rec[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((bbox_rec[2] - bbox_rec[0] + 1.) * (bbox_rec[3] - bbox_rec[1] + 1.) +
                   (gt_bbox[:, 2] - gt_bbox[:, 0] + 1.) *
                   (gt_bbox[:, 3] - gt_bbox[:, 1] + 1.) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            # マッチング条件を判定
            if ovmax > iou_thres:
                tp[d] = 1.
                img_recs[img_id]['bbox'] = gt_bbox[np.arange(gt_bbox.shape[0]) != jmax]
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # 適合率 (Precision) と再現率 (Recall) の計算
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # 平均適合率 (AP) の計算
    ap = _compute_ap(rec, prec)

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