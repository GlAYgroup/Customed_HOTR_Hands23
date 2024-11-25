# ------------------------------------------------------------------------
# HOTR official code : hotr/engine/evaluator_vcoco.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import torch
import time
import datetime

import hotr.util.misc as utils
import hotr.util.logger as loggers
from hotr.data.evaluators.doh_eval import DohEvaluator
import hotr.metrics.utils as metrics_utils

from hotr.util.box_ops import rescale_bboxes, rescale_pairs, rescale_triplet

import wandb

from torch.utils.data import Subset

import cv2
import numpy as np
import shutil
import json


#test結果の集約
@torch.no_grad()
def doh_evaluate(model, criterion, postprocessors, data_loader, device, output_dir, thr, args):
    model.eval()
    criterion.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (DOH)'

    print_freq = 1 # len(data_loader)
    res = {}
    hoi_recognition_time = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = utils.reduce_dict(loss_dict) # ddp gathering

        # print("outputs", outputs)
        #デバッグ用
        outputs_list = tensor_to_list(outputs)
        with open('maeda/outputs/hands23/check_test_outputs.json', 'w') as f:
            json.dump(outputs_list, f, indent=4)

        #outpusは正しくsecond objectを出力しているためOk
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        if args.dataset_file == 'doh':
            results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='doh')
        elif args.dataset_file == 'hands23':
            results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='hands23')
            
        #デバッグ用
        results_list = tensor_to_list(results)
        with open('maeda/outputs/hands23/check_results_list.json', 'w') as f:
            json.dump(results_list, f, indent=4)

        targets = process_target(targets, orig_target_sizes, args)
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        res.update(
            {target['image_id'].item():\
                {'target': target, 'prediction': output} for target, output in zip(targets, results)
            }
        )

        # outputs_list = tensor_to_list(outputs)
        # with open('maeda/outputs/check_doh_output_results.json', 'w') as f:
        #     json.dump(outputs_list, f, indent=4)

    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")

    start_time = time.time()
    gather_res = utils.all_gather(res)
    total_res = {}
    for dist_res in gather_res:
        total_res.update(dist_res)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"[stats] Distributed Gathering Time : {total_time_str}")

    # total_resはTensorを含む可能性がある辞書
    total_res_list = tensor_to_list(total_res)
    with open('maeda/outputs/hands23/check_eval_results.json', 'w') as f:
        json.dump(total_res_list, f, indent=4)

    return total_res

# Visualize results of the test datasets
def doh_visualizer(args, total_res, dataset):
    output_dir = args.output_dir
    root = args.root
    hand_thr = args.hand_threshold
    obj_thr = args.object_threshold
    sobj_thr = args.second_object_threshold

    if isinstance(dataset, Subset):
        # subset_dataset が Subset クラスのインスタンスである場合の処理
        dataset = dataset.dataset
        print("this dataset is subset")

    # output_dir の存在を確認し、存在しない場合はエラーを発生させる
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")
    
    # クラスに対応する色の辞書
    label_colors = { #Memo (Blue, Green, Red)
        'Hand': (255, 0, 0),  # Handの色: 青
        'Active_Obj': (0, 255, 255),  # Active Objectの色: 黄色
        'Second_Obj': (0, 255, 0),  # Second Objectの色: 緑
    }
    # bboxの太さ
    thickness = 6

    # ロガーの初期化
    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    print_freq = 1


    # GTの可視化
    header = 'Visualize GT (DOH)'

    GT_output_dir = os.path.join(output_dir, "GT_visualization")
    # 既存のフォルダがある場合、削除する
    if os.path.exists(GT_output_dir):
        shutil.rmtree(GT_output_dir)
    os.makedirs(GT_output_dir)
    
    for key, value in metric_logger.log_every(total_res.items(), print_freq, header):

        # tqdm.set_description("Visalization GT Processing")
        
        img_info = dataset.get_image_info(key)
        
        target = value["target"]
        target_pair = target['pair_boxes']
        image = cv2.imread(os.path.join(root, img_info))

        # print(target)

        for pair_bbox in target['pair_boxes']:
            if args.task == 'AOD':
                image = draw_GT_bbox_pair(image, pair_bbox, label_colors, thickness)
            elif args.task == 'ASOD':
                image = draw_GT_bbox_triplet(image, pair_bbox, label_colors, thickness)
        
        # 保存先のパスを生成
        basename = os.path.basename(img_info)
        save_path = os.path.join(GT_output_dir, 'GT_' + basename)
        # 画像を保存
        cv2.imwrite(save_path, image)
    

    # 予測の可視化
    header = 'Visualize Pred (DOH)'
    pred_output_dir = os.path.join(output_dir, args.vis_mode)
    pred_output_dir = pred_output_dir + "_pred_visualization"
    print("pred_output_dir: ", pred_output_dir)

    # 既存のフォルダがある場合、削除する
    if os.path.exists(pred_output_dir):
        shutil.rmtree(pred_output_dir)
    os.makedirs(pred_output_dir)

    for key, value in metric_logger.log_every(total_res.items(), print_freq, header):
        # tqdm.set_description("Visalization Pred Processing")

        img_info = dataset.get_image_info(key)
        
        target = value["prediction"]
        image = cv2.imread(os.path.join(root, img_info))
        # print("target", target)

        # 予測されたhoi_num_queryを全て可視化
        if args.vis_mode == "all":
            if args.task == 'AOD':
                # hoi_num_queryを一組ずつ可視化する
                for i in range(args.num_hoi_queries):
                    h_cat_score = target['h_cat_score'][i]
                    o_cat_score = target['o_cat_score'][i]
                    if h_cat_score >= hand_thr and o_cat_score >= obj_thr:
                        image = draw_bbox_pair(image, target, label_colors, i, thickness)
            if args.task == 'ASOD':
                # hoi_num_queryを一組ずつ可視化する
                for i in range(args.num_hoi_queries):
                    h_cat_score = target['h_cat_score'][i]
                    o_cat_score = target['o_cat_score'][i]
                    so_cat_score = target['so_cat_score'][i]
                    if h_cat_score >= hand_thr and o_cat_score >= obj_thr and so_cat_score >= sobj_thr:
                        image = draw_bbox_triplet(image, target, label_colors, i, thickness)

        # 予測されたhoi_num_queryの中で、任意のHandに対する情報を集約する。Active Objectのラベルが物体のものの中でcat_scoreが最も高いもの、かつ、それが閾値を超えていれば描画

        # まず予測されたhoi_num_queryから、あるHandに対応するActive Object情報を集約する.
        # 例：画像内の右手が予測された時にそれに関わるHoi_num_qeuryが3つあったとする. 
        # それぞれのhoi_num_queryのactive objectはそれぞれHand（つまり自己回帰）, Object1, Object2とする.
        # uniqueでは予測されたActive Objectがある際に無条件でHandは除外、つまり自己回帰は除外される
        # 次にObject1, Object2のうちcat_scoreが高いものを選択し、もしそれが閾値を超えていればそれに関わるHandとObjectを描画する
        # 閾値を超えていなかったり、そもそもObject1, Object2などラベルがobjectのものが存在しない場合はHandのみを描画する    
        elif args.vis_mode == "unique_obj":
            
            # ユニークな数ごとにインデックスの集合を格納するディクショナリを初期化
            h_index_dict = {}

            # テンソルをイテレートしてインデックスを集合に追加
            for i, num in enumerate(target['h_index'].tolist()):
                if num in h_index_dict:
                    h_index_dict[num].add(i)
                else:
                    h_index_dict[num] = {i}
            
            for unique_key, index_list in h_index_dict.items():
                if args.task == 'AOD':
                    # ある任意のHandに対するhoi_queryをひとつ選ぶ関数. ラベルがobjectの最大スコアのactive objectのインデックスとそのスコアを返す関数
                    max_index, max_obj_value = find_max_active_obj_value_index(index_list, target, obj_thr)
      
                    h_cat_score = target['h_cat_score'][max_index]
                    o_cat_score = target['o_cat_score'][max_index]
                    #描画方法の条件分岐
                    if h_cat_score >= hand_thr :
                        if o_cat_score >= obj_thr and target['o_cat_label'][max_index] == 2:
                            image = draw_bbox_pair(image, target, label_colors, max_index, thickness)
                        else:
                            image = draw_hand_bbox(image, target, label_colors, max_index, thickness)
                elif args.task == 'ASOD':
                    # ある任意のHandに対するhoi_queryをひとつ選ぶ関数. 最大スコアのオブジェクトのインデックスとそのスコアを返す関数
                    max_index = find_max_second_obj_value_index(index_list, target, obj_thr, sobj_thr)

                    h_cat_score = target['h_cat_score'][max_index]
                    o_cat_score = target['o_cat_score'][max_index]
                    so_cat_score = target['so_cat_score'][max_index]
                    if args.check:
                        print("h_index_dict", h_index_dict)
                        print("index_list", index_list)
                        print("max_index", max_index)
                        print("target['h_cat_score']", target['h_cat_score'])
                        print("target['h_cat_label']", target['h_cat_label'])
                        print("target['o_cat_score']", target['o_cat_score'])
                        print("target['o_cat_label']", target['o_cat_label'])
                        print("target['so_cat_score']", target['so_cat_score'])
                        print("target['so_cat_label']", target['so_cat_label'])
                      
                    #描画方法の条件分岐
                    if h_cat_score >= hand_thr :
                        if (so_cat_score >= sobj_thr and target['so_cat_label'][max_index] == 2) and (o_cat_score >= obj_thr and target['o_cat_label'][max_index] == 2):
                            image = draw_bbox_triplet(image, target, label_colors, max_index, thickness)
                        elif o_cat_score >= obj_thr and target['o_cat_label'][max_index] == 2:
                            image = draw_bbox_pair(image, target, label_colors, max_index, thickness)
                        else:
                            image = draw_hand_bbox(image, target, label_colors, max_index, thickness)        
        # 保存先のパスを生成
        basename = os.path.basename(img_info)
        save_path = os.path.join(pred_output_dir, 'Pred_' + basename)
        # 画像を保存
        cv2.imwrite(save_path, image)
        


#集約されたテスト結果から評価を行う
def doh_accumulate(total_res, args, print_results, wandb_log):
    output_dir = args.output_dir
    hand_thr = args.hand_threshold
    obj_thr = args.object_threshold


    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    print_freq = 1

    # GTとPredのValidationのためのリスト作成
    header = 'Validation (DOH)'
    val_output_dir = os.path.join(output_dir, 'validation')
    print("val_output_dir", val_output_dir)

    # 既存のフォルダがある場合、削除する
    if os.path.exists(val_output_dir):
        shutil.rmtree(val_output_dir)
    os.makedirs(val_output_dir)

    
    results = []
    doh100_hand_boxes = []
    doh100_hand_scores = []
    pred_obj_boxes = []
    pred_obj_scores = []
    pred_confidence_scores = []
    gt_hand_boxes = []
    gt_obj_boxes = []

    # 1imageずつ処理を行う
    for key, value in metric_logger.log_every(total_res.items(), print_freq, header):
        target = value["prediction"]
        gt = value["target"]

        # 手に対応する物体の中で、最も信頼度(cat_score)が高い一つのみを描画する場合
        
        # ユニークな数ごとにインデックスの集合を格納するディクショナリを初期化
        h_index_dict = {}

        # テンソルをイテレートしてインデックスを集合に追加
        for i, num in enumerate(target['h_index'].tolist()):
            if num in h_index_dict:
                h_index_dict[num].add(i)
            else:
                h_index_dict[num] = {i}

        # prsnt("h_index_dict", h_index_dict)
        for unique_key, index_list in h_index_dict.items():
            max_index, max_obj_value = find_max_active_obj_value_index(index_list, target, obj_thr)
          

            h_cat_score = target['h_cat_score'][max_index]
            o_cat_score = target['o_cat_score'][max_index]
            if h_cat_score >= hand_thr :
                if o_cat_score >= obj_thr:
                    results.append([max_index, 1])
                else:
                    results.append([max_index, 0])
          

        # resultには一枚の画像に対してunique_objに倣って
        # 最適な複数のhoi_num_queryのインデックスが入っている

        # predの情報を集約する
        pre_doh100_hand_boxes = []
        pre_doh100_hand_scores = []
        pre_pred_obj_boxes = []
        pre_pred_obj_scores = []
        pre_pred_confidence_scores = []

        for index, flag in results:
            h_index = target['h_index'][index]
            o_index = target['o_index'][index]

            h_labels = target['h_cat_label']
            h_bbox = target['boxes'][h_index]
            h_cat_score = target['h_cat_score'][index]

            o_labels = target['o_cat_label']
            o_bbox = target['boxes'][o_index]
            o_cat_score = target['o_cat_score'][index]

            pair_action = target['pair_action'][index]
            interaction_score = 1 - pair_action[-1]

             


            pre_doh100_hand_boxes.append(h_bbox)
            pre_doh100_hand_scores.append(h_cat_score)
            # handがobject pairを持たないと推論(後処理)された時
            if h_index == o_index:
                pre_pred_obj_boxes.append([0.0, 0.0, 0.0, 0.0])
                interaction_score_2 = h_cat_score
            
            # handがobject pairを持つと推論(後処理)された時
            else:     
                pre_pred_obj_boxes.append(o_bbox)
                interaction_score_2 = h_cat_score * o_cat_score

            pre_pred_obj_scores.append(o_cat_score)
            pre_pred_confidence_scores.append(interaction_score_2)

        doh100_hand_boxes.append(pre_doh100_hand_boxes)
        doh100_hand_scores.append(pre_doh100_hand_scores)
        pred_obj_boxes.append(pre_pred_obj_boxes)
        pred_obj_scores.append(pre_pred_obj_scores)
        pred_confidence_scores.append(pre_pred_confidence_scores)

        results = []

        # gtの情報を集約する
        pre_gt_hand_boxes = []
        pre_gt_obj_boxes = []
    
        for k in gt["pair_boxes"].cpu().numpy():
            pre_gt_hand_boxes.append(k[:4])
            if np.array_equal(k[:4], k[4:]) or np.array_equal(k[4:], [-1, -1, -1, -1]):
                pre_gt_obj_boxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                pre_gt_obj_boxes.append(k[4:])
        
        gt_hand_boxes.append(pre_gt_hand_boxes)
        gt_obj_boxes.append(pre_gt_obj_boxes)

    # # 引数のデータを準備
    # args_dict = {
    #     "doh100_hand_boxes": any_to_list(doh100_hand_boxes),
    #     "doh100_hand_scores": any_to_list(doh100_hand_scores),
    #     "pred_obj_boxes": any_to_list(pred_obj_boxes),
    #     "pred_obj_scores": any_to_list(pred_obj_scores),
    #     "pred_confidence_scores": any_to_list(pred_confidence_scores),
    #     "gt_hand_boxes": any_to_list(gt_hand_boxes),
    #     "gt_obj_boxes": any_to_list(gt_obj_boxes),
    #     "iou_thres": [0.75, 0.5, 0.25]
    # }
    # # JSONファイルに保存
    # with open('get_ap_ho_args.json', 'w') as f:
    #     json.dump(args_dict, f, indent=2)

    doh100_hand_boxes = any_to_numpy(doh100_hand_boxes)
    doh100_hand_scores = any_to_numpy(doh100_hand_scores)
    pred_obj_boxes = any_to_numpy(pred_obj_boxes)
    pred_obj_scores = any_to_numpy(pred_obj_scores)
    pred_confidence_scores = any_to_numpy(pred_confidence_scores)
    gt_hand_boxes = any_to_numpy(gt_hand_boxes)
    gt_obj_boxes = any_to_numpy(gt_obj_boxes)

    ap_results = {}
    for iou_thres in [0.75, 0.5, 0.25]:
        prec, rec, ap = metrics_utils.get_AP_HO(doh100_hand_boxes, doh100_hand_scores, pred_obj_boxes, pred_obj_scores, pred_confidence_scores, 
                                gt_hand_boxes, gt_obj_boxes, iou_thres=iou_thres)
        # print('AP(IoU >{:.2f}): {:.4f}'.format(iou_thres, ap))
        ap_results[iou_thres] = ap

    return ap_results






def any_to_numpy(value):
    if isinstance(value, list):
        return np.array([any_to_numpy(item) for item in value])
    elif torch.is_tensor(value):  # PyTorchのテンソルの場合
        return value.cpu().detach().numpy()
    elif isinstance(value, np.ndarray):
        return value
    return value
   
    
def any_to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif torch.is_tensor(value):  # PyTorchのテンソルの場合
        return any_to_list(value.cpu().detach().numpy())
    elif isinstance(value, list):
        return [any_to_list(item) for item in value]
    return value

def process_target(targets, target_sizes, args):
    for idx, (target, target_size) in enumerate(zip(targets, target_sizes)):
        labels = target['labels']
        valid_boxes_inds = (labels > 0)

        targets[idx]['boxes'] = rescale_bboxes(target['boxes'], target_size) # boxes
        if args.task == 'AOD':
            targets[idx]['pair_boxes'] = rescale_pairs(target['pair_boxes'], target_size) # pairs
        elif args.task == 'ASOD':
            targets[idx]['pair_boxes'] = rescale_triplet(target['pair_boxes'], target_size)

    return targets


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Tensorをリストに変換
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}  # 辞書の各要素を変換
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]  # リストの各要素を変換
    else:
        return obj  # Tensor以外の場合はそのまま返す
    
def draw_bbox(image, labels, bbox, color, count, thickness=4):
    x_min, y_min, x_max, y_max = bbox
    # 整数に変換
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    # バウンディングボックスの描画
    label = labels[count]
    label = label.item()
    color = color.get(label, (0, 0, 0))  # クラスに対応する色を取得し、存在しない場合は黒色を使用
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

def draw_GT_bbox_pair(image, pair_bbox, color, thickness=4):
    h_x_min, h_y_min, h_x_max, h_y_max = pair_bbox[:4]
    o_x_min, o_y_min, o_x_max, o_y_max = pair_bbox[4:]

    if o_x_min == -1 and o_y_min == -1 and o_x_max == -1 and o_y_max == -1:
        # 整数に変換
        h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
         # バウンディングボックスの描画
        h_color = color.get('Hand')  # クラスに対応する色を取得し、存在しない場合は黒色を使用
        cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)

        #それぞれのboxの中心点を描画
        h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
        cv2.circle(image, h_center, 5, h_color, -1)

        return image

    else:
        # 整数に変換
        h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
        o_x_min, o_y_min, o_x_max, o_y_max = map(int, [o_x_min, o_y_min, o_x_max, o_y_max])

        # バウンディングボックスの描画
        h_color = color.get('Hand')  
        o_color = color.get('Active_Obj') 
        cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)
        cv2.rectangle(image, (o_x_min, o_y_min), (o_x_max, o_y_max), o_color, thickness)


        # ペアのbboxの中心同士をむすぶ線を描画
        h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
        o_center = ((o_x_min + o_x_max) // 2, (o_y_min + o_y_max) // 2)
        cv2.line(image, h_center, o_center, h_color, 2)

        # 矢印を描画する
        draw_direction(image, h_center, o_center, h_color, 2)


        #それぞれのboxの中心点を描画
        cv2.circle(image, h_center, 5, h_color, -1)
        cv2.circle(image, o_center, 5, o_color, -1)

        return image
    
def draw_GT_bbox_triplet(image, triplet_bbox, color, thickness=4):
    h_x_min, h_y_min, h_x_max, h_y_max = triplet_bbox[:4]
    o_x_min, o_y_min, o_x_max, o_y_max = triplet_bbox[4:8]
    so_x_min, so_y_min, so_x_max, so_y_max = triplet_bbox[8:]

    if o_x_min == -1 and o_y_min == -1 and o_x_max == -1 and o_y_max == -1 and so_x_min == -1 and so_y_min == -1 and so_x_max == -1 and so_y_max == -1:
        # 整数に変換
        h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
         # バウンディングボックスの描画
        h_color = color.get('Hand')  
        cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)

        #それぞれのboxの中心点を描画
        h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
        cv2.circle(image, h_center, 5, h_color, -1)

        return image
    
    elif so_x_min == -1 and so_y_min == -1 and so_x_max == -1 and so_y_max == -1:
        # 整数に変換
        h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
        o_x_min, o_y_min, o_x_max, o_y_max = map(int, [o_x_min, o_y_min, o_x_max, o_y_max])

        # バウンディングボックスの描画
        h_color = color.get('Hand')  
        o_color = color.get('Active_Obj')
        cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)
        cv2.rectangle(image, (o_x_min, o_y_min), (o_x_max, o_y_max), o_color, thickness)


        # ペアのbboxの中心同士をむすぶ線を描画
        h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
        o_center = ((o_x_min + o_x_max) // 2, (o_y_min + o_y_max) // 2)
        cv2.line(image, h_center, o_center, h_color, 2)

        # 矢印を描画する
        draw_direction(image, h_center, o_center, h_color, 2)


        #それぞれのboxの中心点を描画
        cv2.circle(image, h_center, 5, h_color, -1)
        cv2.circle(image, o_center, 5, o_color, -1)

        return image
    else:
        # 整数に変換
        h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
        o_x_min, o_y_min, o_x_max, o_y_max = map(int, [o_x_min, o_y_min, o_x_max, o_y_max])
        so_x_min, so_y_min, so_x_max, so_y_max = map(int, [so_x_min, so_y_min, so_x_max, so_y_max])

        # バウンディングボックスの描画
        h_color = color.get('Hand')  
        o_color = color.get('Active_Obj')
        so_color = color.get('Second_Obj')
        cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)
        cv2.rectangle(image, (o_x_min, o_y_min), (o_x_max, o_y_max), o_color, thickness)
        cv2.rectangle(image, (so_x_min, so_y_min), (so_x_max, so_y_max), so_color, thickness)


        # ペアのbboxの中心同士をむすぶ線を描画
        h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
        o_center = ((o_x_min + o_x_max) // 2, (o_y_min + o_y_max) // 2)
        so_center = ((so_x_min + so_x_max) // 2, (so_y_min + so_y_max) // 2)
        cv2.line(image, h_center, o_center, h_color, 2)
        cv2.line(image, o_center, so_center, o_color, 2)


        # 矢印を描画する
        draw_direction(image, h_center, o_center, h_color, 2)
        draw_direction(image, o_center, so_center, o_color, 2)


        #それぞれのboxの中心点を描画
        cv2.circle(image, h_center, 5, h_color, -1)
        cv2.circle(image, o_center, 5, o_color, -1)
        cv2.circle(image, so_center, 5, so_color, -1)

        return image








def draw_bbox_pair(image, target, color, count, thickness=4):
    h_index = target['h_index'][count]
    o_index = target['o_index'][count]

    h_bbox = target['boxes'][h_index]
    h_cat_score = target['h_cat_score'][count]

    o_bbox = target['boxes'][o_index]
    o_cat_score = target['o_cat_score'][count]

    h_x_min, h_y_min, h_x_max, h_y_max = h_bbox
    o_x_min, o_y_min, o_x_max, o_y_max = o_bbox
   

    # 整数に変換
    h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
    o_x_min, o_y_min, o_x_max, o_y_max = map(int, [o_x_min, o_y_min, o_x_max, o_y_max])

    # バウンディングボックスの描画
    h_color = color.get('Hand')  # クラスに対応する色を取得
    o_color = color.get('Active_Obj')  # クラスに対応する色を取得し、存在しない場合は黒色を使用
    cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)
    cv2.rectangle(image, (o_x_min, o_y_min), (o_x_max, o_y_max), o_color, thickness)

     # バウンディングボックスの左上にスコアを描画
    cv2.putText(image, f"h_score: {h_cat_score:.2f}", (h_x_min + 5, h_y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1, cv2.LINE_AA)
    cv2.putText(image, f"o_score: {o_cat_score:.2f}", (o_x_min + 5, o_y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, o_color, 1, cv2.LINE_AA)


    # ペアのbboxの中心同士をむすぶ線を描画
    h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
    o_center = ((o_x_min + o_x_max) // 2, (o_y_min + o_y_max) // 2)
    cv2.line(image, h_center, o_center, h_color, 2)

    # 矢印を描画する
    draw_direction(image, h_center, o_center, h_color, 2)


    #それぞれのboxの中心点を描画
    cv2.circle(image, h_center, 5, h_color, -1)
    cv2.circle(image, o_center, 5, o_color, -1)

    return image

def draw_bbox_triplet(image, target, color, count, thickness=4):
    h_index = target['h_index'][count]
    o_index = target['o_index'][count]
    so_index = target['so_index'][count]

    h_bbox = target['boxes'][h_index]
    h_cat_score = target['h_cat_score'][count]

    o_bbox = target['boxes'][o_index]
    o_cat_score = target['o_cat_score'][count]

    so_bbox = target['boxes'][so_index]
    so_cat_score = target['so_cat_score'][count]

    h_x_min, h_y_min, h_x_max, h_y_max = h_bbox
    o_x_min, o_y_min, o_x_max, o_y_max = o_bbox
    so_x_min, so_y_min, so_x_max, so_y_max = so_bbox

    # 整数に変換
    h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])
    o_x_min, o_y_min, o_x_max, o_y_max = map(int, [o_x_min, o_y_min, o_x_max, o_y_max])
    so_x_min, so_y_min, so_x_max, so_y_max = map(int, [so_x_min, so_y_min, so_x_max, so_y_max])

    # バウンディングボックスの描画
    h_color = color.get('Hand')  # クラスに対応する色を取得
    o_color = color.get('Active_Obj')  # クラスに対応する色を取得し、存在しない場合は黒色を使用
    so_color = color.get('Second_Obj')  # クラスに対応する色を取得し、存在しない場合は黒色を使用
    cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)
    cv2.rectangle(image, (o_x_min, o_y_min), (o_x_max, o_y_max), o_color, thickness)
    cv2.rectangle(image, (so_x_min, so_y_min), (so_x_max, so_y_max), so_color, thickness)

    # バウンディングボックスの左上にスコアを描画
    cv2.putText(image, f"h_score: {h_cat_score:.2f}", (h_x_min + 5, h_y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1, cv2.LINE_AA)
    cv2.putText(image, f"o_score: {o_cat_score:.2f}", (o_x_min + 5, o_y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, o_color, 1, cv2.LINE_AA)
    cv2.putText(image, f"so_score: {so_cat_score:.2f}", (so_x_min + 5, so_y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, so_color, 1, cv2.LINE_AA)


    # ペアのbboxの中心同士をむすぶ線を描画
    h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
    o_center = ((o_x_min + o_x_max) // 2, (o_y_min + o_y_max) // 2)
    so_center = ((so_x_min + so_x_max) // 2, (so_y_min + so_y_max) // 2)
    cv2.line(image, h_center, o_center, h_color, 2)
    cv2.line(image, o_center, so_center, o_color, 2)

    # 矢印を描画する
    draw_direction(image, h_center, o_center, h_color, 2)
    draw_direction(image, o_center, so_center, o_color, 2)


    #それぞれのboxの中心点を描画
    cv2.circle(image, h_center, 5, h_color, -1)
    cv2.circle(image, o_center, 5, o_color, -1)
    cv2.circle(image, so_center, 5, so_color, -1)

    return image


def draw_hand_bbox(image, target, color, count, thickness=4):
    h_index = target['h_index'][count]

    h_labels = target['h_cat_label']
    h_bbox = target['boxes'][h_index]
    h_cat_score = target['h_cat_score'][count]


    h_x_min, h_y_min, h_x_max, h_y_max = h_bbox
 

    # 整数に変換
    h_x_min, h_y_min, h_x_max, h_y_max = map(int, [h_x_min, h_y_min, h_x_max, h_y_max])

    # バウンディングボックスの描画
    h_color = color.get('Hand')
    cv2.rectangle(image, (h_x_min, h_y_min), (h_x_max, h_y_max), h_color, thickness)

     # バウンディングボックスの左上にスコアを描画
    cv2.putText(image, f"h_score: {h_cat_score:.2f}", (h_x_min + 5, h_y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1, cv2.LINE_AA)

    #boxの中心点を描画
    h_center = ((h_x_min + h_x_max) // 2, (h_y_min + h_y_max) // 2)
    cv2.circle(image, h_center, 5, h_color, -1)

    return image


def draw_direction(image, start_point, end_point, color, thickness=2, arrow_length=20):
    # 線分の長さを計算
    length = ((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)**0.5
    
    # 線分の長さがゼロまたは非常に小さい場合は矢印を描画せずに終了
    if length <= 1e-6:
        return
    
    # 線分の方向ベクトルを計算
    dx = (end_point[0] - start_point[0]) / length
    dy = (end_point[1] - start_point[1]) / length
    
    # 矢印の終点を計算
    arrow_end_point = (int(end_point[0] - arrow_length * dx), int(end_point[1] - arrow_length * dy))
    
    # 矢印を描画
    cv2.arrowedLine(image, start_point, arrow_end_point, color, thickness)

def find_max_value_index(index_list, target):
    # index_list の中での最大スコアのインデックスを見つける
    score_list = target['o_cat_score']
    max_score = float('-inf')
    max_index = None
    for index in index_list:
        if score_list[index] > max_score:
            max_score = score_list[index]
            max_index = index
    return max_index, max_score

def find_max_active_obj_value_index(index_list, target, obj_thr=0.5):
    score_list = target['o_cat_score']
    o_cat_label = target['o_cat_label']
    max_score = -10
    max_index = None
    for index in index_list:
        if o_cat_label[index].item() == 2:  # o_cat_labelが2の場合のみ考慮する
            if score_list[index] > max_score and score_list[index] >= obj_thr:
                max_score = score_list[index]
                max_index = index

    # obj_cat_labelが2のものがない場合は、obj_cat_labelが2以外の最大スコアのインデックスを見つける
    if max_index is None:
        hand_score_list = target['h_cat_score']
        hand_cat_label = target['h_cat_label']
        max_hand_score = -10
        for index in index_list:
            if hand_score_list[index] > max_hand_score:
                max_hand_score = hand_score_list[index]
                max_index = index

    return max_index, max_score

def find_max_second_obj_value_index(index_list, target, obj_thr=0.5, sobj_thr=0.5):
    max_query_index = None

    sobj_score_list = target['so_cat_score']
    sobj_cat_label = target['so_cat_label']
    max_sobj_score = -10
    for index in index_list:
        if sobj_cat_label[index].item() == 2:  # obj_cat_labelが2の場合のみ考慮する
            if sobj_cat_label[index] > max_sobj_score and sobj_cat_label[index] >= sobj_thr:
                max_sobj_score = sobj_cat_label[index]
                max_query_index = index

    if max_query_index is None:
        obj_score_list = target['o_cat_score']
        obj_cat_label = target['o_cat_label']
        max_obj_score = -10
        for index in index_list:
            if obj_cat_label[index].item() == 2:  # obj_cat_labelが2の場合のみ考慮する
                if obj_score_list[index] > max_obj_score and obj_score_list[index] >= obj_thr:
                    max_obj_score = obj_score_list[index]
                    max_query_index = index

    # obj_cat_labelが2のものがない場合は、obj_cat_labelが2以外の最大スコアのインデックスを見つける
    if max_query_index is None:
        hand_score_list = target['h_cat_score']
        hand_cat_label = target['h_cat_label']
        max_hand_score = -10
        for index in index_list:
            if hand_score_list[index] > max_hand_score:
                max_hand_score = hand_score_list[index]
                max_query_index = index

    return max_query_index