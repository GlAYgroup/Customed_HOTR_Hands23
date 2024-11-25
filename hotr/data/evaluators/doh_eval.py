# Copyright (c) KakaoBrain, Inc. and its affiliates. All Rights Reserved
"""
V-COCO evaluator that works in distributed mode.
"""
import os
import numpy as np
import torch

from hotr.util.misc import all_gather
from hotr.metrics.vcoco.ap_role import APRole
from functools import partial

def init_vcoco_evaluators(human_act_name, object_act_name):
    role_eval1 = APRole(act_name=object_act_name, scenario_flag=True, iou_threshold=0.5)
    role_eval2 = APRole(act_name=object_act_name, scenario_flag=False, iou_threshold=0.5)

    return role_eval1, role_eval2

class DohEvaluator(object):
    def __init__(self, args):
        self.img_ids = []
        self.eval_imgs = []
        self.action_idx = args.valid_ids

    def update(self, outputs):
        img_ids = list(np.unique(list(outputs.keys())))
        prepre_dict = {"p_hand_box":[], "p_hand_score":[], "p_obj_box":[], "p_obj_score":[], "p_confidence_score":[], "gt_hand_box":[],"gt_obj_box":[]}

        for img_num, img_id in enumerate(img_ids):
            print(f"Finding HOI pair : [{(img_num+1):>4}/{len(img_ids):<4}]" ,flush=True, end="\r")
            prediction = outputs[img_id]['prediction']
            target = outputs[img_id]['target']

            # filter optimal(h_box, o_box) pair
            self.filt_opt(prediction, prepre_dict, hand_th=0.9)
            print("prepre_dict", prepre_dict)

       

    def filt_opt(self, prediction, prepre_dict, hand_th=0.9):
        # score with prediction
        h_index, h_cat_label, h_cat_score, o_index, o_cat_label, o_cat_score, boxes, pair_action_max_index = list(map(lambda x: prediction[x], \
            ['h_index', 'h_cat_label', 'h_cat_score', 'o_index', 'o_cat_label', 'o_cat_score', 'boxes', 'pair_action_max_index']))
        print("h_index", h_index)
        for i in range(len(h_index)): 
            if h_cat_score[i] < hand_th:
                h_index[i] = -1
        unique_h_index = torch.unique(h_index)
        print("unique_h_index", unique_h_index)

        print("o_cat_score", o_cat_score)
        score = 0
        list_p_hand_box = []
        list_p_hand_score = []
        list_p_obj_box = []
        list_p_obj_score = []

        for i in unique_h_index: # 予測されたある一つの手に対する処理
            if i == -1:
                continue

            for query, k in enumerate(h_index):
                if i == k and score < o_cat_score[query]:
                    count = query
                    score = o_cat_score[query]
            print("query", count)
            print("score", score)
            p_hand_box = boxes[h_index[count]].to('cpu').detach().numpy().copy()
            p_hand_score = h_cat_score[count]
            pair_action = pair_action_max_index[count].to('cpu').detach().numpy().copy()
            if pair_action_max_index[count] == 0 or h_index[count] == o_index[count]:
                p_obj_box = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                p_obj_box = boxes[o_index[count]].to('cpu').detach().numpy().copy()
            p_obj_score = o_cat_score[count]

            list_p_hand_box.append(p_hand_box)
            list_p_hand_score.append(p_hand_score)
            list_p_obj_box.append(p_obj_box)
            list_p_obj_score.append(p_obj_score)




        prepre_dict["p_hand_box"].append(list_p_hand_box)
        prepre_dict["p_hand_score"].append(list_p_hand_score)
        prepre_dict["p_obj_box"].append(list_p_obj_box)
        prepre_dict["p_obj_score"].append(list_p_obj_score)

        prepre_dict["p_confidence_score"].append(list_p_obj_score)

    
        return prepre_dict



                
            

      


            





