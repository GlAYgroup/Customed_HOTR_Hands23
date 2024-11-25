# ------------------------------------------------------------------------
# HOTR official code : hotr/models/post_process.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import time
import copy
import torch
import torch.nn.functional as F
from torch import nn
from hotr.util import box_ops

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, HOIDet, task='AOD'):
        super().__init__()
        self.HOIDet = HOIDet
        self.task = task

    @torch.no_grad()
    def forward(self, outputs, target_sizes, threshold=0, dataset='coco'):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)#boxesのクラス分類
        scores, labels = prob[..., :-1].max(-1)# torch.Size([2, 100]) -> batch_size, num_boxes
        # scoresがそれぞれのbboxの物体検出のスコア
        # labelsがそれぞれのbboxのクラスラベル

        # print("prob", prob)
        # print("scores", scores)
        # print("labels", labels)

        # convert boxes from [0; 1] to original image scales, 正規化から元の画像のスケールに変換
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        # print("boxes", boxes)

        # Preidction Branch for HOI detection
        if self.HOIDet:
            if dataset == 'vcoco':
                """ Compute HOI triplet prediction score for V-COCO.
                Our scoring function follows the implementation details of UnionDet.
                """
                out_time = outputs['hoi_recognition_time']

                start_time = time.time()
                pair_actions = torch.sigmoid(outputs['pred_actions'])
                h_prob = F.softmax(outputs['pred_hidx'], -1)
                h_idx_score, h_indices = h_prob.max(-1)

                o_prob = F.softmax(outputs['pred_oidx'], -1)
                o_idx_score, o_indices = o_prob.max(-1)
                hoi_recognition_time = (time.time() - start_time) + out_time

                # print("/large/maeda/HOTR/hotr/models/post_process.py")
                # print("h_idx", h_indices)
                # print("o_idx", o_indices)
                results = []
                # iterate for batch size
                for batch_idx, (s, l, b) in enumerate(zip(scores, labels, boxes)):
                  
                    h_inds = (l == 1) & (s > threshold)
                    o_inds = (s > threshold)

                    h_box, h_cat = b[h_inds], s[h_inds]
                    o_box, o_cat = b[o_inds], s[o_inds]

                    # for scenario 1 in v-coco dataset
                    o_inds = torch.cat((o_inds, torch.ones(1).type(torch.bool).to(o_inds.device)))
                    o_box = torch.cat((o_box, torch.Tensor([0, 0, 0, 0]).unsqueeze(0).to(o_box.device)))

                    result_dict = {
                        'h_box': h_box, 'h_cat': h_cat,
                        'o_box': o_box, 'o_cat': o_cat,
                        'scores': s, 'labels': l, 'boxes': b
                    }

                    h_inds_lst = (h_inds == True).nonzero(as_tuple=False).squeeze(-1)
                    o_inds_lst = (o_inds == True).nonzero(as_tuple=False).squeeze(-1)

                    K = boxes.shape[1]
                    n_act = pair_actions[batch_idx][:, :-1].shape[-1]
                    score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
                    sorted_score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
                    id_score = torch.zeros((K, K+1)).to(pair_actions[batch_idx].device)

                    # Score function
                    for hs, h_idx, os, o_idx, pair_action in zip(h_idx_score[batch_idx], h_indices[batch_idx], o_idx_score[batch_idx], o_indices[batch_idx], pair_actions[batch_idx]):
                        matching_score = (1-pair_action[-1]) # no interaction score
                        if h_idx == o_idx: o_idx = -1
                        if matching_score > id_score[h_idx, o_idx]:
                            id_score[h_idx, o_idx] = matching_score
                            sorted_score[:, h_idx, o_idx] = matching_score * pair_action[:-1]
                        score[:, h_idx, o_idx] += matching_score * pair_action[:-1]

                    score += sorted_score
                    score = score[:, h_inds, :]
                    score = score[:, :, o_inds]

                    result_dict.update({
                        'pair_score': score,
                        'hoi_recognition_time': hoi_recognition_time,
                    })

                    results.append(result_dict)

            elif dataset == 'hico-det':
                """ Compute HOI triplet prediction score for HICO-DET.
                For HICO-DET, we follow the same scoring function but do not accumulate the results.
                """
                out_time = outputs['hoi_recognition_time']

                start_time = time.time()
                out_obj_logits, out_verb_logits = outputs['pred_obj_logits'], outputs['pred_actions']
                out_verb_logits = outputs['pred_actions']

                # actions
                matching_scores = (1-out_verb_logits.sigmoid()[..., -1:]) #* (1-out_verb_logits.sigmoid()[..., 57:58])
                verb_scores = out_verb_logits.sigmoid()[..., :-1] * matching_scores

                # hbox, obox
                outputs_hrepr, outputs_orepr = outputs['pred_hidx'], outputs['pred_oidx']
                obj_scores, obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)

                h_prob = F.softmax(outputs_hrepr, -1)
                h_idx_score, h_indices = h_prob.max(-1)

                # targets
                o_prob = F.softmax(outputs_orepr, -1)
                o_idx_score, o_indices = o_prob.max(-1)
                hoi_recognition_time = (time.time() - start_time) + out_time

                # hidx, oidx
                sub_boxes, obj_boxes = [], []
                for batch_id, (box, h_idx, o_idx) in enumerate(zip(boxes, h_indices, o_indices)):
                    sub_boxes.append(box[h_idx, :])
                    obj_boxes.append(box[o_idx, :])
                sub_boxes = torch.stack(sub_boxes, dim=0)
                obj_boxes = torch.stack(obj_boxes, dim=0)

                # accumulate results (iterate through interaction queries)
                results = []
                for os, ol, vs, ms, sb, ob in zip(obj_scores, obj_labels, verb_scores, matching_scores, sub_boxes, obj_boxes):
                    sl = torch.full_like(ol, 0) # self.subject_category_id = 0 in HICO-DET
                    l = torch.cat((sl, ol))
                    b = torch.cat((sb, ob))
                    results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})
                    vs = vs * os.unsqueeze(1)
                    ids = torch.arange(b.shape[0])
                    res_dict = {
                        'verb_scores': vs.to('cpu'),
                        'sub_ids': ids[:ids.shape[0] // 2],
                        'obj_ids': ids[ids.shape[0] // 2:],
                        'hoi_recognition_time': hoi_recognition_time
                    }
                    results[-1].update(res_dict)

            elif dataset == 'doh':
                """ Compute HOI triplet prediction score for DOH.
                Our scoring function follows the implementation details of UnionDet.
                """
                # print("/large/maeda/HOTR/hotr/models/post_process.py")
                out_time = outputs['hoi_recognition_time']

                start_time = time.time()
                pair_actions = torch.sigmoid(outputs['pred_actions']) # (batch_size, hoi_num_queries, num_actions)
                h_prob = F.softmax(outputs['pred_hidx'], -1)
                h_idx_score, h_indices = h_prob.max(-1) # (batch_size, hoi_num_queries)

                o_prob = F.softmax(outputs['pred_oidx'], -1)
                o_idx_score, o_indices = o_prob.max(-1) # (batch_size, hoi_num_queries)
                hoi_recognition_time = (time.time() - start_time) + out_time

                results = []

                # iterate for batch size
                for batch_idx, (s, l, b, h_idx, o_idx, p_a) in enumerate(zip(scores, labels, boxes, h_indices, o_indices, pair_actions)):
                    #ボックスに対するインデックス、h_cat, o_cat, (h_idx_score, o_idx_score)
                    
                    # print("batch_idx", batch_idx)
                    # print("h_idx", h_idx)
                    # print("o_idx", o_idx)
                    h_cat_score = s[h_idx] # h_idxはbboxのインデックスであり、s[h_idx]でそのインデックスのbboxのスコア信頼度を取得
                    h_cat_label = l[h_idx] # h_idxはbboxのインデックスであり、l[h_idx]でそのインデックスのbboxのクラスラベルを取得
                    o_cat_score = s[o_idx]
                    o_cat_label = l[o_idx]

                    # print("h_cat", h_cat_score)
                    # print("h_cat_label", h_cat_label)
                    # print("o_cat", o_cat_score)
                    # print("o_cat_label", o_cat_label)

                    # Calculate the max value and index for each list in the tensor
                    max_values, max_indices = torch.max(p_a, dim=1)

                    max_values, max_indices
                    # print("max_values", max_values)
                    # print("max_indices", max_indices)
                    # print("max_indices.type", type(max_indices))

                    result_dict = {
                        'h_cat_score': h_cat_score, 'h_cat_label': h_cat_label, 'h_index': h_idx,
                        'o_cat_score': o_cat_score, 'o_cat_label': o_cat_label, 'o_index': o_idx,
                        'pair_action': p_a, 'pair_action_max_score':max_values, "pair_action_max_index":max_indices,
                        'boxes': b, 'hoi_recognition_time': hoi_recognition_time
                    }
                
                    
                    results.append(result_dict)
            elif dataset == 'hands23':
                    if self.task == 'AOD':
                        """ Compute HOI triplet prediction score for Hands23
                        Our scoring function follows the implementation details of UnionDet.
                        """
                        out_time = outputs['hoi_recognition_time']

                        start_time = time.time()
                        pair_actions = torch.sigmoid(outputs['pred_actions']) # (batch_size, hoi_num_queries, num_actions)
                        h_prob = F.softmax(outputs['pred_hidx'], -1)
                        h_idx_score, h_indices = h_prob.max(-1) # (batch_size, hoi_num_queries)

                        o_prob = F.softmax(outputs['pred_oidx'], -1)
                        o_idx_score, o_indices = o_prob.max(-1) # (batch_size, hoi_num_queries)
                        hoi_recognition_time = (time.time() - start_time) + out_time

                        results = []

                        # iterate for batch size
                        for batch_idx, (s, l, b, h_idx, o_idx, p_a) in enumerate(zip(scores, labels, boxes, h_indices, o_indices, pair_actions)):
                            #ボックスに対するインデックス、h_cat, o_cat, (h_idx_score, o_idx_score)
                            
                            # print("batch_idx", batch_idx)
                            # print("h_idx", h_idx)
                            # print("o_idx", o_idx)
                            h_cat_score = s[h_idx] # h_idxはbboxのインデックスであり、s[h_idx]でそのインデックスのbboxのスコア信頼度を取得
                            h_cat_label = l[h_idx] # h_idxはbboxのインデックスであり、l[h_idx]でそのインデックスのbboxのクラスラベルを取得
                            o_cat_score = s[o_idx]
                            o_cat_label = l[o_idx]

                            # print("h_cat", h_cat_score)
                            # print("h_cat_label", h_cat_label)
                            # print("o_cat", o_cat_score)
                            # print("o_cat_label", o_cat_label)

                            # Calculate the max value and index for each list in the tensor
                            max_values, max_indices = torch.max(p_a, dim=1)
                            # print("max_values", max_values)
                            # print("max_indices", max_indices)
                            # print("max_indices.type", type(max_indices))

                            result_dict = {
                                'h_cat_score': h_cat_score, 'h_cat_label': h_cat_label, 'h_index': h_idx,
                                'o_cat_score': o_cat_score, 'o_cat_label': o_cat_label, 'o_index': o_idx,
                                'pair_action': p_a, 'pair_action_max_score':max_values, "pair_action_max_index":max_indices,
                                'boxes': b, 'hoi_recognition_time': hoi_recognition_time
                            }
                            results.append(result_dict)

                    elif self.task == 'ASOD':
                        """ Compute HOI triplet and Second object prediction score for Hands23
                        Our scoring function follows the implementation details of UnionDet.
                        """
                        # print("/large/maeda/HOTR/hotr/models/post_process.py")
                        out_time = outputs['hoi_recognition_time']

                        start_time = time.time()
                        pair_actions = torch.sigmoid(outputs['pred_actions']) # (batch_size, hoi_num_queries, num_actions)
                        h_prob = F.softmax(outputs['pred_hidx'], -1)
                        h_idx_score, h_indices = h_prob.max(-1) # (batch_size, hoi_num_queries)

                        o_prob = F.softmax(outputs['pred_oidx'], -1)
                        o_idx_score, o_indices = o_prob.max(-1) # (batch_size, hoi_num_queries)

                        so_prob = F.softmax(outputs['pred_soidx'], -1)
                        so_idx_score, so_indices = so_prob.max(-1) # (batch_size, hoi_num_queries)

                        hoi_recognition_time = (time.time() - start_time) + out_time

                        results = []

                        # iterate for batch size
                        for batch_idx, (s, l, b, h_idx, o_idx, so_idx, p_a) in enumerate(zip(scores, labels, boxes, h_indices, o_indices, so_indices, pair_actions)):
                            #ボックスに対するインデックス、h_cat, o_cat, (h_idx_score, o_idx_score)
                            
                            # print("batch_idx", batch_idx)
                            # print("h_idx", h_idx)
                            # print("o_idx", o_idx)
                            h_cat_score = s[h_idx] # h_idxはbboxのインデックスであり、s[h_idx]でそのインデックスのbboxのスコア信頼度を取得
                            h_cat_label = l[h_idx] # h_idxはbboxのインデックスであり、l[h_idx]でそのインデックスのbboxのクラスラベルを取得
                            o_cat_score = s[o_idx]
                            o_cat_label = l[o_idx]
                            so_cat_score = s[so_idx]
                            so_cat_label = l[so_idx]

                            # print("h_cat", h_cat_score)
                            # print("h_cat_label", h_cat_label)
                            # print("o_cat", o_cat_score)
                            # print("o_cat_label", o_cat_label)

                            # Calculate the max value and index for each list in the tensor
                            max_values, max_indices = torch.max(p_a, dim=1)

                            result_dict = {
                                'h_cat_score': h_cat_score, 'h_cat_label': h_cat_label, 'h_index': h_idx,
                                'o_cat_score': o_cat_score, 'o_cat_label': o_cat_label, 'o_index': o_idx,
                                'so_cat_score': so_cat_score, 'so_cat_label': so_cat_label, 'so_index': so_idx,
                                'pair_action': p_a, 'pair_action_max_score':max_values, "pair_action_max_index":max_indices,
                                'boxes': b, 'hoi_recognition_time': hoi_recognition_time
                            }
                            results.append(result_dict)
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        # print("results", results)
        return results
