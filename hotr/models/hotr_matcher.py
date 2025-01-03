# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr_matcher.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from hotr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

import hotr.util.misc as utils
import wandb

class HungarianPairMatcher(nn.Module):
    def __init__(self, args):
        """Creates the matcher
        Params:
            cost_action: This is the relative weight of the multi-label action classification error in the matching cost
            cost_hbox: This is the relative weight of the classification error for human idx in the matching cost
            cost_obox: This is the relative weight of the classification error for object idx in the matching cost
        """
        super().__init__()
        self.cost_action = args.set_cost_act
        self.cost_hbox = self.cost_obox = args.set_cost_idx
        self.cost_target = args.set_cost_tgt
        # 追加のパラメータ
        if args.task == 'ASOD':
            self.cost_sobox = args.set_cost_soidx
        self.task = args.task  # タスク情報を保存

        self.log_printer = args.wandb
        self.is_vcoco = (args.dataset_file == 'vcoco')
        self.is_hico = (args.dataset_file == 'hico-det')
        self.is_doh = (args.dataset_file == 'doh')
        self.is_hands23 = (args.dataset_file == 'hands23')

        if self.is_vcoco:
            self.valid_ids = args.valid_ids
            self.invalid_ids = args.invalid_ids
        elif self.is_doh:
            self.valid_ids = args.valid_ids
            self.invalid_ids = args.invalid_ids
        elif self.is_hands23:
            self.valid_ids = args.valid_ids
            self.invalid_ids = args.invalid_ids
        assert self.cost_action != 0 or self.cost_hbox != 0 or self.cost_obox != 0, "all costs cant be 0"

    def reduce_redundant_gt_box(self, tgt_bbox, indices):
        """Filters redundant Ground-Truth Bounding Boxes
        Due to random crop augmentation, there exists cases where there exists
        multiple redundant labels for the exact same bounding box and object class.
        This function deals with the redundant labels for smoother HOTR training.
        """
        tgt_bbox_unique, map_idx, idx_cnt = torch.unique(tgt_bbox, dim=0, return_inverse=True, return_counts=True)

        k_idx, bbox_idx = indices
        triggered = False
        if (len(tgt_bbox) != len(tgt_bbox_unique)):
            map_dict = {k: v for k, v in enumerate(map_idx)}
            map_bbox2kidx = {int(bbox_id): k_id for bbox_id, k_id in zip(bbox_idx, k_idx)}

            bbox_lst, k_lst = [], []
            for bbox_id in bbox_idx:
                if map_dict[int(bbox_id)] not in bbox_lst:
                    bbox_lst.append(map_dict[int(bbox_id)])
                    k_lst.append(map_bbox2kidx[int(bbox_id)])
            bbox_idx = torch.tensor(bbox_lst)
            k_idx = torch.tensor(k_lst)
            tgt_bbox_res = tgt_bbox_unique
        else:
            tgt_bbox_res = tgt_bbox
        bbox_idx = bbox_idx.to(tgt_bbox.device)

        return tgt_bbox_res, k_idx, bbox_idx

    @torch.no_grad()
    def forward(self, outputs, targets, indices, log=False):
        # print("outputs", outputs)
        # print("targets", targets)
        assert "pred_actions" in outputs, "There is no action output for pair matching"
        num_obj_queries = outputs["pred_boxes"].shape[1]
        bs, num_queries = outputs["pred_actions"].shape[:2]
        detr_query_num = outputs["pred_logits"].shape[1] \
            if (outputs["pred_oidx"].shape[-1] == (outputs["pred_logits"].shape[1] + 1)) else -1

        return_list = []
        if self.log_printer and log:
            log_dict = {'h_cost': [], 'o_cost': [], 'act_cost': []}
            if self.is_hico: log_dict['tgt_cost'] = []
            if self.task == 'ASOD':
                log_dict['so_cost'] = []

        for batch_idx in range(bs):
            tgt_bbox = targets[batch_idx]["boxes"] # (num_boxes, 4)
            tgt_cls = targets[batch_idx]["labels"] # (num_boxes)

            if self.is_vcoco:
                targets[batch_idx]["pair_actions"][:, self.invalid_ids] = 0
                keep_idx = (targets[batch_idx]["pair_actions"].sum(dim=-1) != 0)
                targets[batch_idx]["pair_boxes"] = targets[batch_idx]["pair_boxes"][keep_idx]
                targets[batch_idx]["pair_actions"] = targets[batch_idx]["pair_actions"][keep_idx]
                targets[batch_idx]["pair_targets"] = targets[batch_idx]["pair_targets"][keep_idx]

                tgt_pbox = targets[batch_idx]["pair_boxes"] # (num_pair_boxes, 8)
                tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 29)
                tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)

                tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)
                tgt_obox = tgt_pbox[:, 4:] # (num_pair_boxes, 4)



            elif self.is_hico:
                tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 117)
                tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)

                tgt_hbox = targets[batch_idx]["sub_boxes"] # (num_pair_boxes, 4)
                tgt_obox = targets[batch_idx]["obj_boxes"] # (num_pair_boxes, 4)

            elif self.is_doh:
                tgt_pbox = targets[batch_idx]["pair_boxes"] # (num_pair_boxes, 8)
                tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 5)
                tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)

                tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)
                tgt_obox = tgt_pbox[:, 4:] # (num_pair_boxes, 4)
                # print('tgt_pbox', tgt_pbox)
                # print('tgt_act', tgt_act)
                # print('tgt_tgt', tgt_tgt)
                # print('tgt_hbox', tgt_hbox)
                # print('tgt_obox', tgt_obox)

            elif self.is_hands23:
                tgt_pbox = targets[batch_idx]["pair_boxes"] # (num_pair_boxes, 8or12)
                tgt_act = targets[batch_idx]["pair_actions"] # (num_pair_boxes, 5)
                tgt_tgt = targets[batch_idx]["pair_targets"] # (num_pair_boxes)
                

                if self.task == 'AOD':
                    tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)
                    tgt_obox = tgt_pbox[:, 4:] # (num_pair_boxes, 4)
                if self.task == 'ASOD':
                    tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)
                    tgt_obox = tgt_pbox[:, 4:8] # (num_pair_boxes, 4)
                    tgt_sobox = tgt_pbox[:, 8:] # (num_pair_boxes, 4)

                    tgt_so_tgt = targets[batch_idx]["triplet_targets"] # (num_pair_boxes)

            # find which gt boxes match the h, o boxes in the pair
            if self.is_vcoco:
                hbox_with_cls = torch.cat([tgt_hbox, torch.ones((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
            elif self.is_hico:
                hbox_with_cls = torch.cat([tgt_hbox, torch.zeros((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
            elif self.is_doh:
                hbox_with_cls = torch.cat([tgt_hbox, torch.ones((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)
            elif self.is_hands23:
                hbox_with_cls = torch.cat([tgt_hbox, torch.ones((tgt_hbox.shape[0], 1)).to(tgt_hbox.device)], dim=1)

            obox_with_cls = torch.cat([tgt_obox, tgt_tgt.unsqueeze(-1)], dim=1)
            obox_with_cls[obox_with_cls[:, :4].sum(dim=1) == -4, -1] = -1 # turn the class of occluded objects to -1
            # Second Objectの処理（ASODの場合）
            if self.task == 'ASOD':
                so_box_with_cls = torch.cat([tgt_sobox, tgt_so_tgt.unsqueeze(-1)], dim=1)
                so_box_with_cls[so_box_with_cls[:, :4].sum(dim=1) == -4, -1] = -1  # 無効なクラスを-1に設定

            bbox_with_cls = torch.cat([tgt_bbox, tgt_cls.unsqueeze(-1)], dim=1)
            bbox_with_cls, k_idx, bbox_idx = self.reduce_redundant_gt_box(bbox_with_cls, indices[batch_idx])
            bbox_with_cls = torch.cat((bbox_with_cls, torch.as_tensor([-1.]*5).unsqueeze(0).to(tgt_cls.device)), dim=0)

            cost_hbox = torch.cdist(hbox_with_cls, bbox_with_cls, p=1)
            cost_obox = torch.cdist(obox_with_cls, bbox_with_cls, p=1)
            # Second Objectのコスト（ASODの場合）
            if self.task == 'ASOD':
                cost_sobox = torch.cdist(so_box_with_cls, bbox_with_cls, p=1)

            # find which gt boxes matches which prediction in K
            h_match_indices = torch.nonzero(cost_hbox == 0, as_tuple=False) # (num_hbox, num_boxes)
            o_match_indices = torch.nonzero(cost_obox == 0, as_tuple=False) # (num_obox, num_boxes)

            # Second Objectのマッチング（ASODの場合）
            if self.task == 'ASOD':
                so_match_indices = torch.nonzero(cost_sobox == 0, as_tuple=False)  # (num_sobox, num_boxes)
                tgt_soids = []

            tgt_hids, tgt_oids = [], []

            # print("h_match_indices", h_match_indices)
            # print("o_match_indices", o_match_indices)
            # print("so_match_indices", so_match_indices)
            # print("len(h_match_indices)", len(h_match_indices))
            # print("len(o_match_indices)", len(o_match_indices))
            # print("len(so_match_indices)", len(so_match_indices))
            # print("cost_h_box.size()", cost_hbox.size())
            # print("cost_o_box.size()", cost_obox.size())
            # print("cost_sobox.size()", cost_sobox.size())
            # # h_match_indices = torch.empty((0, 0), dtype=torch.long, device=h_match_indices.device)
            # # o_match_indices = torch.empty((0, 0), dtype=torch.long, device=o_match_indices.device)
            # # so_match_indices = torch.empty((0, 0), dtype=torch.long, device=so_match_indices.device)
            # print("after_h_match_indices", h_match_indices)
            # print("after_o_match_indices", o_match_indices)
            # print("after_so_match_indices", so_match_indices)
            

            # obtain ground truth indices for h
            if len(h_match_indices) != len(o_match_indices):
                print("batch_idx", batch_idx)
                print("targets", targets)
                print("outputs", outputs)
                print("h_match_indices", h_match_indices)
                print("o_match_indices", o_match_indices)
                print("len(h_match_indices)", len(h_match_indices))
                print("len(o_match_indices)", len(o_match_indices))
                # import pdb; pdb.set_trace()
                # 空のテンソルで初期化
                h_match_indices = torch.empty((0, 2), dtype=torch.long, device=h_match_indices.device)
                o_match_indices = torch.empty((0, 2), dtype=torch.long, device=o_match_indices.device)

            if self.task == 'ASOD' and (len(h_match_indices) != len(so_match_indices) or (len(h_match_indices) != len(o_match_indices)) or (len(o_match_indices) != len(so_match_indices))):
                print("batch_idx", batch_idx)
                print("target", targets)
                print("outputs", outputs)
                print("h_match_indices", h_match_indices)
                print("so_match_indices", so_match_indices)
                print("len(h_match_indices)", len(h_match_indices))
                print("len(so_match_indices)", len(so_match_indices))
                # import pdb; pdb.set_trace()
                # 空のテンソルで初期化
                h_match_indices = torch.empty((0, 2), dtype=torch.long, device=h_match_indices.device)
                o_match_indices = torch.empty((0, 2), dtype=torch.long, device=o_match_indices.device)
                so_match_indices = torch.empty((0, 2), dtype=torch.long, device=so_match_indices.device)


            for idx in range(len(h_match_indices)):
                h_match_idx = h_match_indices[idx]
                o_match_idx = o_match_indices[idx]

                hbox_idx, H_bbox_idx = h_match_idx
                obox_idx, O_bbox_idx = o_match_idx
                if O_bbox_idx == (len(bbox_with_cls)-1): # if the object class is -1
                    O_bbox_idx = H_bbox_idx # the target object may not appear

                GT_idx_for_H = (bbox_idx == H_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_H = k_idx[GT_idx_for_H]
                tgt_hids.append(query_idx_for_H)

                GT_idx_for_O = (bbox_idx == O_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_O = k_idx[GT_idx_for_O]
                tgt_oids.append(query_idx_for_O)

                # Second Objectの処理（ASODの場合）
                if self.task == 'ASOD':
                    so_match_idx = so_match_indices[idx]
                    sobox_idx, SO_bbox_idx = so_match_idx

                    if SO_bbox_idx == (len(bbox_with_cls)-1): # if the object class is -1
                        SO_bbox_idx = H_bbox_idx # the target object may not appear

                    GT_idx_for_SO = (bbox_idx == SO_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                    query_idx_for_SO = k_idx[GT_idx_for_SO]
                    tgt_soids.append(query_idx_for_SO)

            # print("batch_idx", batch_idx)
            # print("tgt_hids", tgt_hids)
            # print("tgt_oids", tgt_oids)
            # print("tgt_soids", tgt_soids)

            # check if empty
            if len(tgt_hids) == 0: tgt_hids.append(torch.as_tensor([-1])) # we later ignore the label -1
            if len(tgt_oids) == 0: tgt_oids.append(torch.as_tensor([-1])) # we later ignore the label -1
            if self.task == 'ASOD' and len(tgt_soids) == 0: tgt_soids.append(torch.as_tensor([-1]))  # we later ignore the label -1

            tgt_sum = (tgt_act.sum(dim=-1)).unsqueeze(0)
            flag = False
            if tgt_act.shape[0] == 0:
                tgt_act = torch.zeros((1, tgt_act.shape[1])).to(targets[batch_idx]["pair_actions"].device)
                targets[batch_idx]["pair_actions"] = torch.zeros((1, targets[batch_idx]["pair_actions"].shape[1])).to(targets[batch_idx]["pair_actions"].device)
                if self.is_hico:
                    pad_tgt = -1 # outputs["pred_obj_logits"].shape[-1]-1
                    tgt_tgt = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_targets"])
                    targets[batch_idx]["pair_targets"] = torch.tensor([pad_tgt]).to(targets[batch_idx]["pair_targets"].device)
                tgt_sum = (tgt_act.sum(dim=-1) + 1).unsqueeze(0)

            # Concat target label
            tgt_hids = torch.cat(tgt_hids)
            tgt_oids = torch.cat(tgt_oids)
            if self.task == 'ASOD':
                tgt_soids = torch.cat(tgt_soids)
               

            out_hprob = outputs["pred_hidx"][batch_idx].softmax(-1)
            out_oprob = outputs["pred_oidx"][batch_idx].softmax(-1)
            out_act = outputs["pred_actions"][batch_idx].clone()
            if self.task == 'ASOD':
                out_soprob = outputs["pred_soidx"][batch_idx].softmax(-1)

            if self.is_vcoco: out_act[..., self.invalid_ids] = 0
            if self.is_hico:
                out_tgt = outputs["pred_obj_logits"][batch_idx].softmax(-1)
                out_tgt[..., -1] = 0 # don't get cost for no-object
            if self.is_doh: out_act[..., self.invalid_ids] = 0 # dohではinvaidなアクションクラスはないので書く必要ないが念の為記した
            if self.is_hands23: out_act[..., self.invalid_ids] = 0 # hands23ではinvaidなアクションクラスはないので書く必要ないが念の為記した
            
            tgt_act = torch.cat([tgt_act, torch.zeros(tgt_act.shape[0]).unsqueeze(-1).to(tgt_act.device)], dim=-1)


            cost_hclass = -out_hprob[:, tgt_hids] # [batch_size * num_queries, detr.num_queries+1]
            cost_oclass = -out_oprob[:, tgt_oids] # [batch_size * num_queries, detr.num_queries+1]
            # Second Objectのコスト（ASODの場合）
            if self.task == 'ASOD':
                cost_soclass = -out_soprob[:, tgt_soids]
                
                


            cost_pos_act = (-torch.matmul(out_act, tgt_act.t().float())) / tgt_sum
            cost_neg_act = (torch.matmul(out_act, (~tgt_act.bool()).type(torch.int64).t().float())) / (~tgt_act.bool()).type(torch.int64).sum(dim=-1).unsqueeze(0)
            cost_action = cost_pos_act + cost_neg_act

            h_cost = self.cost_hbox * cost_hclass
            o_cost = self.cost_obox * cost_oclass
            act_cost = self.cost_action * cost_action
            if self.task == 'ASOD':
                so_cost = self.cost_sobox * cost_soclass

            C = h_cost + o_cost + act_cost
            # Second Objectのコストを総コストに追加（ASODの場合）
            if self.task == 'ASOD':
                C += so_cost
            
            if self.is_hico:
                cost_target = -out_tgt[:, tgt_tgt]
                tgt_cost = self.cost_target * cost_target
                C += tgt_cost
            C = C.view(num_queries, -1).cpu()

            # print("h_cost", h_cost) 
            # print("o_cost", o_cost)
            # print("so_cost", so_cost)

            return_list.append(linear_sum_assignment(C))
            # print('return_list', return_list)
            # print("tgt_hids", tgt_hids)
            # print("tgt_oids", tgt_oids)
            # print("tgt_soids", tgt_soids)
            # print("tgt_hbox", tgt_hbox)
            # print("tgt_obox", tgt_obox)
            # print("tgt_sobox", tgt_sobox)
            targets[batch_idx]["h_labels"] = tgt_hids.to(tgt_hbox.device)
            targets[batch_idx]["o_labels"] = tgt_oids.to(tgt_obox.device)
            if self.task == 'ASOD':
                targets[batch_idx]["so_labels"] = tgt_soids.to(tgt_sobox.device)
            log_act_cost = torch.zeros([1]).to(tgt_act.device) if tgt_act.shape[0] == 0 else act_cost.min(dim=0)[0].mean()

            # print("h_labels", targets[batch_idx]["h_labels"])  
            # print("o_labels", targets[batch_idx]["o_labels"])
            # print("so_labels", targets[batch_idx]["so_labels"])

            if self.log_printer and log:
                log_dict['h_cost'].append(h_cost.min(dim=0)[0].mean())
                log_dict['o_cost'].append(o_cost.min(dim=0)[0].mean())
                log_dict['act_cost'].append(act_cost.min(dim=0)[0].mean())
                if self.task == 'ASOD':
                    log_dict['so_cost'].append(so_cost.min(dim=0)[0].mean())

                if self.is_hico: log_dict['tgt_cost'].append(tgt_cost.min(dim=0)[0].mean())


        if self.log_printer and log:
            log_dict['h_cost'] = torch.stack(log_dict['h_cost']).mean()
            log_dict['o_cost'] = torch.stack(log_dict['o_cost']).mean()
            log_dict['act_cost'] = torch.stack(log_dict['act_cost']).mean()
            if self.task == 'ASOD':
                log_dict['so_cost'] = torch.stack(log_dict['so_cost']).mean()

            if self.is_hico: log_dict['tgt_cost'] = torch.stack(log_dict['tgt_cost']).mean()
            if utils.get_rank() == 0: wandb.log(log_dict)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in return_list], targets

def build_hoi_matcher(args):
    return HungarianPairMatcher(args)
