# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime

from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP

class HOTR(nn.Module):
    def __init__(self, detr,
                 num_hoi_queries,
                 num_actions,
                 interaction_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 hoi_aux_loss,
                 return_obj_class=None,
                 task='AOD', # for ASOD
                 args=None  # hand_poseフラグを追加
                 ):
        super().__init__()
        self.task = task # for ASOD
        self.hand_pose = args.hand_pose  # hand_pose状態を保持

        # * Instance Transformer ---------------
        self.detr = detr
        if freeze_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim = detr.transformer.d_model
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_hoi_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.H_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.action_embed = nn.Linear(hidden_dim, num_actions+1)


        # Second Object用のFFNを追加（taskが'ASOD'のときのみ）
        if self.task == 'ASOD':
            self.SO_Pointer_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # * Hand pose Options for adding Interaction Query -------------------
        if self.hand_pose == 'add_in_d_0':
            # 21点 * 2座標 = 42次元をhidden_dimへ射影するMLP
            self.hand_pose_mlp = MLP(42, hidden_dim, hidden_dim, 3)
            # concat後の (hidden_dim*2) を hidden_dim に戻すためのプロジェクション層
            self.hand_pose_proj_0 = nn.Linear(hidden_dim*2, hidden_dim)
        if self.hand_pose == 'add_in_d_1':
            self.hand_pose_proj_1 = MLP(hidden_dim+42, hidden_dim, hidden_dim, 3)
        # --------------------------------------------------------------------

        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1]+1]
        # ------------------------------------------------------------------

        # * Transformer Options ---------------------------------------------
        self.interaction_transformer = interaction_transformer

        if share_enc: # share encoder
            self.interaction_transformer.encoder = detr.transformer.encoder

        if pretrained_dec: # free variables for interaction decoder
            self.interaction_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.interaction_transformer.decoder.parameters():
                p.requires_grad_(True)
        # ---------------------------------------------------------------------

        # * Loss Options -------------------
        self.tau = temperature
        self.hoi_aux_loss = hoi_aux_loss
        # ----------------------------------


    def forward(self, samples: NestedTensor, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        # ----------------------------------------------

        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        start_time = time.time()
        hs, _ = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1])
        inst_repr = F.normalize(hs[-1], p=2, dim=2) # instance representations

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        object_detection_time = time.time() - start_time
        # -----------------------------------------------

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        start_time = time.time()
        assert hasattr(self, 'interaction_transformer'), "Missing Interaction Transformer."

        # [Hand Pose]
        if (self.hand_pose == 'add_in_d_0' or self.hand_pose == 'add_in_d_1') and targets is not None:
            hand_queries_batch = []
            hidden_dim = inst_repr.shape[-1]

            for tgt in targets:
                if "hand_2d_key_points" in tgt and tgt["hand_2d_key_points"].numel() > 0:
                    kp = tgt["hand_2d_key_points"]        # (N_hand, 21, 2)
                    conf = tgt["hand_kp_confidence"]      # (N_hand,)

                    # (N_hand, 21, 2) -> (N_hand, 42)
                    N_hand = kp.shape[0]
                    kp_flat = kp.view(N_hand, 42)

                    if self.hand_pose == 'add_in_d_0':
                        # キーポイントをhidden_dimへ射影
                        hp_feats = self.hand_pose_mlp(kp_flat) # (N_hand, hidden_dim)
                    elif self.hand_pose == 'add_in_d_1':
                        hp_feats = kp_flat

                    # confidence順に並び替え
                    sorted_idx = conf.argsort(descending=True)
                    hp_feats = hp_feats[sorted_idx]

                    # num_queriesと比較し切り取りまたはパディング
                    if hp_feats.shape[0] > self.num_queries:
                        hp_feats = hp_feats[:self.num_queries]
                    elif hp_feats.shape[0] < self.num_queries:
                        pad_len = self.num_queries - hp_feats.shape[0]
                        if self.hand_pose == 'add_in_d_0':
                            pad = torch.zeros(pad_len, hidden_dim, device=hp_feats.device)
                        elif self.hand_pose == 'add_in_d_1':
                            pad = torch.zeros(pad_len, 42, device=hp_feats.device)
                        hp_feats = torch.cat([hp_feats, pad], dim=0)
                else:
                    # キーポイントがない場合は全て0でパディング
                    if self.hand_pose == 'add_in_d_0':
                        hp_feats = torch.zeros(self.num_queries, hidden_dim, device=src.device)
                    elif self.hand_pose == 'add_in_d_1':
                        hp_feats = torch.zeros(self.num_queries, 42, device=src.device)
                    # キーポイントがない場合でもhand_pose_mlpを通すダミー処理
                    # dummy_input = torch.zeros(1, 42, device=src.device)
                    # dummy_feats = self.hand_pose_mlp(dummy_input) # (1, hidden_dim)
                    # dummy_feats = dummy_feats.expand(self.num_queries, -1) # (num_queries, hidden_dim)
                    # hp_feats = dummy_feats if hp_feats is None else hp_feats

                hand_queries_batch.append(hp_feats.unsqueeze(0))  # (1, num_queries, hidden_dim)

            hand_query_stack = torch.cat(hand_queries_batch, dim=0) # (bs, num_queries, hidden_dim)
            base_query = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1) # (bs, num_queries, hidden_dim)

            # concatでpositional embeddingと手の特徴を結合
            # concat: (bs, num_queries, hidden_dim) + (bs, num_queries, hidden_dim)
            # → (bs, num_queries, hidden_dim*2)
            query_concat = torch.cat([base_query, hand_query_stack], dim=2)

            # MLP(Linear)で (hidden_dim*2) → hidden_dim に戻す
            if self.hand_pose == 'add_in_d_0':
                query_embed = self.hand_pose_proj_0(query_concat) # (bs, num_queries, hidden_dim)
            elif self.hand_pose == 'add_in_d_1':
                query_embed = self.hand_pose_proj_1(query_concat)
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)        

        # interaction_hs = self.interaction_transformer(self.detr.input_proj(src), mask, self.query_embed.weight, pos[-1])[0] # interaction representations
        interaction_hs = self.interaction_transformer(self.detr.input_proj(src), mask, query_embed, pos[-1])[0] # interaction representations

        # [HO Pointers]
        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1)
        outputs_hidx = [(torch.bmm(H_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]

        # [S Pointers]
        if self.task == 'ASOD':
            SO_Pointer_reprs = F.normalize(self.SO_Pointer_embed(interaction_hs), p=2, dim=-1)
            outputs_soidx = [(torch.bmm(SO_Pointer_repr, inst_repr.transpose(1, 2))) / self.tau for SO_Pointer_repr in SO_Pointer_reprs]
            
        # [Action Classification]
        outputs_action = self.action_embed(interaction_hs)
        # --------------------------------------------------
        hoi_detection_time = time.time() - start_time
        hoi_recognition_time = max(hoi_detection_time - object_detection_time, 0)
        # -------------------------------------------------------------------

        # [Target Classification]
        if self.return_obj_class:
            detr_logits = outputs_class[-1, ..., self._valid_obj_ids]
            o_indices = [output_oidx.max(-1)[-1] for output_oidx in outputs_oidx]
            obj_logit_stack = [torch.stack([detr_logits[batch_, o_idx, :] for batch_, o_idx in enumerate(o_indice)], 0) for o_indice in o_indices]
            outputs_obj_class = obj_logit_stack

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_actions": outputs_action[-1],
            "hoi_recognition_time": hoi_recognition_time,
        }
        # Second Objectの出力を追加
        if self.task == 'ASOD':
            out['pred_soidx'] = outputs_soidx[-1]

        if self.return_obj_class: out["pred_obj_logits"] = outputs_obj_class[-1]

        if self.hoi_aux_loss:  # auxiliary loss
            if self.task == 'ASOD':
                out['hoi_aux_outputs'] = \
                    self._set_aux_loss_with_tgt(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_obj_class, outputs_soidx) \
                    if self.return_obj_class else \
                    self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_soidx)
            elif self.task == 'AOD':
                out['hoi_aux_outputs'] = \
                    self._set_aux_loss_with_tgt(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_obj_class) \
                    if self.return_obj_class else \
                    self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_soidx=None):
        if self.task == 'ASOD' and outputs_soidx is not None:
            return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_soidx': f}
                for a, b, c, d, e, f in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_soidx[:-1])]
        
        elif self.task == 'AOD':
            return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_tgt, outputs_soidx=None):
        if self.task == 'ASOD' and outputs_soidx is not None:
            return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f, 'pred_soidx': g}
                for a, b, c, d, e, f, g in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_tgt[:-1],
                    outputs_soidx[:-1])]
        elif self.task == 'AOD':
            return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
                    for a, b, c, d, e, f in zip(
                        outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                        outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                        outputs_hidx[:-1],
                        outputs_oidx[:-1],
                        outputs_action[:-1],
                        outputs_tgt[:-1])]