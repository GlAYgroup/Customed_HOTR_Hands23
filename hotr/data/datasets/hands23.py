# Copyright (c) Kakaobrain, Inc. and its affiliates. All Rights Reserved
"""
100doh dataset which returns image_id for evaluation.
"""
from pathlib import Path

from PIL import Image
import os
import numpy as np
import json
import torch
import torch.utils.data
import torchvision
import itertools

from torch.utils.data import Dataset


from hotr.data.datasets import builtin_meta
import hotr.data.transforms.hands23_transforms as T

import cv2


class handsDetection(Dataset):
    def __init__(self,
                 img_folder,
                 ann_file,
                 image_set,
                 filter_empty_gt=True,
                 transforms=None,
                 args=None,
                 ):
        self.img_folder = img_folder
        self.file_meta = dict()
        self._transforms = transforms

        self.image_set = image_set

        self.task = args.task
        self.hand_pose = args.hand_pose
        if self.task == 'AOD':
            self.inst_act_list = ['no_contact', 'other_person_contact', 'self_contact', 'object_contact'] #Hands23 the hand contact state(0~3)
            self.act_list = ['no_contact', 'other_person_contact', 'self_contact', 'object_contact'] #Hands23 the hand contact state(0~3)
        elif self.task == 'ASOD':
            self.inst_act_list = ['Hand', 'Hand_Active', 'Hand_Active_Second']
            self.act_list = ['Hand', 'Hand_Active', 'Hand_Active_Second']
        self.file_meta['action_classes'] = self.act_list
        self.hands_classes = ['N/A', 'hand', 'object']
        self.file_meta['hands_classes'] = self.hands_classes
        # self.images_attribute = ['boardgame', 'diy', 'drink', 'food', 'furniture', 'gardening', 'housework', 'packing', 'puzzle', 'repair', 'study', 'vlog']
        
        self.ann_file = ann_file
        print('ann_file', ann_file)
        with open(self.ann_file) as f:
            self.annotations = json.load(f)
        # if self.image_set == 'train':
        #     print('annotations', self.annotations.keys())
        #self.all_file = all_file
        self.filter_empty_gt = filter_empty_gt

       
        # Load Doh Dataset
        # self.doh_all = self.load_doh(self.ann_file)
        
        # Make Image Infos
        self.img_infos = self.load_img_infos()

    ############################################################################
    # Annotation Loader
    ############################################################################
    # >>> 1. instance & pair on the AOD task
    def load_instance_pair_annotations(self, image_index):
        cat_hand = self.hands_classes.index('hand')
        cat_obj = self.hands_classes.index('object')
        inst_action = [] 
        inst_bbox = []
        inst_category = []
        pair_bbox = []
        pair_action = [] #contact_state
        pair_target = []

        annotation = self.annotations[self.img_infos[image_index]]

        for side_annotation in annotation:
            new_hand_bbox = [side_annotation["x1"] * side_annotation["width"],
                side_annotation["y1"] * side_annotation["height"],
                side_annotation["x2"] * side_annotation["width"],
                side_annotation["y2"] * side_annotation["height"]]
            inst_bbox.append(new_hand_bbox)
            inst_category.append(cat_hand)
            
            new_inst_action = [0] * len(self.file_meta['action_classes'])
            new_inst_action[side_annotation["contact_state"]] = 1
            inst_action.append(new_inst_action)

            
            if side_annotation["obj_bbox"] is not None:
                new_obj_bbox = [side_annotation["obj_bbox"]["x1"] * side_annotation["width"],
                                side_annotation["obj_bbox"]["y1"] * side_annotation["height"],
                                side_annotation["obj_bbox"]["x2"] * side_annotation["width"],
                                side_annotation["obj_bbox"]["y2"] * side_annotation["height"]]
                inst_bbox.append(new_obj_bbox)
                inst_category.append(cat_obj)
                new_inst_action = [0] * len(self.file_meta['action_classes'])
                new_inst_action[side_annotation["contact_state"]] = 1
                inst_action.append(new_inst_action)

                new_pair_bbox = sum([new_hand_bbox, new_obj_bbox], [])
                pair_bbox.append(new_pair_bbox)

                new_pair_action = [0] * len(self.file_meta['action_classes'])
                new_pair_action[side_annotation["contact_state"]] = 1
                pair_action.append(new_pair_action)

                pair_target.append(cat_obj)
            else:
                new_pair_bbox = sum([new_hand_bbox, [-1, -1, -1, -1]], [])
                pair_bbox.append(new_pair_bbox)

                new_pair_action = [0] * len(self.file_meta['action_classes'])
                new_pair_action[0] = 1
                pair_action.append(new_pair_action)

                pair_target.append(-1)
            
        # numpy 配列に変換して2次元配列を作成
        inst_bbox = np.array(inst_bbox)
        inst_category = np.array(inst_category)
        inst_action = np.array(inst_action)
        pair_bbox = np.array(pair_bbox)
        pair_action = np.array(pair_action)
        pair_target = np.array(pair_target)
    
        return inst_bbox, inst_category, inst_action, pair_bbox, pair_action, pair_target

    # >>> 1. instance & pair on the ASOD task
    def load_instance_triplet_annotations(self, image_index):
        cat_hand = self.hands_classes.index('hand')
        cat_obj = self.hands_classes.index('object')
        inst_action = [] 
        inst_bbox = []
        inst_category = []
        pair_bbox = []
        pair_action = [] #contact_state
        pair_target = []
        pair_so_target = []
        ann_id = []

        #for hand pose
        if  self.hand_pose == 'add_in_d_0' or self.hand_pose == 'add_in_d_1' or self.hand_pose == 'add_in_d_2':
            hand_bboxes = []
            hand_2d_key_points = []
            hand_kp_confidence = []

        annotation = self.annotations[self.img_infos[image_index]]

        for side_annotation in annotation:
            new_hand_bbox = [side_annotation["x1"] * side_annotation["width"],
                side_annotation["y1"] * side_annotation["height"],
                side_annotation["x2"] * side_annotation["width"],
                side_annotation["y2"] * side_annotation["height"]]
            inst_bbox.append(new_hand_bbox)
            inst_category.append(cat_hand)
            
            new_inst_action = [0] * len(self.file_meta['action_classes'])
            # print('side_annotation["contact_state"]', side_annotation["contact_state"])
            # print('new_inst_action', new_inst_action)
            new_inst_action[side_annotation["contact_state"]] = 1
            inst_action.append(new_inst_action)

            if "ann_id" in side_annotation:
                ann_id.append(side_annotation["ann_id"])

            if  (self.hand_pose == 'add_in_d_0' or self.hand_pose == 'add_in_d_1' or self.hand_pose == 'add_in_d_2') and "pred_2d_keypoints" in side_annotation:
                hand_bboxes.append(new_hand_bbox)
                hand_2d_key_points.append(side_annotation["pred_2d_keypoints"])
                hand_kp_confidence.append(side_annotation["pred_confidence"])

            if side_annotation["second_obj_bbox"] is not None:
                new_obj_bbox = [side_annotation["obj_bbox"]["x1"] * side_annotation["width"],
                                side_annotation["obj_bbox"]["y1"] * side_annotation["height"],
                                side_annotation["obj_bbox"]["x2"] * side_annotation["width"],
                                side_annotation["obj_bbox"]["y2"] * side_annotation["height"]]
                inst_bbox.append(new_obj_bbox)
                inst_category.append(cat_obj)
                new_inst_obj_action = [0] * len(self.file_meta['action_classes'])
                new_inst_obj_action[side_annotation["contact_state"]] = 1
                inst_action.append(new_inst_obj_action)

                new_second_obj_bbox = [side_annotation["second_obj_bbox"]["x1"] * side_annotation["width"],
                                side_annotation["second_obj_bbox"]["y1"] * side_annotation["height"],
                                side_annotation["second_obj_bbox"]["x2"] * side_annotation["width"],
                                side_annotation["second_obj_bbox"]["y2"] * side_annotation["height"]]
                inst_bbox.append(new_second_obj_bbox)
                inst_category.append(cat_obj)
                new_inst_second_action = [0] * len(self.file_meta['action_classes'])
                new_inst_second_action[side_annotation["contact_state"]] = 1
                inst_action.append(new_inst_second_action)

                new_pair_bbox = sum([new_hand_bbox, new_obj_bbox, new_second_obj_bbox], [])
                pair_bbox.append(new_pair_bbox)

                new_pair_action = [0] * len(self.file_meta['action_classes'])
                new_pair_action[side_annotation["contact_state"]] = 1
                pair_action.append(new_pair_action)

                pair_target.append(cat_obj)
                pair_so_target.append(cat_obj)  
            elif side_annotation["obj_bbox"] is not None:
                new_obj_bbox = [side_annotation["obj_bbox"]["x1"] * side_annotation["width"],
                                side_annotation["obj_bbox"]["y1"] * side_annotation["height"],
                                side_annotation["obj_bbox"]["x2"] * side_annotation["width"],
                                side_annotation["obj_bbox"]["y2"] * side_annotation["height"]]
                inst_bbox.append(new_obj_bbox)
                inst_category.append(cat_obj)
                new_inst_action = [0] * len(self.file_meta['action_classes'])
                new_inst_action[side_annotation["contact_state"]] = 1
                inst_action.append(new_inst_action)

                new_pair_bbox = sum([new_hand_bbox, new_obj_bbox, [-1, -1, -1, -1]], [])
                pair_bbox.append(new_pair_bbox)

                new_pair_action = [0] * len(self.file_meta['action_classes'])
                new_pair_action[side_annotation["contact_state"]] = 1
                pair_action.append(new_pair_action)

                pair_target.append(cat_obj)
                pair_so_target.append(-1)
            else:
                new_pair_bbox = sum([new_hand_bbox, [-1, -1, -1, -1], [-1, -1, -1, -1]], [])
                pair_bbox.append(new_pair_bbox)

                new_pair_action = [0] * len(self.file_meta['action_classes'])
                new_pair_action[0] = 1
                pair_action.append(new_pair_action)

                pair_target.append(-1)
                pair_so_target.append(-1)

        # numpy 配列に変換して2次元配列を作成
        inst_bbox = np.array(inst_bbox)
        inst_category = np.array(inst_category)
        inst_action = np.array(inst_action)
        pair_bbox = np.array(pair_bbox)
        pair_action = np.array(pair_action)
        pair_target = np.array(pair_target)
        pair_so_target = np.array(pair_so_target)
        if self.hand_pose == 'add_in_d_0' or self.hand_pose == 'add_in_d_1' or self.hand_pose == 'add_in_d_2':
            hand_bboxes = np.array(hand_bboxes)
            hand_2d_key_points = np.array(hand_2d_key_points)
            hand_kp_confidence = np.array(hand_kp_confidence)
            return inst_bbox, inst_category, inst_action, pair_bbox, pair_action, pair_target, pair_so_target, hand_bboxes, hand_2d_key_points, hand_kp_confidence, ann_id
        else:
            return inst_bbox, inst_category, inst_action, pair_bbox, pair_action, pair_target, pair_so_target

    ############################################################################
    # Image Name Loader
    ############################################################################
    # image infos
    def load_img_infos(self):
        img_infos = []
        
        for info in self.annotations.keys():
            img_infos.append(info)

        return img_infos
    
    ############################################################################
    # Check Method
    ############################################################################
    def vis_sample_from_bbox(self, idx):
        # アノテーションを前処理したものが正しいか確認するための可視化関数
        img_info = self.img_infos[idx]
        print('img_info', img_info)
        target = self.get_ann_info(idx)
        image_path = os.path.join(self.img_folder, img_info)
        image = cv2.imread(image_path)
        
        # 画像が正しく読み込まれたか確認
        if image is None:
            print(f"Failed to read image at {image_path}")
            return
        
        for bbox in target['boxes']:
            x_min, y_min, x_max, y_max = bbox
            # 整数に変換
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            # バウンディングボックスの描画
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        basename = os.path.basename(img_info)
        img_path = os.path.join('hands23/vis_images/hands23_vis_by_vis_sample/', 'vis_' + basename)
        # ディレクトリが存在することを確認
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # 画像を保存
        print('img_path', img_path)
        success = cv2.imwrite(img_path, image)
        # 結果をチェック
        if not success:
            print(f"Failed to save image at {img_path}")

    def vis_sample_from_pair_boxes(self, idx):
        # アノテーションを前処理したものが正しいか確認するための可視化関数
        img_info = self.img_infos[idx]
        print('img_info', img_info)
        target = self.get_ann_info(idx)
        image_path = os.path.join(self.img_folder, img_info)
        image = cv2.imread(image_path)
        
        # 画像が正しく読み込まれたか確認
        if image is None:
            print(f"Failed to read image at {image_path}")
            return
        
        # pair_boxes からバウンディングボックスを描画
        for pair in target['pair_boxes']:
            # pair が Tensor の場合、NumPy 配列に変換
            if isinstance(pair, torch.Tensor):
                pair = pair.numpy()
            hand_box = pair[0:4]
            active_obj_box = pair[4:8]
            second_obj_box = pair[8:12]
            
            points = []  # 中心点を格納するリスト

            # Hand bounding box を描画
            if not np.all(hand_box == -1):
                x_min, y_min, x_max, y_max = hand_box.astype(int)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # 中心点を計算
                hand_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                points.append(hand_center)
                # ラベルを追加（オプション）
                cv2.putText(image, 'Hand', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                hand_center = None

            # Active Object bounding box を描画
            if not np.all(active_obj_box == -1):
                x_min, y_min, x_max, y_max = active_obj_box.astype(int)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                # 中心点を計算
                active_obj_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                points.append(active_obj_center)
                # ラベルを追加（オプション）
                cv2.putText(image, 'Active Object', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                active_obj_center = None

            # Second Object bounding box を描画
            if not np.all(second_obj_box == -1):
                x_min, y_min, x_max, y_max = second_obj_box.astype(int)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                # 中心点を計算
                second_obj_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                points.append(second_obj_center)
                # ラベルを追加（オプション）
                cv2.putText(image, 'Second Object', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                second_obj_center = None

            # 中心点同士を線で結ぶ
            if len(points) >= 2:
                # 中心点を順番に線で結ぶ
                for i in range(len(points) - 1):
                    if points[i] is not None and points[i+1] is not None:
                        cv2.line(image, points[i], points[i+1], (0, 255, 255), 2)
    
        basename = os.path.basename(img_info)
        img_path = os.path.join('hands23/vis_images/hands23_vis_by_vis_sample/', 'vis_' + basename)
        # ディレクトリが存在することを確認
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # 画像を保存
        print('img_path', img_path)
        success = cv2.imwrite(img_path, image)
        # 結果をチェック
        if not success:
            print(f"Failed to save image at {img_path}")

    ############################################################################
    # Preprocessing
    ############################################################################
    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        image = Image.open(os.path.join(self.img_folder, img_info)).convert('RGB')
        target = self.get_ann_info(idx)

        # print("/large/maeda/HOTR/hotr/data/datasets/doh.py", img_info)

        w, h = image.size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(image, target) # "size" gets converted here
        
        return img, target
        # return image

    ############################################################################
    # Get Method
    ############################################################################
    def __getitem__(self, idx):
        img, target = self.prepare_img(idx)
        # print('=================START===============')
        # print('image_info:', self.img_infos[idx])
        # print('idx:', idx)
        # print("/large/maeda/HOTR/hotr/data/datasets/doh.py target: ")
        # print(target)
        # print('=================END=================')
        return img, target
    
    def __len__(self):
        return len(self.img_infos)
    
    def get_actions(self):
        return self.act_list
    
    def get_inst_action(self):
        return self.inst_act_list
    
    def get_human_action(self):
        return self.inst_act_list
    
    def get_object_action(self):
        return self.inst_act_list
    
    def get_ann_info(self, idx):
        img_idx = idx

        # load each annotation
        if self.task == 'AOD':#ひとまずhand poseのフラグはADOには追加していない
            inst_bbox, inst_label, inst_actions, pair_bbox, pair_actions, pair_targets = self.load_instance_pair_annotations(img_idx)
        elif self.task == 'ASOD':
            if self.hand_pose == 'add_in_d_0' or self.hand_pose == 'add_in_d_1' or self.hand_pose == 'add_in_d_2':
                inst_bbox, inst_label, inst_actions, pair_bbox, pair_actions, pair_targets, pair_so_targets, hand_bboxes, hand_2d_key_points, hand_kp_confidence, ann_id = self.load_instance_triplet_annotations(img_idx)
            else:
                inst_bbox, inst_label, inst_actions, pair_bbox, pair_actions, pair_targets, pair_so_targets = self.load_instance_triplet_annotations(img_idx)
        else:
            raise ValueError(f'unknown task {self.task}')

        sample = {
            'image_id' : torch.tensor([img_idx]),
            'boxes': torch.as_tensor(inst_bbox, dtype=torch.float32),
            'labels': torch.tensor(inst_label, dtype=torch.int64),
            'inst_actions': torch.tensor(inst_actions, dtype=torch.int64),
            'pair_boxes': torch.as_tensor(pair_bbox, dtype=torch.float32),
            'pair_actions': torch.tensor(pair_actions, dtype=torch.int64),
            'pair_targets': torch.tensor(pair_targets, dtype=torch.int64),            
            #100dohのGTにはinst_actions, pair_actions情報がないので適切なインスタンス数に対応した0のリストを入れておく
        }
        if self.task == 'ASOD':
            sample['triplet_targets'] = torch.tensor(pair_so_targets, dtype=torch.int64)

        if self.hand_pose == 'add_in_d_0' or self.hand_pose == 'add_in_d_1' or self.hand_pose == 'add_in_d_2':
            if hand_bboxes is not None and len(hand_bboxes) > 0:
                sample['hand_bboxes'] = torch.as_tensor(hand_bboxes, dtype=torch.float32)
                sample['hand_2d_key_points'] = torch.as_tensor(hand_2d_key_points, dtype=torch.float32)
                sample['hand_kp_confidence'] = torch.as_tensor(hand_kp_confidence, dtype=torch.float32)
                sample['ann_id'] = torch.tensor(ann_id, dtype=torch.int64)
                    

        # print('sample', sample)
        return sample
    
    def get_image_info(self, idx):
        img_info = self.img_infos[idx]
        return img_info

    ############################################################################
    # Number Method
    ############################################################################
    def num_category(self):
        return len(self.hands_classes)
    
    def num_action(self):
        return len(self.act_list)
    
    def num_human_act(self):
        return len(self.inst_act_list)


def make_hoi_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_simple_hoi_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            normalize
        ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



def build(image_set, args):
    root = Path(args.data_path)
    print('TASK------>>>>', args.task)

    part_del = 'excl_nohand_'
    second_only = '2nd_only_'
    hand_kp = 'w_2dkp_'
    id = 'id_'
    # order id->second_only->part_del->hand_kp 
    
    assert root.exists(), f'provided Hands23 path {root} does not exist'
    PATHS = {
            "train": (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (id + part_del + hand_kp + 'train.json')),
            "val"  : (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (id + part_del + hand_kp + 'val.json')),
            "test" : (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (part_del + hand_kp + 'test.json')),
        }
    
    # PATHS = {
    #         "train": (root / 'hands23_data' / 'sub_allMergedBlur', root / 'hands23_data' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'train.json')),
    #         "val"  : (root / 'hands23_data' / 'sub_allMergedBlur', root / 'hands23_data' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'val.json')),
    #         "test" : (root / 'hands23_data' / 'sub_allMergedBlur', root / 'hands23_data' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'test.json')),
    #     }
    if args.check:
        # PATHS = {
        #     "train": (root / 'hands23_data' / 'sub_100' / 'sub_allMergedBlur', root / 'hands23_data' / 'sub_100' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'train.json')),
        #     "val"  : (root / 'hands23_data' / 'sub_100' / 'sub_allMergedBlur', root / 'hands23_data' / 'sub_100' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'val.json')),
        #     "test" : (root / 'hands23_data' / 'sub_100' / 'sub_allMergedBlur', root / 'hands23_data' / 'sub_100' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'test.json')),
        # }
        PATHS = {
            "train": (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (part_del + hand_kp + 'train.json')),
            "val"  : (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (part_del + hand_kp + 'val.json')),
            "test" : (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (second_only + part_del + 'test.json')),
        }
    img_folder, ann_file = PATHS[image_set]
    print('Annotation path is', ann_file)

    dataset = handsDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        image_set = image_set,
        filter_empty_gt=True,
        transforms = make_hoi_transforms(image_set),
        args = args,
        # transforms = make_simple_hoi_transforms(image_set)
    )
    dataset.file_meta['dataset_file'] = args.dataset_file
    dataset.file_meta['image_set'] = image_set

    return dataset



def main(image_set, args):
    # rootをPathオブジェクトに変更
    root = Path('hands23')
    task = 'ASOD'
    print('task', task)
    print('root', root)
    
    assert root.exists(), f'provided Hands23 path {root} does not exist'
    part_del = 'excl_nohand_'
    second_only = '2nd_only_'
    hand_kp = 'w_2dkp_'
    
    PATHS = {
            "train": (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (part_del + hand_kp + 'train.json')),
            "val"  : (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (part_del + hand_kp + 'val.json')),
            "test" : (root / 'hands23_data' / 'allMergedBlur', root / 'hands23_data' / 'doh_format_dataset' / (part_del + hand_kp + 'test.json')),
        }
    
    # PATHS = {
    #         "train": (root / 'hands23_data' / 'sub_allMergedBlur', root / 'hands23_data' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'train.json')),
    #         "val"  : (root / 'hands23_data' / 'sub_allMergedBlur', root / 'hands23_data' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'val.json')),
    #         "test" : (root / 'hands23_data' / 'sub_allMergedBlur', root / 'hands23_data' / 'doh_format_sub_dataset' / ('sub_' + second_only + part_del + 'test.json')),
    #     }

    img_folder, ann_file = PATHS[image_set]

    dataset = handsDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        image_set = image_set,
        filter_empty_gt=True,
        transforms = make_hoi_transforms(image_set),
        args = args,
        # transforms = make_simple_hoi_transforms(image_set)
    )

    return dataset