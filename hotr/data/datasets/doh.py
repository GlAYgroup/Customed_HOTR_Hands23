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
import hotr.data.transforms.transforms as T

# import cv2


class DohDetection(Dataset):
    def __init__(self,
                 img_folder,
                 ann_file,
                 image_set,
                 filter_empty_gt=True,
                 transforms=None,
                 ):
        self.img_folder = img_folder
        self.file_meta = dict()
        self._transforms = transforms

        self.image_set = image_set
        self.inst_act_list = ['no_contact', 'self_contact', 'other_person_contact', 'portable_object_contact', 'stationary_object_contact'] #100doh the hand contact state(0~4)
        self.act_list = ['no_contact', 'self_contact', 'other_person_contact', 'portable_object_contact', 'stationary_object_contact'] #100doh the hand contact state(0~4)
        self.file_meta['action_classes'] = self.act_list
        self.doh_classes = ['N/A', 'hand', 'object']
        self.file_meta['doh_classes'] = self.doh_classes
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
    # >>> 1. instance & pair
    def load_instance_pair_annotations(self, image_index):
        cat_hand = self.doh_classes.index('hand')
        cat_obj = self.doh_classes.index('object')
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
                pass



        # numpy 配列に変換して2次元配列を作成
        inst_bbox = np.array(inst_bbox)
        inst_category = np.array(inst_category)
        inst_action = np.array(inst_action)
        pair_bbox = np.array(pair_bbox)
        pair_action = np.array(pair_action)
        pair_target = np.array(pair_target)
    
        return inst_bbox, inst_category, inst_action, pair_bbox, pair_action, pair_target


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
    # def vis_sample(self, idx):
    #     img_info = self.img_infos[idx]
    #     target = self.get_ann_info(idx)
    #     image = cv2.imread(os.path.join(self.img_folder, img_info))

    #     for bbox in target['boxes']:
    #         x_min, y_min, x_max, y_max = bbox
    #         # 整数に変換
    #         x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    #         # バウンディングボックスの描画
    #         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    #     basename = os.path.basename(img_info)
    #     img_path = os.path.join('100doh_vis_maeda/', 'vis_' + basename)
    #     # 画像を保存
    #     cv2.imwrite(img_path, image)



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
        # print("/large/maeda/HOTR/hotr/data/datasets/doh.py target: ", target)
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
        inst_bbox, inst_label, inst_actions, pair_bbox, pair_actions, pair_targets = self.load_instance_pair_annotations(img_idx)

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
        # print('sample', sample)
        return sample
    
    def get_image_info(self, idx):
        img_info = self.img_infos[idx]
        return img_info

    ############################################################################
    # Number Method
    ############################################################################
    def num_category(self):
        return len(self.doh_classes)
    
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
    print('root', root)
    
    assert root.exists(), f'provided Doh path {root} does not exist'
    PATHS = {
        "train": (root / "raw", root / "file" / 'trainval.json'),
        "val": (root / "raw", root / "file" / 'test.json'),
        "test": (root / "raw", root / "file" / 'test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    print('Annotation path is', ann_file)


    dataset = DohDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        image_set = image_set,
        filter_empty_gt=True,
        transforms = make_hoi_transforms(image_set)
        # transforms = make_simple_hoi_transforms(image_set)
    )
    dataset.file_meta['dataset_file'] = args.dataset_file

    dataset.file_meta['image_set'] = image_set

    return dataset



def main(image_set, args):
    #"image_set"はtrain, val, testのいずれか
    root = '100doh'
    # PATHS = {
    #     "train": (root + "/raw", root + '/file/trainval.json'),
    #     "val": (root + "/raw", root + '/file/test.json'),
    #     "test": (root + "/raw", root + '/file/test.json'),
    # }
    PATHS = {
        "train": (root + "/raw", root + '/file/check_trainval.json'),
        "val": (root + "/raw", root + '/file/check_test.json'),
        "test": (root + "/raw", root + '/file/check_test.json'),
    }



    img_folder, ann_file = PATHS[image_set]
    dataset = DohDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        image_set = image_set,
        filter_empty_gt=True,
        # transforms = make_hoi_transforms(image_set)
        transforms = make_simple_hoi_transforms(image_set)
    )
    # dataset.file_meta['dataset_file'] = args.dataset_file

    dataset.file_meta['image_set'] = image_set

    return dataset