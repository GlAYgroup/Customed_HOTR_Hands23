# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from hotr.util.box_ops import box_xyxy_to_cxcywh
from hotr.util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])
    max_size = torch.as_tensor([w, h], dtype=torch.float32)

    fields = ["labels", "area", "iscrowd"] # add additional fields
    if "inst_actions" in target.keys():
        fields.append("inst_actions")

    if "boxes" in target:
        boxes = target["boxes"]
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "pair_boxes" in target or ("sub_boxes" in target and "obj_boxes" in target):
        if "pair_boxes" in target:
            pair_boxes = target["pair_boxes"]
            num_boxes_per_pair = pair_boxes.shape[1] // 4  # 1ペアあたりのボックス数

            hboxes = pair_boxes[:, :4]
            oboxes = pair_boxes[:, 4:8]
            if num_boxes_per_pair == 3:
                soboxes = pair_boxes[:, 8:12]
            else:
                soboxes = None
        if ("sub_boxes" in target and "obj_boxes" in target):
            hboxes = target["sub_boxes"]
            oboxes = target["obj_boxes"]

        cropped_hboxes = hboxes - torch.as_tensor([j, i, j, i])
        cropped_hboxes = torch.min(cropped_hboxes.reshape(-1, 2, 2), max_size)
        cropped_hboxes = cropped_hboxes.clamp(min=0)
        hboxes = cropped_hboxes.reshape(-1, 4)

        obj_mask = (oboxes[:, 0] != -1)
        if obj_mask.sum() != 0:
            cropped_oboxes = oboxes[obj_mask] - torch.as_tensor([j, i, j, i])
            cropped_oboxes = torch.min(cropped_oboxes.reshape(-1, 2, 2), max_size)
            cropped_oboxes = cropped_oboxes.clamp(min=0)
            oboxes[obj_mask] = cropped_oboxes.reshape(-1, 4)
        else:
            cropped_oboxes = oboxes

        if soboxes is not None:
            sobj_mask = (soboxes[:, 0] != -1)
            if sobj_mask.sum() != 0:
                cropped_soboxes = soboxes[sobj_mask] - torch.as_tensor([j, i, j, i])
                cropped_soboxes = torch.min(cropped_soboxes.reshape(-1, 2, 2), max_size)
                cropped_soboxes = cropped_soboxes.clamp(min=0)
                soboxes[sobj_mask] = cropped_soboxes.reshape(-1, 4)
            else:
                cropped_soboxes = soboxes

        if soboxes is not None:
            cropped_pair_boxes = torch.cat([hboxes, oboxes, soboxes], dim=1)
        else:
            cropped_pair_boxes = torch.cat([hboxes, oboxes], dim=1)
        target["pair_boxes"] = cropped_pair_boxes
        pair_fields = ["pair_boxes", "pair_actions", "pair_targets", "triplet_targets"]

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes[?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "hand_bboxes" in target:
        hand_bboxes = target["hand_bboxes"]
        cropped_hand_bboxes = hand_bboxes - torch.as_tensor([j, i, j, i])
        cropped_hand_bboxes = torch.min(cropped_hand_bboxes.reshape(-1, 2, 2), max_size)
        cropped_hand_bboxes = cropped_hand_bboxes.clamp(min=0)
        target["hand_bboxes"] = cropped_hand_bboxes.reshape(-1, 4)

    if "hand_2d_key_points" in target:
        keypoints = target["hand_2d_key_points"]
        keypoints = keypoints - torch.as_tensor([j, i])
        target["hand_2d_key_points"] = keypoints

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            if field in target: # added this because there is no 'iscrowd' field in v-coco dataset
                target[field] = target[field][keep]

    # remove elements that have redundant area
    if "boxes" in target and "labels" in target:
        cropped_boxes = target['boxes']
        cropped_labels = target['labels']

        cnr, keep_idx = [], []
        for idx, (cropped_box, cropped_lbl) in enumerate(zip(cropped_boxes, cropped_labels)):
            if str((cropped_box, cropped_lbl)) not in cnr:
                cnr.append(str((cropped_box, cropped_lbl)))
                keep_idx.append(True)
            else: keep_idx.append(False)

        for field in fields:
            if field in target:
                target[field] = target[field][keep_idx]

    # remove elements for which pair boxes have zero area
    if "pair_boxes" in target:
        pair_boxes = target["pair_boxes"]
        num_boxes_per_pair = pair_boxes.shape[1] // 4  # 1ペアあたりのボックス数

        cropped_hboxes = pair_boxes[:, :4].reshape(-1, 2, 2)
        cropped_oboxes = pair_boxes[:, 4:8].reshape(-1, 2, 2)
        if num_boxes_per_pair == 3:
            cropped_soboxes = pair_boxes[:, 8:12].reshape(-1, 2, 2)
        else:
            cropped_soboxes = None


        keep_h = torch.all(cropped_hboxes[:, 1, :] > cropped_hboxes[:, 0, :], dim=1)
        keep_o = torch.all(cropped_oboxes[:, 1, :] > cropped_oboxes[:, 0, :], dim=1)
        if cropped_soboxes is not None:
            keep_so = torch.all(cropped_soboxes[:, 1, :] > cropped_soboxes[:, 0, :], dim=1)
        
        not_empty_o = torch.all(target["pair_boxes"][:, 4:8] >= 0, dim=1)
        discard_o = (~keep_o) & not_empty_o
        if (discard_o).sum() > 0:
            target["pair_boxes"][discard_o, 4:8] = -1
        
        if cropped_soboxes is not None:
            not_empty_so = torch.all(target["pair_boxes"][:, 8:12] >= 0, dim=1)
            discard_so = (~keep_so) & not_empty_so
            if (discard_so).sum() > 0:
                target["pair_boxes"][discard_so, 8:12] = -1

        for pair_field in pair_fields:
            target[pair_field] = target[pair_field][keep_h]

    if "hand_2d_key_points" in target and "hand_bboxes" in target and "pair_boxes" in target and "hand_kp_confidence" in target:
        hand_bboxes = target["hand_bboxes"]  # [num_hand_bboxes, 4]
        pair_hand_boxes = target["pair_boxes"][:, :4]  # [num_pairs, 4]

        # 手のバウンディングボックスが pair_boxes に存在するか確認
        # pair_boxes に同じ値が存在するかを確認（直接比較）
        # mask: hand_bboxes に存在する手が pair_hand_boxes に存在するか
        mask = torch.any(torch.all(pair_hand_boxes.unsqueeze(0) == hand_bboxes.unsqueeze(1), dim=2), dim=1)

        # 一致する hand_bboxes と hand_2d_key_points を保持
        target["hand_bboxes"] = hand_bboxes[mask]
        target["hand_2d_key_points"] = target["hand_2d_key_points"][mask]
        target["hand_kp_confidence"] = target["hand_kp_confidence"][mask]

       

    return cropped_image, target

def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "pair_boxes" in target:
        pair_boxes = target["pair_boxes"]
        num_boxes_per_pair = pair_boxes.shape[1] // 4  # ボックスの数（hand, object, second object）

        hboxes = pair_boxes[:, 0:4]
        oboxes = pair_boxes[:, 4:8]
        if num_boxes_per_pair == 3:
            soboxes = pair_boxes[:, 8:12]
        else:
            soboxes = None

        # human flip
        hboxes = hboxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])

        # Active-object flip
        obj_mask = (oboxes[:, 0] != -1)
        if obj_mask.sum() != 0:
            o_tmp = oboxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            oboxes[obj_mask] = o_tmp[obj_mask]

        # Second-object flip
        if soboxes is not None:
            sobj_mask = (soboxes[:, 0] != -1)
            if sobj_mask.sum() != 0:
                so_tmp = soboxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
                soboxes[sobj_mask] = so_tmp[sobj_mask]    
            
             # 反転後のボックスを結合
            pair_boxes = torch.cat([hboxes, oboxes, soboxes], dim=1)
        else:
            # セカンドオブジェクトがない場合
            pair_boxes = torch.cat([hboxes, oboxes], dim=1)

        target["pair_boxes"] = pair_boxes

    if "hand_bboxes" in target:
        hand_bboxes = target["hand_bboxes"]
        hand_bboxes = hand_bboxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["hand_bboxes"] = hand_bboxes

        # hand_2d_key_pointsの処理を追加
    if "hand_2d_key_points" in target:
        keypoints = target["hand_2d_key_points"].clone()
        # x座標を反転
        keypoints = keypoints * torch.as_tensor([-1, 1]) + torch.as_tensor([w, 0])
        target["hand_2d_key_points"] = keypoints

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "pair_boxes" in target:
        pair_boxes = target["pair_boxes"]
        num_boxes_per_pair = pair_boxes.shape[1] // 4  # 1ペアあたりのボックス数

        hboxes = pair_boxes[:, :4]
        if num_boxes_per_pair == 3:
            oboxes = pair_boxes[:, 4:8]
            soboxes = pair_boxes[:, 8:12]
        else:
            oboxes = pair_boxes[:, 4:8]
            soboxes = None

        # hand(human)のボックスをリサイズ
        scaled_hboxes = hboxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        hboxes = scaled_hboxes

        # Active-objectのボックスをリサイズ
        obj_mask = (oboxes[:, 0] != -1)
        if obj_mask.sum() != 0:
            scaled_oboxes = oboxes[obj_mask] * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            oboxes[obj_mask] = scaled_oboxes
        
        # Second-objectのボックスをリサイズ
        if soboxes is not None:
            sobj_mask = (soboxes[:, 0] != -1)
            if sobj_mask.sum() != 0:
                scaled_soboxes = soboxes[sobj_mask] * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
                soboxes[sobj_mask] = scaled_soboxes
        
        # ボックスの再結合
        if soboxes is not None:
            pair_boxes = torch.cat([hboxes, oboxes, soboxes], dim=1)
        else:
            pair_boxes = torch.cat([hboxes, oboxes], dim=1)

        target["pair_boxes"] = pair_boxes
        
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        
    if "hand_bboxes" in target:
        hand_bboxes = target["hand_bboxes"]
        hand_bboxes = hand_bboxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["hand_bboxes"] = hand_bboxes
        
    # hand_2d_key_pointsのリサイズ処理を追加
    if "hand_2d_key_points" in target:
        keypoints = target["hand_2d_key_points"]
        keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height])
        target["hand_2d_key_points"] = keypoints

    return rescaled_image, target


# hand keypointの処理を加えてないためとりあえずコメントアウト
# def pad(image, target, padding):
#     # assumes that we only pad on the bottom right corners
#     padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
#     if target is None:
#         return padded_image, None
#     target = target.copy()
#     # should we do something wrt the original size?
#     target["size"] = torch.tensor(padded_image[::-1])
#     if "masks" in target:
#         target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
#     return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

# hand keypointの処理を加えてないためとりあえずコメントアウト
# class CenterCrop(object):
#     def __init__(self, size):
#         self.size = size
# 
#     def __call__(self, img, target):
#         image_width, image_height = img.size
#         crop_height, crop_width = self.size
#         crop_top = int(round((image_height - crop_height) / 2.))
#         crop_left = int(round((image_width - crop_width) / 2.))
#         return crop(img, target, (crop_top, crop_left, crop_height, crop_width))

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


# hand keypointの処理を加えてないためとりあえずコメントアウト
# class RandomPad(object):
#     def __init__(self, max_pad):
#         self.max_pad = max_pad

#     def __call__(self, img, target):
#         pad_x = random.randint(0, self.max_pad)
#         pad_y = random.randint(0, self.max_pad)
#         return pad(img, target, (pad_x, pad_y))

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


# class RandomErasing(object):

#     def __init__(self, *args, **kwargs):
#         self.eraser = T.RandomErasing(*args, **kwargs)

#     def __call__(self, img, target):
#         return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if "pair_boxes" in target:
            pair_boxes = target["pair_boxes"]
            num_boxes_per_pair = pair_boxes.shape[1] // 4  # ボックスの数（hand, object, second object）

            hboxes = pair_boxes[:, :4]
            oboxes = pair_boxes[:, 4:8]
            if num_boxes_per_pair == 3:
                soboxes = pair_boxes[:, 8:12]
            else:
                soboxes = None

            # ハンドボックスの正規化
            hboxes = box_xyxy_to_cxcywh(hboxes)
            hboxes = hboxes / torch.tensor([w, h, w, h], dtype=torch.float32)

            # Active-objectボックスの正規化
            obj_mask = (oboxes[:, 0] != -1)
            if obj_mask.sum() != 0:
                oboxes[obj_mask] = box_xyxy_to_cxcywh(oboxes[obj_mask])
                oboxes[obj_mask] = oboxes[obj_mask] / torch.tensor([w, h, w, h], dtype=torch.float32)
            
            # Second-objectボックスの正規化
            if soboxes is not None:
                sobj_mask = (soboxes[:, 0] != -1)
                if sobj_mask.sum() != 0:
                    soboxes[sobj_mask] = box_xyxy_to_cxcywh(soboxes[sobj_mask])
                    soboxes[sobj_mask] = soboxes[sobj_mask] / torch.tensor([w, h, w, h], dtype=torch.float32)            
            
            # ボックスの再結合
            if soboxes is not None:
                pair_boxes = torch.cat([hboxes, oboxes, soboxes], dim=1)
            else:
                pair_boxes = torch.cat([hboxes, oboxes], dim=1)

            target["pair_boxes"] = pair_boxes

        # for hand pose
        if "hand_bboxes" in target:
            hand_bboxes = target["hand_bboxes"]
            hand_bboxes = box_xyxy_to_cxcywh(hand_bboxes)
            hand_bboxes = hand_bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["hand_bboxes"] = hand_bboxes
        if "hand_2d_key_points" in target:
            hand_pose = target["hand_2d_key_points"]
            hand_pose = hand_pose / torch.tensor([w, h], dtype=torch.float32)
            target["hand_2d_key_points"] = hand_pose

        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturatio=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturatio, hue)

    def __call__(self, img, target):
        return self.color_jitter(img), target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string