# ------------------------------------------------------------------------
# HOTR official code : engine/trainer.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import math
import torch
import sys
import hotr.util.misc as utils
import hotr.util.logger as loggers
from typing import Iterable
import wandb

import json
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_epoch: int, max_norm: float = 0, dataset_file: str = 'coco', log: bool = False):
    model.train()
    criterion.train()
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    space_fmt = str(len(str(max_epoch)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch+1, end_epoch=max_epoch, fill=space_fmt)
    print_freq = int(len(data_loader)/1)
    if len(data_loader) >= 1000:
        print_freq = int(len(data_loader)/2)

    #pair_target の作成：ここでは、物体ID（o_id）に基づいて、物体のカテゴリー（o_cat）を取得し、それを pair_target に追加します。
    #物体IDが -1（つまり背景）の場合は、物体カテゴリーを -1（背景）としています。

    print(f"\n>>> Epoch #{(epoch+1)}")
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print('samples', samples)
        # print('targets', targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        outputs = model(samples)
        # outputsを保存
        save_outputs(outputs, "maeda/outputs/hands23/check_train_output.json")
        
        loss_dict = criterion(outputs, targets, log)
        weight_dict = criterion.weight_dict
      
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if utils.get_rank() == 0 and log: wandb.log(loss_dict_reduced_scaled)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        if "obj_class_error" in loss_dict:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# outputsの中身を確認する関数
def print_tensor_sizes(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f"Key: {key}, Value is a dictionary:")
            # 辞書の中身を再帰的に探索
            print_tensor_sizes(value)
        elif isinstance(value, torch.Tensor):
            size = value.size()
            print(f"Key: {key}, Tensor size: {size}, Dimension count: {len(size)}")
        else:
            # 詳細な型情報を表示
            value_type = type(value).__name__
            print(f"Key: {key}, Value type: {value_type}")

def save_outputs(dictionary, file_name):
    serializable_dict = convert_to_serializable(dictionary)
    # 辞書をJSON形式でファイルに保存
    with open(file_name, 'w') as f:
        # 保存するファイル名
        json.dump(serializable_dict, f, indent=4)

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj