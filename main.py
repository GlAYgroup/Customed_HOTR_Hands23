# ------------------------------------------------------------------------
# HOTR official code : main.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import hotr.data.datasets as datasets
import hotr.util.misc as utils
from hotr.engine.arg_parser import get_args_parser
from hotr.data.datasets import build_dataset, get_coco_api_from_dataset
from hotr.engine.trainer import train_one_epoch
from hotr.engine import hoi_evaluator, hoi_accumulator, hoi_visualizer
from hotr.models import build_model
import wandb

from hotr.util.logger import print_params, print_args

from torch.utils.data import Subset

import matplotlib.pyplot as plt

import maeda_loss_vis as loss_vis
import random

# 確認用(どの層がfreezeされているか確認)
def check_trainable_parameters(model):
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
    print("=== Trainable Parameters ===")
    for name in trainable:
        print(name)
    print("\n=== Frozen Parameters ===")
    for name in frozen:
        print(name)
    print(f"\nTotal Parameters: {len(trainable) + len(frozen)}")
    print(f"Trainable Parameters: {len(trainable)}")
    print(f"Frozen Parameters: {len(frozen)}")


def save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename):
    # save_ckpt: function for saving checkpoints
    output_dir = Path(args.output_dir)
    if filename == 'checkpoint':
        if args.output_dir:
            checkpoint_path = output_dir / f'checkpoints/{filename}{epoch:04d}.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
    else:
        if args.output_dir:
            checkpoint_path = output_dir / f'{filename}.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

def save_log(args, dataset_train, dataset_val, name):
    output_dir = Path(args.output_dir)
    log_file_path = output_dir / name

   
    # プログラムの開始時にenvironment.txtの内容をクリア
    with log_file_path.open("w") as f:
        pass  ## ファイルを開いてすぐに閉じることで内容をクリア

    if args.task == 'AOD':
        log_stats = {"lr":args.lr,
                    "epochs":args.epochs,
                    "len_train":len(dataset_train),
                    "len_val":len(dataset_val),
                    "num_hoi_queries":args.num_hoi_queries,
                    "set_cost_idx":args.set_cost_idx,
                    "set_cost_act":args.set_cost_act,
                    "hoi_idx_loss_coef":args.hoi_idx_loss_coef,
                    "hoi_act_loss_coef":args.hoi_act_loss_coef,
                    "hoi_eos_coef":args.hoi_eos_coef,
                    **vars(args)}
    elif args.task == 'ASOD':
        log_stats = {"lr":args.lr,
                    "epochs":args.epochs,
                    "len_train":len(dataset_train),
                    "len_val":len(dataset_val),
                    "num_hoi_queries":args.num_hoi_queries,
                    "set_cost_idx":args.set_cost_idx,
                    "set_cost_soidx":args.set_cost_soidx,
                    "set_cost_act":args.set_cost_act,
                    "hoi_idx_loss_coef":args.hoi_idx_loss_coef,
                    "hoi_soidx_loss_coef":args.hoi_soidx_loss_coef,
                    "hoi_act_loss_coef":args.hoi_act_loss_coef,
                    "hoi_eos_coef":args.hoi_eos_coef,
                    **vars(args)}
    # 再帰的にシリアライズ
    log_stats_serializable = convert_to_serializable(log_stats)

    with log_file_path.open("a") as f:
        f.write(json.dumps(log_stats_serializable, indent=4) + "\n")


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(args)

    # Data Setup
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val' if not args.eval else 'test', args=args)   

    assert dataset_train.num_action() == dataset_val.num_action(), "Number of actions should be the same between splits"
    args.num_classes = dataset_train.num_category()
    print(f"Number of actions: {args.num_classes}")
    args.num_actions = dataset_train.num_action()
    args.action_names = dataset_train.get_actions()
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    if args.dataset_file == 'vcoco':
        # Save V-COCO dataset statistics
        args.valid_ids = np.array(dataset_train.get_object_label_idx()).nonzero()[0]
        args.invalid_ids = np.argwhere(np.array(dataset_train.get_object_label_idx()) == 0).squeeze(1)       
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
    elif args.dataset_file == 'hico-det':
        args.valid_obj_ids = dataset_train.get_valid_obj_ids()
    elif args.dataset_file == 'doh':
        print("----DOH Dataset----")
        args.valid_ids = np.array([i for i in range(len(dataset_train.file_meta['action_classes']))])
        args.invalid_ids = np.argwhere(0) # 空の配列を作成
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
        # assert args.dataset_file != 'doh'
    elif args.dataset_file == 'hands23':
        print("----Hands23 Dataset----")
        args.valid_ids = np.array([i for i in range(len(dataset_train.file_meta['action_classes']))])
        args.invalid_ids = np.argwhere(0) # 空の配列を作成
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
 


    # Change dataset to small dataset if you want to check something
    if args.check == True:
        print("----Check Mode----")            
        # 最初のn個のインデックスを選択
        if args.check_num_images == -1:
            print("Images of train dataset: ", len(dataset_train))
            print("Images of val dataset: ", len(dataset_val))
        else:
            num_train = len(dataset_train)
            num_val = len(dataset_val)

            # 要求された画像数がデータセットのサイズを超えないように調整
            if args.check_num_images > num_train:
                print(f"要求されたトレーニング画像数 {args.check_num_images} は、データセットの総数 {num_train} を超えています。")
                args.check_num_images = num_train

            if args.check_num_images > num_val:
                print(f"要求されたバリデーション画像数 {args.check_num_images} は、データセットの総数 {num_val} を超えています。")
                args.check_num_images = num_val

            # ランダムにインデックスを選択
            indices_train = random.sample(range(num_train), args.check_num_images)
            indices_val = random.sample(range(num_val), args.check_num_images)

            print("Images of train dataset: ", len(indices_train))
            print("Images of val dataset: ", len(indices_val))

            # サブセットを作成
            dataset_train = Subset(dataset_train, indices_train)
            dataset_val = Subset(dataset_val, indices_val)
    else:
        print("----Normal Mode----")
        print("Images of train dataset: ", len(dataset_train))
        print("Images of val dataset: ", len(dataset_val))
    

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        # sampler_train = DistributedSampler(dataset_train, shuffle=False)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

        sampler_train_for_val = DistributedSampler(dataset_train, shuffle=False)


    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        sampler_train_for_val = torch.utils.data.SequentialSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train_for_val = DataLoader(dataset_train, args.batch_size, sampler=sampler_train_for_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        print("Distributed training enabled")
    n_parameters = print_params(model)

    # モデルのパラメータがtrainされるかfreezeされるか確認
    # check_trainable_parameters(model_without_ddp)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # print("model.state_dict().keys()", model_without_ddp.state_dict().keys())
    
    
    # # モデルの状態辞書からキーとパラメータサイズを取得
    # state_dict_info = {}
    # for key, value in model_without_ddp.state_dict().items():
    #     # パラメータの形状（サイズ）を取得
    #     param_size = list(value.size())
    #     # キーとサイズを辞書に追加
    #     state_dict_info[key] = param_size

    # # JSONファイルに保存
    # with open('HOTR_state_dict_info.json', 'w') as json_file:
    #     json.dump(state_dict_info, json_file, indent=4)

    # print("各レイヤーのパラメータサイズが 'state_dict_info.json' に保存されました。")

    # for name, p in model_without_ddp.named_parameters():
    #     print(name, p.requires_grad)
    # print("param_dicts", param_dicts)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Weight Setup
    if args.frozen_weights is not None:
        if args.frozen_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.frozen_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')      

        # you can choose 1 or 2 code
            
        ## 1
        # layer nameとその重みのshapeが合うweightsをmodelにロードする            
        # checkpoint_dict = checkpoint['model']
        # renamed_checkpoint_dict = {'detr.' + k: v for k, v in checkpoint_dict.items()}
        # # set weights defined in the model with the same shape
        # model_dict = {name: params for name, params in model.state_dict().items()}
        # new_checkpoint_dict = {}
        # for k, v in renamed_checkpoint_dict.items():
        #     if k in model_dict.keys() and v.shape == model_dict[k].shape:
        #         new_checkpoint_dict[k] = v
        # model_without_ddp.load_state_dict(new_checkpoint_dict, strict=False) # load weights, strict=Falseでnew_checkpoint_dict内に存在するレイヤーのみロードする
        
        ## 2
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
        # print(checkpoint['model'].keys())

        # check if the weights are loaded correctly
        """        
        check_new_checkpoint_dict = {k: v.shape for k, v in new_checkpoint_dict.items()}
        check_renamed_checkpoint_dict = {k: v.shape for k, v in renamed_checkpoint_dict.items()}
        check_model_dict = {k: v.shape for k, v in model_dict.items()}
        # print('new_checkpoint_dict', check_new_checkpoint_dict)
        # print('renamed_checkpoint_dict', check_renamed_checkpoint_dict)
        # print('model_dict', check_model_dict)
        
        with open('maeda/check_params.json', 'w') as f:
            json.dump(check_new_checkpoint_dict, f)
            json.dump(check_renamed_checkpoint_dict, f)
            json.dump(check_model_dict, f)
        """
        


    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])


    # Evaluation
    if args.eval:
        print("----test mode----")
        # test only mode
        if args.HOIDet:
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                vis = hoi_visualizer(args, total_res)

                sc1, sc2 = hoi_accumulator(args, total_res, True, False)
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f}')
                print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f}')
                print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f}')
            elif args.dataset_file == 'doh':
                #Inference
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)

                #Visualization
                if args.vis:
                    vis = hoi_visualizer(args, total_res, dataset_val)

                #Evaluation
                val = hoi_accumulator(args, total_res, True, False)
                print(f'| AP (0.75)\t\t: {val[0.75]:.5f}')
                print(f'| AP (0.50)\t\t: {val[0.5]:.5f}')
                print(f'| AP (0.25)\t\t: {val[0.25]:.5f}')

            elif args.dataset_file == 'hands23':
                #Inference
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)

                #Visualization
                if args.vis:
                    vis = hoi_visualizer(args, total_res, dataset_val)

                #Evaluation
                ho_val, hos_val, hand_val, active_obj_val, second_obj_val = hoi_accumulator(args, total_res, True, False)
                print('----[Hand, Active-object] AP----')
                print(f'| AP (0.75)\t\t: {ho_val[0.75]:.5f}')
                print(f'| AP (0.50)\t\t: {ho_val[0.5]:.5f}')
                print(f'| AP (0.25)\t\t: {ho_val[0.25]:.5f}')
                print('----[Hand, Active-object, Second-object] AP----')
                print(f'| AP (0.75)\t\t: {hos_val[0.75]:.5f}')
                print(f'| AP (0.50)\t\t: {hos_val[0.5]:.5f}')
                print(f'| AP (0.25)\t\t: {hos_val[0.25]:.5f}')
                print('----[Hand] AP----')
                print(f'| AP (0.75)\t\t: {hand_val[0.75]:.5f}')
                print(f'| AP (0.50)\t\t: {hand_val[0.5]:.5f}')
                print(f'| AP (0.25)\t\t: {hand_val[0.25]:.5f}')
                print('----[Active-object] AP----')
                print(f'| AP (0.75)\t\t: {active_obj_val[0.75]:.5f}')
                print(f'| AP (0.50)\t\t: {active_obj_val[0.5]:.5f}')
                print(f'| AP (0.25)\t\t: {active_obj_val[0.25]:.5f}')
                print('----[Second-object] AP----')
                print(f'| AP (0.75)\t\t: {second_obj_val[0.75]:.5f}')
                print(f'| AP (0.50)\t\t: {second_obj_val[0.5]:.5f}')
                print(f'| AP (0.25)\t\t: {second_obj_val[0.25]:.5f}')

                test_log_stats = {
                                'AOD_three_quaters_AP': ho_val[0.75],
                                'AOD_half_AP': ho_val[0.5],
                                'AOD_quarter_AP': ho_val[0.25],
                                'ASOD_three_quaters_AP': hos_val[0.75],
                                'ASOD_half_AP': hos_val[0.5],
                                'ASOD_quarter_AP': hos_val[0.25],
                                'Hand_three_quaters_AP': hand_val[0.75],
                                'Hand_half_AP': hand_val[0.5],
                                'Hand_quarter_AP': hand_val[0.25],
                                'Active_object_three_quaters_AP': active_obj_val[0.75],
                                'Active_object_half_AP': active_obj_val[0.5],
                                'Active_object_quarter_AP': active_obj_val[0.25],
                                'Second_object_three_quaters_AP': second_obj_val[0.75],
                                'Second_object_half_AP': second_obj_val[0.5],
                                'Second_object_quarter_AP': second_obj_val[0.25]
                                }
                fixed_test_log_stats = {
                                'AOD__0.75_AP': f'{ho_val[0.75]:.5f}',
                                'AOD__0.50_AP': f'{ho_val[0.5]:.5f}',
                                'AOD__0.25_AP': f'{ho_val[0.25]:.5f}',
                                'ASOD_0.75_AP': f'{hos_val[0.75]:.5f}',
                                'ASOD_0.50_AP': f'{hos_val[0.5]:.5f}',
                                'ASOD_0.25_AP': f'{hos_val[0.25]:.5f}',
                                'Hand_0.75_AP': f'{hand_val[0.75]:.5f}',
                                'Hand_0.50_AP': f'{hand_val[0.5]:.5f}',
                                'Hand_0.25_AP': f'{hand_val[0.25]:.5f}',
                                'AObj_0.75_AP': f'{active_obj_val[0.75]:.5f}',
                                'AObj_0.50_AP': f'{active_obj_val[0.5]:.5f}',
                                'AObj_0.25_AP': f'{active_obj_val[0.25]:.5f}',
                                'SObj_0.75_AP': f'{second_obj_val[0.75]:.5f}',
                                'SObj_0.50_AP': f'{second_obj_val[0.5]:.5f}',
                                'SObj_0.25_AP': f'{second_obj_val[0.25]:.5f}'
                                }

                
                with(Path(args.output_dir) / "test_log.txt").open("w") as f:
                    f.write(json.dumps(test_log_stats, indent=4))
                with(Path(args.output_dir) / "test_log_fixed.txt").open("w") as f:
                    f.write(json.dumps(fixed_test_log_stats, indent=4))

          
                                

            else: raise ValueError(f'dataset {args.dataset_file} is not supported.')
            return
        else:
            test_stats, coco_evaluator = evaluate_coco(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return

    # stats
    scenario1, scenario2 = 0, 0
    best_mAP, best_rare, best_non_rare = 0, 0, 0
    quarter_mAP, half_mAP, three_quaters_mAP = 0, 0, 0
    if args.task == 'ASOD':
        quarter_so_mAP, half_so_mAP, three_quaters_so_mAP = 0, 0, 0
        

    # add argparse
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            name=args.run_name,
            config=args
        )
        wandb.watch(model)

    # Training starts here!
    start_time = time.time()

    # ログファイルのパスを設定
    output_dir = Path(args.output_dir)
    log_file_path = output_dir / "log.txt"
    # プログラムの開始時にlog.txtの内容をクリア
    with log_file_path.open("w") as f:
        pass  # ファイルを開いてすぐに閉じることで内容をクリア

    # ログファイルに環境情報を保存(train時にのみ実行してほしいのでここに書く)
    save_log(args, dataset_train, dataset_val, "environment.txt")
    Path(args.output_dir + "/checkpoints").mkdir(parents=True, exist_ok=True)


    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.epochs,
            args.clip_max_norm, dataset_file=args.dataset_file, log=args.wandb)
        lr_scheduler.step()

        # Clear CUDA cache after each epoch
        torch.cuda.empty_cache()

        # Validation
        if args.validate:
            print('-'*100)
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)

                if utils.get_rank() == 0:
                    sc1, sc2 = hoi_accumulator(args, total_res, False, args.wandb)
                    if sc1 > scenario1:
                        scenario1 = sc1
                        scenario2 = sc2
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| Scenario #1 mAP : {sc1:.2f} ({scenario1:.2f})')
                    print(f'| Scenario #2 mAP : {sc2:.2f} ({scenario2:.2f})')
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    if test_stats['mAP'] > best_mAP:
                        best_mAP = test_stats['mAP']
                        best_rare = test_stats['mAP rare']
                        best_non_rare = test_stats['mAP non-rare']
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f} ({best_mAP:.2f})')
                    print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f} ({best_rare:.2f})')
                    print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f} ({best_non_rare:.2f})')
                    if args.wandb and utils.get_rank() == 0:
                        wandb.log({
                            'mAP': test_stats['mAP']
                        })
            elif args.dataset_file == 'doh':
                #Inference
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    #Evaluation
                    val = hoi_accumulator(args, total_res, True, False)
                    if val[0.5] > half_mAP:
                        three_quaters_mAP = val[0.75]
                        half_mAP = val[0.5]
                        quarter_mAP = val[0.25]
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| mAP (0.75)\t\t: {val[0.75]:.5f} ({three_quaters_mAP:.5f})')
                    print(f'| mAP (0.50)\t\t: {val[0.5]:.5f} ({half_mAP:.5f})')
                    print(f'| mAP (0.25)\t\t: {val[0.25]:.5f} ({quarter_mAP:.5f})')
            
            elif args.dataset_file == 'hands23':
                #Inference
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    #Evaluation
                    val_ho, val_hos, val_hand, val_active_obj, val_second_obj = hoi_accumulator(args, total_res, True, False)
                    if val_ho[0.5] > half_mAP:
                        three_quaters_mAP = val_ho[0.75]
                        half_mAP = val_ho[0.5]
                        quarter_mAP = val_ho[0.25]
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best_active')
                    if val_hos[0.5] > half_so_mAP:
                        three_quaters_so_mAP = val_hos[0.75]
                        half_so_mAP = val_hos[0.5]
                        quarter_so_mAP = val_hos[0.25]
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best_second')
                    print('----[Hand, Active-object] AP----')
                    print(f'| AP (0.75)\t\t: {val_ho[0.75]:.5f} ({three_quaters_mAP:.5f})')
                    print(f'| AP (0.50)\t\t: {val_ho[0.5]:.5f} ({half_mAP:.5f})')
                    print(f'| AP (0.25)\t\t: {val_ho[0.25]:.5f} ({quarter_mAP:.5f})')
                    print('----[Hand, Active-object, Second-object] AP----')
                    print(f'| AP (0.75)\t\t: {val_hos[0.75]:.5f} ({three_quaters_so_mAP:.5f})')
                    print(f'| AP (0.50)\t\t: {val_hos[0.5]:.5f} ({half_so_mAP:.5f})')
                    print(f'| AP (0.25)\t\t: {val_hos[0.25]:.5f} ({quarter_so_mAP:.5f})')
                    print('----[Hand] AP----')
                    print(f'| AP (0.75)\t\t: {val_hand[0.75]:.5f}')
                    print(f'| AP (0.50)\t\t: {val_hand[0.5]:.5f}')
                    print(f'| AP (0.25)\t\t: {val_hand[0.25]:.5f}')
                    print('----[Active-object] AP----')
                    print(f'| AP (0.75)\t\t: {val_active_obj[0.75]:.5f}')
                    print(f'| AP (0.50)\t\t: {val_active_obj[0.5]:.5f}')
                    print(f'| AP (0.25)\t\t: {val_active_obj[0.25]:.5f}')
                    print('----[Second-object] AP----')
                    print(f'| AP (0.75)\t\t: {val_second_obj[0.75]:.5f}')
                    print(f'| AP (0.50)\t\t: {val_second_obj[0.5]:.5f}')
                    print(f'| AP (0.25)\t\t: {val_second_obj[0.25]:.5f}')

                    epoch_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                    #  **{f'val_{k}': v for k, v in test_stats.items()},
                                    'epoch': epoch,
                                    'n_parameters': n_parameters,
                                    'AOD_three_quaters_mAP': val_ho[0.75],
                                    'AOD_half_mAP': val_ho[0.5],
                                    'AOD_quarter_mAP': val_ho[0.25],
                                    'ASOD_three_quaters_mAP': val_hos[0.75],
                                    'ASOD_half_mAP': val_hos[0.5],
                                    'ASOD_quarter_mAP': val_hos[0.25],
                                    'Hand_three_quaters_mAP': val_hand[0.75],
                                    'Hand_half_mAP': val_hand[0.5],
                                    'Hand_quarter_mAP': val_hand[0.25],
                                    'Active_object_three_quaters_mAP': val_active_obj[0.75],
                                    'Active_object_half_mAP': val_active_obj[0.5],
                                    'Active_object_quarter_mAP': val_active_obj[0.25],
                                    'Second_object_three_quaters_mAP': val_second_obj[0.75],
                                    'Second_object_half_mAP': val_second_obj[0.5],
                                    'Second_object_quarter_mAP': val_second_obj[0.25]
                                    }

            print('-'*100)
        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='checkpoint')

        if args.dataset_file != 'hands23':
            epoch_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        #  **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters
                        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(epoch_log_stats) + "\n")
        
        # Clear CUDA cache after validation
        torch.cuda.empty_cache()
      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.dataset_file == 'vcoco':
        print(f'| Scenario #1 mAP : {scenario1:.2f}')
        print(f'| Scenario #2 mAP : {scenario2:.2f}')
    elif args.dataset_file == 'hico-det':
        print(f'| mAP (full)\t\t: {best_mAP:.2f}')
        print(f'| mAP (rare)\t\t: {best_rare:.2f}')
        print(f'| mAP (non-rare)\t: {best_non_rare:.2f}')
    elif args.dataset_file == 'doh':
        print(f'| mAP (0.75)\t\t: {three_quaters_mAP:.5f}')
        print(f'| mAP (0.50)\t\t: {half_mAP:.5f}')
        print(f'| mAP (0.25)\t\t: {quarter_mAP:.5f}')
        pass


    if args.check == True:
       log_file_path = 'checkpoints/check/' + args.dataset_file + '/' + args.group_name + '/' + args.run_name
       loss_vis.main(log_file_path)

    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)