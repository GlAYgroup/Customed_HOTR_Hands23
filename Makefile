#Hands23
hands23_multi_train:
	CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
		--nproc_per_node=2 \
		--use_env main.py \
		--task ASOD \
		--group_name SatoLab_HOTR \
		--run_name hands23_multi_hand_run_000001 \
		--batch_size 32 \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 300 \
		--num_hoi_queries 10 \
		--set_cost_idx 10 \
		--set_cost_soidx 10 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_soidx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hands23 \
		--frozen_weights resume/hands23/gpu_hands23_multi_hand_run_000001/checkpoint0299.pth \
		--data_path hands23 \
		--output_dir checkpoints/hands23/ \

#Hands23
hands23_multi_train_check:
	python -m torch.distributed.launch \
		--nproc_per_node=2 \
		--use_env main.py \
		--task ASOD \
		--group_name SatoLab_HOTR \
		--run_name hands23_multi_hand_run_000001 \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 300 \
		--num_hoi_queries 10 \
		--set_cost_idx 10 \
		--set_cost_soidx 10 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_soidx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hands23 \
		--frozen_weights resume/hands23/gpu_hands23_multi_hand_run_000001/checkpoint0299.pth \
		--data_path hands23 \
		--output_dir checkpoints/hands23/ \
		--check True \

hands23_single_train:
	python main.py \
		--task ASOD \
		--group_name SatoLab_HOTR \
		--run_name hands23_single_hand_run_000001 \
		--batch_size 2 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 3 \
		--num_hoi_queries 6 \
		--set_cost_idx 10 \
		--set_cost_soidx 10 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_soidx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hands23 \
		--frozen_weights resume/hands23/gpu_hands23_multi_hand_run_000001/checkpoint0299.pth \
		--data_path hands23  \
		--output_dir checkpoints/hands23 \

hands23_single_train_resume:
	python main.py \
		--task ASOD \
		--group_name SatoLab_HOTR \
		--run_name hands23_single_hand_run_000001_resume_11 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-5 \
		--epochs 100 \
		--num_hoi_queries 6 \
		--set_cost_idx 10 \
		--set_cost_soidx 10 \
		--set_cost_act 0 \
		--hoi_act_loss_coef 0.0 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hands23 \
		--data_path hands23  \
		--output_dir checkpoints/hands23/ \
		--start_epoch 10 \
		--resume checkpoints/hands23/SatoLab_HOTR/hands23_single_hand_run_000001/checkpoint.pth \

hands23_single_train_check:
	python main.py \
		--task ASOD \
		--group_name SatoLab_HOTR \
		--run_name hands23_single_hand_run_000009 \
		--batch_size 8 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 3 \
		--num_hoi_queries 6 \
		--set_cost_idx 10 \
		--set_cost_soidx 10 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_soidx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hands23 \
		--frozen_weights resume/hands23/gpu_hands23_multi_hand_run_000001/checkpoint0299.pth \
		--data_path hands23  \
		--output_dir checkpoints/check/hands23 \
		--check True \
	   
#frozen_weightsを変更する必要あり
hands23_single_test:
	python main.py \
		--task ASOD \
		--resume checkpoints/hands23/SatoLab_HOTR/hands23_single_hand_run_000001/checkpoint.pth \
		--run_name hands23_single_hand_run_000001 \
		--second_object_threshold 0.2 \
		--root hands23/hands23_data/allMergedBlur \
		--vis_mode unique_obj \
		--group_name SatoLab_HOTR \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 6 \
		--hand_threshold 0.6 \
		--object_threshold 0.2 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--vis \
		--dataset_file hands23 \
		--data_path hands23 \
		--output_dir checkpoints/hands23 \
		--check True \
		--check_num_images 30

# frozen_weightsを変更する必要あり
hands23_single_test_check:
	python main.py \
		--task ASOD \
		--resume checkpoints/check/hands23/SatoLab_HOTR/hands23_single_hand_run_000008/checkpoint.pth \
		--run_name hands23_single_hand_run_000008 \
		--root hands23/hands23_data/sub_allMergedBlur \
		--vis_mode unique_obj \
		--group_name SatoLab_HOTR \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 6 \
		--hand_threshold 0.7 \
		--object_threshold 0.5 \
		--second_object_threshold 0.3 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--vis \
		--dataset_file hands23 \
		--data_path hands23 \
		--output_dir checkpoints/check/hands23 \
		--check True \
		
# 100DOH
## [Doh] single-gpu train (runs in 1 GPU)
doh_single_train:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name doh_single_hand_run_000004 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 100 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file doh \
		--frozen_weights resume/doh/checkpoint0299.pth \
		--data_path 100doh  \
		--output_dir checkpoints/doh/ \


doh_single_train_resume:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name doh_single_hand_run_000001_resume_91 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-5 \
		--epochs 100 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file doh \
		--data_path 100doh  \
		--output_dir checkpoints/doh/ \
		--start_epoch 90 \
		--resume checkpoints/doh/SatoLab_HOTR/doh_single_hand_run_000001/checkpoint.pth \


doh_single_train_check:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name doh_single_hand_run_000003 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-3 \
		--epochs 10 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--set_cost_act 0 \
		--hoi_act_loss_coef 0 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file doh \
		--frozen_weights resume/doh/checkpoint0299.pth \
		--data_path 100doh  \
		--output_dir checkpoints/check/doh \
		--check True \
		--check_num_images 16
#--hoi_act_loss_coefはlossのactionの誤差の重みの係数
#--set_cost_act(hotr/models/hotr_matcher.pyの中で使用)はGTとoutputの間でペアを見つけるためのコスト関数内のactionの重みの係数


doh_single_test:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name doh_single_hand_run_000003 \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 6 \
		--object_threshold 0.5 \
		--hand_threshold 0.8 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--dataset_file doh \
		--data_path 100doh \
		--output_dir checkpoints/doh \
		--root 100doh/raw \
		--vis_mode unique_obj \
		--resume checkpoints/doh/SatoLab_HOTR/doh_single_hand_run_000003/checkpoint.pth
#		--vis 
# --resume checkpoints/check/doh/SatoLab_HOTR/doh_single_hand_run_000003/checkpoint.pth

doh_single_test_check:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name doh_single_hand_run_000001 \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 6 \
		--object_threshold 0.5 \
		--hand_threshold 0.6 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--vis \
		--dataset_file doh \
		--data_path 100doh \
		--check True \
		--check_num_images 10 \
		--output_dir checkpoints/check/doh \
		--root 100doh/raw \
		--vis_mode unique_obj \
		--resume checkpoints/doh/SatoLab_HOTR/doh_single_hand_run_000003/checkpoint.pth
		
# --resume checkpoints/check/doh/SatoLab_HOTR/doh_single_hand_run_000003/checkpoint.pth

# V-COCO
## [V-COCO] single-gpu train (runs in 1 GPU)
vcoco_single_train:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name vcoco_single_run_000002 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 100 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path v-coco  \
		--output_dir checkpoints/vcoco/

vcoco_single_train_check:
	python main.py \
		--group_name SatoLab_HOTR \
		--run_name vcoco_single_run_000002 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--epochs 100 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path v-coco  \
		--output_dir checkpoints/check/vcoco \
		--check True \
		--check_num_images 16



# [V-COCO] multi-gpu train (runs in 8 GPUs)
vcoco_multi_train:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--group_name SatoLab_HOTR \
		--run_name vcoco_multi_run_000001 \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--num_hoi_queries 16 \
		--set_cost_idx 10 \
		--set_cost_act 1 \
		--hoi_idx_loss_coef 1 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.05 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file vcoco \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path v-coco \
		--output_dir checkpoints/vcoco/


# single-gpu test (runs in 1 GPU)
vcoco_single_test_check:
	python main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--dataset_file vcoco \
		--data_path v-coco \
 		--resume checkpoints/check/vcoco/SatoLab_HOTR/vcoco_single_run_000001/checkpoint.pth \
		--check True \
		--check_num_images 16



###未編集



# [V-COCO] single-gpu test (runs in 1 GPU)
vcoco_single_test:
	python main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--dataset_file vcoco \
		--data_path v-coco \
 		--resume checkpoints/vcoco/vcoco_q16.pth
#		--resume checkpoints/vcoco/KakaoBrain_HOTR_vcoco/vcoco_single_run_000001/best.pth


# [V-COCO] multi-gpu test (runs in 8 GPUs)
vcoco_multi_test:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.05 \
		--no_aux_loss \
		--eval \
		--dataset_file vcoco \
		--data_path v-coco  \
		--resume checkpoints/vcoco/vcoco_q16.pth

# [HICO-DET] single-gpu train (runs in 1 GPU)
hico_single_train:
	python main.py \
		--group_name KakaoBrain_HOTR_hicodet \
		--run_name hicodet_single_run_000001 \
		--HOIDet \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--num_hoi_queries 16 \
		--set_cost_idx 20 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.2 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hico-det \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path hico_20160224_det \
		--output_dir checkpoints/hico_det/


# [HICO-DET] multi-gpu train (runs in 8 GPUs)
hico_multi_train:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--group_name KakaoBrain_HOTR_hicodet \
		--run_name hicodet_multi_run_000001 \
		--HOIDet \
		--wandb \
		--validate \
		--share_enc \
		--pretrained_dec \
		--lr 1e-4 \
		--lr_drop 80 \
		--epochs 100 \
		--num_hoi_queries 16 \
		--set_cost_idx 20 \
		--hoi_act_loss_coef 10 \
		--hoi_eos_coef 0.1 \
		--temperature 0.2 \
		--no_aux_loss \
		--hoi_aux_loss \
		--dataset_file hico-det \
		--frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
		--data_path hico_20160224_det \
		--output_dir checkpoints/hico_det/

## [HICO-DET] single-gpu test
hico_single_test:
	python main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.2 \
		--no_aux_loss \
		--eval \
		--dataset_file hico-det \
		--data_path hico_20160224_det \
		--resume checkpoints/hico_det/hico_q16.pth

# [HICO-DET] multi-gpu test (runs in 8 GPUs)
hico_multi_test:
	python -m torch.distributed.launch \
		--nproc_per_node=8 \
		--use_env main.py \
		--HOIDet \
		--share_enc \
		--pretrained_dec \
		--num_hoi_queries 16 \
		--object_threshold 0 \
		--temperature 0.2 \
		--no_aux_loss \
		--eval \
		--dataset_file hico-det \
		--data_path hico_20160224_det \
		--resume checkpoints/hico_det/hico_q16.pth
