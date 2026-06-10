#!/bin/bash
set -uo pipefail
GPUS="${GPUS:-6,7,8}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
i=0
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 dx --view3 logsig --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type transformer --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 7 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 0 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 1 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 2 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 3 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 4 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 5 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 6 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 7 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 8 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode stream --logsig_window_size 0 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window_smooth --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 128 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
GPU="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_finetune.py --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 --num_feature 6 --num_target 7 --view2 logsig --view3 xf --encoder_type mlp_logsig --logsig_depth 2 --logsig_mode window --logsig_window_size 64 --logsig_smoothing tukey --logsig_smooth_param 0.5 --logsig_stride 1 --logsig_pool auto --epochs_pretrain 2 --epochs_finetune 10 --feature hidden --loss_type ALL --lam 0.0 --interaction_type bilinear --run_modes finetune --seed 9 > /dev/null 2>&1
i=$((i+1))
