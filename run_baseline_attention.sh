#!/bin/bash
# Run xt+dx+xf transformer attention (mean-pool) baseline for all seeds and targets.
# This fills the missing non-bilinear baseline that run_sweep_capture24.sh skipped.
set -uo pipefail

GPUS="${GPUS:-6,7,8}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
TARGETS=(
  "_DA_HAR70plus_256_00 6 7"
  "_DA_WISDM_256_00 3 6"
  "_DA_WISDM2_256_00 3 6"
  "_DA_USC_HAD_256_00 6 12"
  "_DA_Opportunity_256_00 113 18"
  "_DA_Skoda_256_00 60 11"
)
EPOCHS_PRETRAIN=2
EPOCHS_FINETUNE=10
job=0

_gpu() { echo "${GPU_LIST[$(( job % ${#GPU_LIST[@]} ))]}"; }

for SEED in "${SEEDS[@]}"; do
  PT_TAG="_DA_capture24_256_00_v2dx_v3xf_ep${EPOCHS_PRETRAIN}_${SEED}"
  PT_OUT="out_pretrain/_DA_capture24_256_00/${PT_TAG}"
  GPU="$(_gpu)"

  if [ ! -f "$PT_OUT" ]; then
    echo "[pretrain seed=$SEED GPU=$GPU]"
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TQDM_DISABLE=1 \
    python -u run_pretrain.py \
      --data_name _DA_capture24_256_00 \
      --num_feature 3 --num_target 1 \
      --view2 dx --view3 xf \
      --encoder_type transformer \
      --batch_size_pretrain 256 \
      --epochs_pretrain $EPOCHS_PRETRAIN \
      --seed $SEED > /dev/null 2>&1
  fi
  job=$(( job + 1 ))

  for TARGET_SPEC in "${TARGETS[@]}"; do
    read FT_DATA FT_FEAT FT_CLS <<< "$TARGET_SPEC"
    PT_CKPT="model_pretrain/_DA_capture24_256_00/${PT_TAG}.pth"
    FT_OUT="out_finetune/${FT_DATA}/${FT_DATA}_pt-${PT_TAG}_hidden_ALL_0.0_0_finetune"

    if [ ! -f "$PT_CKPT" ]; then
      echo "  Skip finetune $FT_DATA seed=$SEED — no checkpoint"
      continue
    fi
    if [ -f "$FT_OUT" ]; then
      echo "  Skip (exists): $FT_DATA seed=$SEED"
      continue
    fi

    GPU="$(_gpu)"
    PCA_ARGS=""
    [ "$FT_DATA" = "_DA_Opportunity_256_00" ] && PCA_ARGS="--pca_components 32"
    [ "$FT_DATA" = "_DA_Skoda_256_00"       ] && PCA_ARGS="--pca_components 4"

    echo "[finetune $FT_DATA seed=$SEED GPU=$GPU]"
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TQDM_DISABLE=1 \
    python run_finetune.py \
      --data_name "$FT_DATA" \
      --pretrain_data_name _DA_capture24_256_00 \
      --num_feature $FT_FEAT --num_target $FT_CLS \
      --view2 dx --view3 xf \
      --encoder_type transformer \
      --epochs_pretrain $EPOCHS_PRETRAIN \
      --epochs_finetune $EPOCHS_FINETUNE \
      --feature hidden --loss_type ALL --lam 0.0 \
      --run_modes finetune \
      --seed $SEED \
      $PCA_ARGS > /dev/null 2>&1
    job=$(( job + 1 ))
  done
done

echo "=== baseline attention runs done ==="
