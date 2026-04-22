#!/bin/bash
# =============================================================================
# Finetune sweep on Epilepsy — MLP logsig encoder variant
# Uses run_finetune_mlp_logsig.py and expects checkpoints produced by
# run_sweep_mlp_logsig.sh (i.e. with _mlp_logsig suffix).
# Usage: bash run_finetune_epilepsy_sweep_mlp_logsig.sh
# =============================================================================

# --- Source pretrained models -------------------------------------------------
PRETRAIN_DATA="_DA_SleepEEG_256_00"
SEEDS=(0)
PRETRAIN_EPOCHS=(2)
VIEW2S=("dx" "logsig")
VIEW3S=("xf" "xf")

# --- Finetune target ----------------------------------------------------------
FINETUNE_DATA="_DA_Epilepsy_256_00"
NUM_FEATURE=1
NUM_TARGET=2
EPOCHS_FINETUNE=10

# --- Optional finetune args ---------------------------------------------------
FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"
LOGSIG_DEPTH=2

# --- Misc ---------------------------------------------------------------------
PARALLEL=false

mkdir -p logs

num_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
job_index=0

for PRE_EPOCHS in "${PRETRAIN_EPOCHS[@]}"; do
  for i in "${!VIEW2S[@]}"; do
    V2="${VIEW2S[$i]}"
    V3="${VIEW3S[$i]}"

    for j in "${!SEEDS[@]}"; do
      SEED="${SEEDS[$j]}"
      PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${PRE_EPOCHS}_${SEED}_mlp_logsig"
      PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"

      if [ ! -f "$PT_CKPT" ]; then
        echo "Skipping missing checkpoint: ${PT_CKPT}"
        continue
      fi

      TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
      LOG="logs/${TAG}.log"

      CMD="python run_finetune_mlp_logsig.py \
        --data_name ${FINETUNE_DATA} \
        --pretrain_data_name ${PRETRAIN_DATA} \
        --num_feature ${NUM_FEATURE} \
        --num_target ${NUM_TARGET} \
        --view2 ${V2} \
        --view3 ${V3} \
        --logsig_depth ${LOGSIG_DEPTH} \
        --epochs_pretrain ${PRE_EPOCHS} \
        --epochs_finetune ${EPOCHS_FINETUNE} \
        --feature ${FEATURE} \
        --loss_type ${LOSS_TYPE} \
        --lam ${LAM} \
        --seed ${SEED}"

      if [ "$PARALLEL" = true ]; then
        GPU_ID=$((job_index % num_gpus))
        echo "Launching [GPU ${GPU_ID}]: ${TAG}"
        CUDA_VISIBLE_DEVICES=$GPU_ID $CMD 2>&1 | tee "$LOG" &
      else
        echo "Running: ${TAG}"
        $CMD 2>&1 | tee "$LOG"
      fi

      job_index=$((job_index + 1))
    done
  done
done

if [ "$PARALLEL" = true ]; then
  echo "Waiting for all jobs to finish..."
  wait
  echo "All jobs done."
fi
