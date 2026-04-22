#!/bin/bash
# =============================================================================
# Pretrain sweep script — MLP logsig encoder variant
# Uses run_pretrain_mlp_logsig.py (logsig views processed by per-timestep MLP
# instead of transformer). Checkpoints are saved with _mlp_logsig suffix.
# Usage: bash run_sweep_mlp_logsig.sh
# =============================================================================

# --- Dataset ------------------------------------------------------------------
DATA="_DA_SleepEEG_256_00"
NUM_FEATURE=1
NUM_TARGET=5

# --- Sweep axes ---------------------------------------------------------------
SEEDS=(0)                           # add more seeds: (0 1 2)
EPOCHS=(2)                          # add more epoch counts: (100 200)
VIEW2S=("dx" "logsig")              # second view per combination
VIEW3S=("xf" "xf")                  # third view per combination (must match length of VIEW2S)

# --- Misc ---------------------------------------------------------------------
PARALLEL=false                      # set to true to run jobs on separate GPUs in background
DEFAULT_BATCH_SIZE=64
LOGSIG_BATCH_SIZE=32
DISABLE_TQDM=0
SKIP_TAGS=()
# =============================================================================

mkdir -p logs

num_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
job_index=0

for EPOCHS_VAL in "${EPOCHS[@]}"; do
  for i in "${!VIEW2S[@]}"; do
    V2="${VIEW2S[$i]}"
    V3="${VIEW3S[$i]}"

    seed_list=("${SEEDS[@]}")

    for j in "${!seed_list[@]}"; do
      SEED="${seed_list[$j]}"
      TAG="${DATA}_v2${V2}_v3${V3}_ep${EPOCHS_VAL}_seed${SEED}_mlp_logsig"
      LOG="logs/${TAG}.log"

      should_skip=false
      for SKIP_TAG in "${SKIP_TAGS[@]}"; do
        if [ "$TAG" = "$SKIP_TAG" ]; then
          should_skip=true
          break
        fi
      done

      if [ "$should_skip" = true ]; then
        echo "Skipping: ${TAG}"
        job_index=$((job_index + 1))
        continue
      fi

      BATCH_SIZE=${DEFAULT_BATCH_SIZE}
      if [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; then
        BATCH_SIZE=${LOGSIG_BATCH_SIZE}
      fi

      CMD="python -u run_pretrain_mlp_logsig.py \
        --data_name ${DATA} \
        --num_feature ${NUM_FEATURE} \
        --num_target ${NUM_TARGET} \
        --view2 ${V2} \
        --view3 ${V3} \
        --batch_size_pretrain ${BATCH_SIZE} \
        --epochs_pretrain ${EPOCHS_VAL} \
        --seed ${SEED}"

      if [ "$PARALLEL" = true ]; then
        GPU_ID=$((job_index % num_gpus))
        echo "Launching [GPU ${GPU_ID}] (batch ${BATCH_SIZE}): ${TAG}"
        TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$GPU_ID $CMD | tee "$LOG" &
      else
        echo "Running (batch ${BATCH_SIZE}): ${TAG}"
        TQDM_DISABLE=${DISABLE_TQDM} $CMD | tee "$LOG"
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
