#!/bin/bash
# =============================================================================
# Pretrain sweep script
# Usage: bash run_sweep.sh
#
# Configure the arrays below, then run. If SEEDS has only one entry it is
# reused for every combination. Jobs run sequentially by default; set
# PARALLEL=true to assign each job to a separate GPU via CUDA_VISIBLE_DEVICES.
# =============================================================================

# --- Dataset ------------------------------------------------------------------
DATA="_DA_SleepEEG_256_00"
NUM_FEATURE=1
NUM_TARGET=5

# --- Sweep axes ---------------------------------------------------------------
SEEDS=(0)                           # add more seeds: (0 1 2)
EPOCHS=(200)                        # add more epoch counts: (100 200)
VIEW2S=("dx" "logsig")              # second view per combination
VIEW3S=("xf" "xf")                  # third view per combination (must match length of VIEW2S)

# --- Misc ---------------------------------------------------------------------
PARALLEL=false                      # set to true to run jobs on separate GPUs in background
# =============================================================================

mkdir -p logs

num_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
job_index=0

for EPOCHS_VAL in "${EPOCHS[@]}"; do
  for i in "${!VIEW2S[@]}"; do
    V2="${VIEW2S[$i]}"
    V3="${VIEW3S[$i]}"

    # Use the single seed if only one was provided, otherwise iterate
    seed_list=("${SEEDS[@]}")

    for j in "${!seed_list[@]}"; do
      SEED="${seed_list[$j]}"
      TAG="${DATA}_v2${V2}_v3${V3}_ep${EPOCHS_VAL}_seed${SEED}"
      LOG="logs/${TAG}.log"

      CMD="python run_pretrain.py \
        --data_name ${DATA} \
        --num_feature ${NUM_FEATURE} \
        --num_target ${NUM_TARGET} \
        --view2 ${V2} \
        --view3 ${V3} \
        --epochs_pretrain ${EPOCHS_VAL} \
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

# Wait for all background jobs if running in parallel
if [ "$PARALLEL" = true ]; then
  echo "Waiting for all jobs to finish..."
  wait
  echo "All jobs done."
fi
