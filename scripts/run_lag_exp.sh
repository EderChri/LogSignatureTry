#!/bin/bash
# Lag experiment: view2=logsig, view3=xf, window_size=64, logsig_normalize
# Usage: bash scripts/run_lag_exp.sh <GPU> <SEED_START> <SEED_END>
#
# Pairs: SleepEEG->Epilepsy, HARTH->HAR70plus, capture24->WISDM
# Lags: 8 16 32
# Seeds: determined by args

set -euo pipefail
GPU=$1
SEED_START=$2
SEED_END=$3

cd "$(dirname "$0")/.."

PYTHON=/opt/conda/envs/myenv/bin/python
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

LAG_VALUES=(8 16 32)

# Each entry: "PRETRAIN_DATA PT_NF PT_NT FINETUNE_DATA FT_NF FT_NT"
PAIRS=(
    "_DA_SleepEEG_256_00 1 5 _DA_Epilepsy_256_00 1 2"
    "_DA_HARTH_256_00 6 12 _DA_HAR70plus_256_00 6 7"
    "_DA_capture24_256_00 3 1 _DA_WISDM_256_00 3 6"
)

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    for LAG in "${LAG_VALUES[@]}"; do
        for PAIR in "${PAIRS[@]}"; do
            read -r PT_DATA PT_NF PT_NT FT_DATA FT_NF FT_NT <<< "$PAIR"

            echo "========================================"
            echo "GPU=$GPU  seed=$SEED  lag=$LAG"
            echo "pretrain: $PT_DATA  finetune: $FT_DATA"
            echo "========================================"

            CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_pretrain.py \
                --data_name "$PT_DATA" \
                --num_feature "$PT_NF" --num_target "$PT_NT" \
                --view2 logsig --view3 xf \
                --logsig_mode window --logsig_window_size 64 \
                --logsig_normalize \
                --logsig_lag "$LAG" \
                --batch_size_pretrain 128 --epochs_pretrain 2 \
                --seed "$SEED"

            CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_finetune.py \
                --data_name "$FT_DATA" \
                --pretrain_data_name "$PT_DATA" \
                --num_feature "$FT_NF" --num_target "$FT_NT" \
                --view2 logsig --view3 xf \
                --logsig_mode window --logsig_window_size 64 \
                --logsig_normalize \
                --logsig_lag "$LAG" \
                --epochs_pretrain 2 --epochs_finetune 10 \
                --feature hidden --loss_type ALL --lam 0.0 \
                --seed "$SEED"
        done
    done
done

echo "Done: GPU=$GPU seeds $SEED_START-$SEED_END"
