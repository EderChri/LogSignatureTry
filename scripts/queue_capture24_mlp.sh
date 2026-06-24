#!/bin/bash
# capture24→wisdm + capture24→har70+ pretrain+finetune, mlp_logsig
# Pass seeds as argument: e.g. "0 1 2 3 4" or "5 6 7 8 9"
PROJ=/home/ceder/benchmarking/Multi-View_Contrastive_Learning
PYTHON=/opt/conda/envs/myenv/bin/python
export PYTHONPATH=$PROJ

SEEDS=${1:-"0 1 2 3 4 5 6 7 8 9"}
COMMON_LOGSIG="--view2 logsig --view3 xf --logsig_mode window --logsig_window_size 64 --logsig_normalize --logsig_lag 16 --encoder_type mlp_logsig"

for seed in $SEEDS; do
    # Pretrain on capture24 (shared across both finetune targets)
    $PYTHON $PROJ/scripts/run_pretrain.py \
        --data_name _DA_capture24_256_00 --num_feature 3 --num_target 1 \
        $COMMON_LOGSIG \
        --batch_size_pretrain 128 --epochs_pretrain 2 \
        --seed $seed

    # Finetune → WISDM
    $PYTHON $PROJ/scripts/run_finetune.py \
        --data_name _DA_WISDM_256_00 --pretrain_data_name _DA_capture24_256_00 \
        --num_feature 3 --num_target 6 \
        $COMMON_LOGSIG \
        --epochs_pretrain 2 --epochs_finetune 10 \
        --feature hidden --loss_type ALL --lam 0.0 \
        --seed $seed

    # Finetune → HAR70plus
    $PYTHON $PROJ/scripts/run_finetune.py \
        --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_capture24_256_00 \
        --num_feature 6 --num_target 7 \
        $COMMON_LOGSIG \
        --epochs_pretrain 2 --epochs_finetune 10 \
        --feature hidden --loss_type ALL --lam 0.0 \
        --seed $seed
done
