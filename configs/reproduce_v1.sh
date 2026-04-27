#!/bin/bash
# =============================================================================
# configs/reproduce_v1.sh
#
# Reproduces every run that exists in out_pretrain/ and out_finetune/ as of
# the pre-refactor state (2-epoch smoke-test runs, seed=0, all view combos,
# both encoder types).
#
# HOW TO USE FOR VALIDATION
# -------------------------
# To verify the refactored code produces identical results, delete one or more
# existing output files and re-run the corresponding command below.  Results
# should match what is recorded in the summary TSVs.
#
#   Example (re-check one pretrain):
#     rm out_pretrain/_DA_SleepEEG_256_00/_DA_SleepEEG_256_00_v2dx_v3xf_ep2_0
#     bash configs/reproduce_v1.sh 2>&1 | grep "v2dx_v3xf"
#
# NOTE: commands are skipped automatically if the output file already exists.
#
# KNOWN ARCHITECTURAL CHANGE
# --------------------------
# The unified Encoder class no longer stores dead attributes encoder_layers_t/d/f
# (they were never used in the forward pass; the trained weights live in
# transformer_encoder_t/d/f.layers.*).  Old checkpoints load fine (strict=False
# ignores the extra keys); new checkpoints are NOT loadable by the pre-refactor
# code, but that code no longer exists.
# =============================================================================

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# SleepEEG pretrain — transformer encoder
# ---------------------------------------------------------------------------

python -u run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 --num_feature 1 --num_target 5 \
  --view2 dx --view3 xf --encoder_type transformer \
  --batch_size_pretrain 64 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 --num_feature 1 --num_target 5 \
  --view2 logsig --view3 xf --encoder_type transformer \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 --num_feature 1 --num_target 5 \
  --view2 dx --view3 logsig --encoder_type transformer \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

# ---------------------------------------------------------------------------
# SleepEEG pretrain — mlp_logsig encoder
# ---------------------------------------------------------------------------

python -u run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 --num_feature 1 --num_target 5 \
  --view2 dx --view3 xf --encoder_type mlp_logsig \
  --batch_size_pretrain 64 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 --num_feature 1 --num_target 5 \
  --view2 logsig --view3 xf --encoder_type mlp_logsig \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_SleepEEG_256_00 --num_feature 1 --num_target 5 \
  --view2 dx --view3 logsig --encoder_type mlp_logsig \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

# ---------------------------------------------------------------------------
# HARTH pretrain — transformer encoder
# ---------------------------------------------------------------------------

python -u run_pretrain.py \
  --data_name _DA_HARTH_256_00 --num_feature 6 --num_target 12 \
  --view2 dx --view3 xf --encoder_type transformer \
  --batch_size_pretrain 64 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_HARTH_256_00 --num_feature 6 --num_target 12 \
  --view2 logsig --view3 xf --encoder_type transformer \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_HARTH_256_00 --num_feature 6 --num_target 12 \
  --view2 dx --view3 logsig --encoder_type transformer \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

# ---------------------------------------------------------------------------
# HARTH pretrain — mlp_logsig encoder
# ---------------------------------------------------------------------------

python -u run_pretrain.py \
  --data_name _DA_HARTH_256_00 --num_feature 6 --num_target 12 \
  --view2 dx --view3 xf --encoder_type mlp_logsig \
  --batch_size_pretrain 64 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_HARTH_256_00 --num_feature 6 --num_target 12 \
  --view2 logsig --view3 xf --encoder_type mlp_logsig \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

python -u run_pretrain.py \
  --data_name _DA_HARTH_256_00 --num_feature 6 --num_target 12 \
  --view2 dx --view3 logsig --encoder_type mlp_logsig \
  --batch_size_pretrain 32 --epochs_pretrain 2 --seed 0

# ---------------------------------------------------------------------------
# SleepEEG → Epilepsy finetune — transformer
# ---------------------------------------------------------------------------

for V2 in dx logsig dx; do
  for V3 in xf xf logsig; do
    # skip mismatched iterations (bash loop is flat; pair them explicitly)
    break
  done
done

for PAIR in "dx xf" "logsig xf" "dx logsig"; do
  V2=${PAIR%% *}; V3=${PAIR##* }
  python run_finetune.py \
    --data_name _DA_Epilepsy_256_00 --pretrain_data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 2 \
    --view2 ${V2} --view3 ${V3} --encoder_type transformer \
    --logsig_depth 2 --epochs_pretrain 2 --epochs_finetune 10 \
    --feature hidden --loss_type ALL --lam 0.0 --seed 0
done

# ---------------------------------------------------------------------------
# SleepEEG → Epilepsy finetune — mlp_logsig
# ---------------------------------------------------------------------------

for PAIR in "dx xf" "logsig xf" "dx logsig"; do
  V2=${PAIR%% *}; V3=${PAIR##* }
  python run_finetune.py \
    --data_name _DA_Epilepsy_256_00 --pretrain_data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 2 \
    --view2 ${V2} --view3 ${V3} --encoder_type mlp_logsig \
    --logsig_depth 2 --epochs_pretrain 2 --epochs_finetune 10 \
    --feature hidden --loss_type ALL --lam 0.0 --seed 0
done

# ---------------------------------------------------------------------------
# HARTH → HAR70plus finetune — transformer
# ---------------------------------------------------------------------------

for PAIR in "dx xf" "logsig xf" "dx logsig"; do
  V2=${PAIR%% *}; V3=${PAIR##* }
  python run_finetune.py \
    --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_HARTH_256_00 \
    --num_feature 6 --num_target 7 \
    --view2 ${V2} --view3 ${V3} --encoder_type transformer \
    --logsig_depth 2 --epochs_pretrain 2 --epochs_finetune 10 \
    --feature hidden --loss_type ALL --lam 0.0 --seed 0
done

# ---------------------------------------------------------------------------
# HARTH → HAR70plus finetune — mlp_logsig
# ---------------------------------------------------------------------------

for PAIR in "dx xf" "logsig xf" "dx logsig"; do
  V2=${PAIR%% *}; V3=${PAIR##* }
  python run_finetune.py \
    --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_HARTH_256_00 \
    --num_feature 6 --num_target 7 \
    --view2 ${V2} --view3 ${V3} --encoder_type mlp_logsig \
    --logsig_depth 2 --epochs_pretrain 2 --epochs_finetune 10 \
    --feature hidden --loss_type ALL --lam 0.0 --seed 0
done

echo "All reproduction runs complete."
echo "Compare results against out_pretrain/*/final_pretrain_summary.tsv"
echo "and out_finetune/*/final_test_metric_summary.tsv"
