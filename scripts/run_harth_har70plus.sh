#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

set -euo pipefail

python -m pip install --upgrade --no-deps --ignore-requires-python -r requirements-idun-torch-stack.txt

for f in \
  preprocessed_data/_DA_HARTH_256_00.pkl \
  preprocessed_data/_DA_HAR70plus_256_00.pkl
do
  if [ ! -f "$f" ]; then
    echo "Missing preprocessed file: $f"
    echo "Run preprocess.slurm first."
    exit 1
  fi
done

SEED="${SEED:-0}"
EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-1}"
EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-1}"
VIEW2="${VIEW2:-dx}"
VIEW3="${VIEW3:-xf}"
ENCODER_TYPE="${ENCODER_TYPE:-transformer}"
BATCH_PRETRAIN="${BATCH_PRETRAIN:-64}"
BATCH_FINETUNE="${BATCH_FINETUNE:-8}"
LOGSIG_DEPTH="${LOGSIG_DEPTH:-2}"
LOGSIG_MODE="${LOGSIG_MODE:-stream}"
LOGSIG_WINDOW_SIZE="${LOGSIG_WINDOW_SIZE:-32}"
LOGSIG_SMOOTHING="${LOGSIG_SMOOTHING:-tukey}"
LOGSIG_SMOOTH_PARAM="${LOGSIG_SMOOTH_PARAM:-0.5}"

COMMON_ARGS=(
  --view2 "$VIEW2"
  --view3 "$VIEW3"
  --encoder_type "$ENCODER_TYPE"
  --logsig_depth "$LOGSIG_DEPTH"
  --logsig_mode "$LOGSIG_MODE"
  --logsig_window_size "$LOGSIG_WINDOW_SIZE"
  --logsig_smoothing "$LOGSIG_SMOOTHING"
  --logsig_smooth_param "$LOGSIG_SMOOTH_PARAM"
  --epochs_pretrain "$EPOCHS_PRETRAIN"
  --seed "$SEED"
)

echo "Starting minimal HARTH -> HAR70plus pipeline"
echo "seed=$SEED pretrain_epochs=$EPOCHS_PRETRAIN finetune_epochs=$EPOCHS_FINETUNE"
echo "views=xt,$VIEW2,$VIEW3 encoder=$ENCODER_TYPE"

python -u scripts/run_pretrain.py \
  --data_name _DA_HARTH_256_00 \
  --num_feature 6 \
  --num_target 12 \
  --batch_size_pretrain "$BATCH_PRETRAIN" \
  "${COMMON_ARGS[@]}"

python -u scripts/run_finetune.py \
  --data_name _DA_HAR70plus_256_00 \
  --pretrain_data_name _DA_HARTH_256_00 \
  --num_feature 6 \
  --num_target 7 \
  --batch_size_finetune "$BATCH_FINETUNE" \
  --epochs_finetune "$EPOCHS_FINETUNE" \
  --feature hidden \
  --loss_type ALL \
  --lam 0.0 \
  --run_modes finetune \
  "${COMMON_ARGS[@]}"

python -u scripts/run_probe.py \
  --probe_type raw \
  --data_name _DA_HAR70plus_256_00 \
  --pretrain_data_name _DA_HARTH_256_00 \
  --num_feature 6 \
  --num_target 7 \
  --batch_size_finetune "$BATCH_FINETUNE" \
  --epochs_finetune "$EPOCHS_FINETUNE" \
  "${COMMON_ARGS[@]}"

if [ "$ENCODER_TYPE" = "transformer" ] && \
   [ "$LOGSIG_MODE" = "stream" ] && \
   [ "$LOGSIG_DEPTH" = "2" ]; then
  python -u scripts/run_probe.py \
    --probe_type pretrained \
    --data_name _DA_HAR70plus_256_00 \
    --pretrain_data_name _DA_HARTH_256_00 \
    --num_feature 6 \
    --num_target 7 \
    --batch_size_finetune "$BATCH_FINETUNE" \
    --epochs_finetune "$EPOCHS_FINETUNE" \
    "${COMMON_ARGS[@]}"
else
  echo "Skipping pretrained probe for non-default checkpoint naming."
fi

python -u scripts/aggregate_results.py

echo "Done."
echo "Pretrain summary: out_pretrain/_DA_HARTH_256_00/final_pretrain_summary.tsv"
echo "Finetune/probe summary: out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv"
echo "Aggregated summary: out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary_agg.tsv"
