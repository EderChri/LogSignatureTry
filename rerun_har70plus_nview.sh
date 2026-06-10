#!/bin/bash
# Re-run the 180 HAR70plus nview finetune jobs that failed before the trainer.py shape-mismatch fix.
# All pretrain checkpoints exist; this only runs missing finetune output files.
set -uo pipefail
GPUS="${GPUS:-6,7,8}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
i=0

PRETRAIN_DATA="_DA_capture24_256_00"
FINETUNE_DATA="_DA_HAR70plus_256_00"

_launch() {
  local gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  ((i++)) || true
  local logf="$1"; shift
  echo "[GPU $gpu] $*"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$@" >> "$logf" 2>&1
}

for SEED in $(seq 0 9); do
  for ENC in transformer mlp_logsig; do
    ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"

    # --- stream ---
    LS=""
    PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep2_${SEED}${ENC_SUFFIX}${LS}_ilbilinear"
    FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_hidden_ALL_0.0_0_finetune"
    PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
    if [ ! -f "$FT_OUT" ]; then
      if [ ! -f "$PT_CKPT" ]; then
        echo "Skip (no pretrain ckpt): $PT_CKPT"; continue
      fi
      _launch "logs/rerun_nview_har70_s${SEED}_${ENC}_stream.log" \
        python run_finetune_nview.py \
          --data_name "${FINETUNE_DATA}" --pretrain_data_name "${PRETRAIN_DATA}" \
          --num_feature 6 --num_target 7 \
          --view2 logsig --encoder_type "${ENC}" \
          --logsig_mode stream --logsig_depth 2 \
          --epochs_pretrain 2 --epochs_finetune 10 \
          --feature hidden --lam 0.0 --run_modes finetune \
          --interaction_type bilinear --seed "${SEED}"
    fi

    # --- window / window_smooth ---
    for MODE in window window_smooth; do
      for WSIZ in 64 128; do
        for STRIDE in 1 7; do
          if [ "$MODE" = "window" ]; then
            LS="_win${WSIZ}"
          else
            LS="_tukey${WSIZ}"
          fi
          [ "$STRIDE" -gt 1 ] && LS="${LS}_s${STRIDE}"

          PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep2_${SEED}${ENC_SUFFIX}${LS}_ilbilinear"
          FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_hidden_ALL_0.0_0_finetune"
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"

          if [ ! -f "$FT_OUT" ]; then
            if [ ! -f "$PT_CKPT" ]; then
              echo "Skip (no pretrain ckpt): $PT_CKPT"; continue
            fi
            _launch "logs/rerun_nview_har70_s${SEED}_${ENC}${LS}.log" \
              python run_finetune_nview.py \
                --data_name "${FINETUNE_DATA}" --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature 6 --num_target 7 \
                --view2 logsig --encoder_type "${ENC}" \
                --logsig_mode "${MODE}" \
                --logsig_window_size "${WSIZ}" \
                --logsig_smoothing tukey \
                --logsig_stride "${STRIDE}" \
                --logsig_depth 2 \
                --epochs_pretrain 2 --epochs_finetune 10 \
                --feature hidden --lam 0.0 --run_modes finetune \
                --interaction_type bilinear --seed "${SEED}"
          fi
        done
      done
    done
  done
done

echo "=== HAR70plus nview rerun done ==="
