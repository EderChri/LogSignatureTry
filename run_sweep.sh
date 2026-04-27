#!/bin/bash
# =============================================================================
# run_sweep.sh — unified pretrain + finetune + probe sweep for any dataset pair
#
# Usage:
#   bash run_sweep.sh [pair]          # run one pair defined in datasets.cfg
#   bash run_sweep.sh                 # run all pairs
#   bash run_sweep.sh sleepeeg_epilepsy
#   bash run_sweep.sh harth_har70plus
#
# Each pair runs three stages in order:
#   1. Pretrain  — all view combos × encoder types × logsig modes × window sizes
#   2. Finetune  — same axes (finetune/freeze/baseline)
#   3. Probe     — raw + pretrained-encoder probe (transformer only)
#
# Set SKIP_STAGES to omit stages, e.g. SKIP_STAGES="probe" bash run_sweep.sh
#
# Logsig mode notes:
#   stream         — running log-sig of [0,t]; no window size needed.
#   window         — sliding window, rectangular (constant) weighting.
#   window_smooth  — sliding window with Tukey or EMA weighting.
#
# Checkpoint names encode the mode so runs never overwrite each other:
#   stream         → no suffix   (backward-compatible with reproduce_v1.sh)
#   window/size W  → _winW       e.g. _win32
#   window_smooth  → _tukeyW or _emaW
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Common sweep parameters — edit here to change all pairs at once
# ---------------------------------------------------------------------------
SEEDS=(0)
EPOCHS_PRETRAIN=(2)
EPOCHS_FINETUNE=10
EPOCHS_PROBE=50
VIEW2S=("dx" "logsig" "dx")
VIEW3S=("xf"  "xf"   "logsig")
ENCODER_TYPES=("transformer" "mlp_logsig")

# Logsig mode sweep — add entries to include more modes / sizes.
# LOGSIG_MODES entries: "stream", "window", "window_smooth"
# LOGSIG_WINDOW_SIZES: ignored for "stream" (window size doesn't apply)
# LOGSIG_SMOOTHING: used only for "window_smooth" mode
LOGSIG_MODES=("stream" "window" "window_smooth")
LOGSIG_WINDOW_SIZES=(128, 256)
LOGSIG_SMOOTHING="tukey"
LOGSIG_DEPTH=2

FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"

PARALLEL=false
DEFAULT_BATCH=64
LOGSIG_BATCH=32
DISABLE_TQDM=0

SKIP_STAGES="${SKIP_STAGES:-}"   # space-separated: "pretrain", "finetune", "probe"
# ---------------------------------------------------------------------------

source datasets.cfg

ALL_PAIRS=("sleepeeg_epilepsy" "harth_har70plus")
REQUESTED="${1:-all}"
if [ "$REQUESTED" = "all" ]; then
  RUN_PAIRS=("${ALL_PAIRS[@]}")
else
  RUN_PAIRS=("$REQUESTED")
fi

mkdir -p logs
num_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
[ "$num_gpus" -eq 0 ] && num_gpus=1

_launch() {
  local tag="$1"; local log="$2"; shift 2; local cmd=("$@")
  if [ "$PARALLEL" = true ]; then
    local gid=$(( job_index % num_gpus ))
    echo "Launching [GPU ${gid}]: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" 2>&1 | tee "$log" &
  else
    echo "Running: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} "${cmd[@]}" 2>&1 | tee "$log"
  fi
  job_index=$(( job_index + 1 ))
}

_skip_stage() { [[ " ${SKIP_STAGES} " == *" $1 "* ]]; }

# Build the logsig suffix that matches _logsig_suffix() in run_pretrain/finetune.py.
# stream → ""  |  window W → "_winW"  |  window_smooth W → "_tukeyW" / "_emaW"
_lsig_suffix() {
  local mode="$1" wsiz="$2" smoothing="$3"
  if [ "$mode" = "stream" ]; then
    echo ""
  elif [ "$mode" = "window" ]; then
    echo "_win${wsiz}"
  else
    echo "_${smoothing}${wsiz}"
  fi
}

# ===========================================================================

for PAIR in "${RUN_PAIRS[@]}"; do
  # Load dataset-specific variables from datasets.cfg
  cfg_var="cfg_${PAIR}[@]"
  for assignment in "${!cfg_var}"; do
    eval "$assignment"
  done

  echo ""
  echo "======================================================================="
  echo " Pair: ${PAIR}"
  echo "   Pretrain: ${PRETRAIN_DATA}  (${PRETRAIN_NUM_FEATURE} ch, ${PRETRAIN_NUM_TARGET} cls)"
  echo "   Finetune: ${FINETUNE_DATA}  (${FINETUNE_NUM_FEATURE} ch, ${FINETUNE_NUM_TARGET} cls)"
  echo "======================================================================="

  job_index=0

  # -------------------------------------------------------------------------
  # Stage 1: Pretrain
  # -------------------------------------------------------------------------
  if ! _skip_stage pretrain; then
    echo "--- Stage: pretrain ---"
    for ENC in "${ENCODER_TYPES[@]}"; do
      ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
      for EP in "${EPOCHS_PRETRAIN[@]}"; do
        for i in "${!VIEW2S[@]}"; do
          V2="${VIEW2S[$i]}"; V3="${VIEW3S[$i]}"
          BATCH=$DEFAULT_BATCH
          { [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; } && BATCH=$LOGSIG_BATCH
          # Skip window/window_smooth modes when no view is logsig — they would
          # produce identical data to stream (dx and xf ignore logsig params).
          if [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; then
            ACTIVE_MODES=("${LOGSIG_MODES[@]}")
          else
            ACTIVE_MODES=("stream")
          fi
          for MODE in "${ACTIVE_MODES[@]}"; do
            if [ "$MODE" = "stream" ]; then
              WSIZES=(0)   # window size irrelevant for stream
            else
              WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
            fi
            for WSIZ in "${WSIZES[@]}"; do
              LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING")"
              for SEED in "${SEEDS[@]}"; do
                TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
                _launch "$TAG" "logs/${TAG}.log" \
                  python -u run_pretrain.py \
                    --data_name "${PRETRAIN_DATA}" \
                    --num_feature "${PRETRAIN_NUM_FEATURE}" \
                    --num_target  "${PRETRAIN_NUM_TARGET}" \
                    --view2 "${V2}" --view3 "${V3}" \
                    --encoder_type "${ENC}" \
                    --batch_size_pretrain "${BATCH}" \
                    --epochs_pretrain "${EP}" \
                    --logsig_depth "${LOGSIG_DEPTH}" \
                    --logsig_mode "${MODE}" \
                    --logsig_window_size "${WSIZ}" \
                    --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                    --seed "${SEED}"
              done
            done
          done
        done
      done
    done
    [ "$PARALLEL" = true ] && { wait; echo "Pretrain jobs done."; }
  fi

  # -------------------------------------------------------------------------
  # Stage 2: Finetune
  # -------------------------------------------------------------------------
  if ! _skip_stage finetune; then
    echo "--- Stage: finetune ---"
    for ENC in "${ENCODER_TYPES[@]}"; do
      ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
      for EP in "${EPOCHS_PRETRAIN[@]}"; do
        for i in "${!VIEW2S[@]}"; do
          V2="${VIEW2S[$i]}"; V3="${VIEW3S[$i]}"
          if [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; then
            ACTIVE_MODES=("${LOGSIG_MODES[@]}")
          else
            ACTIVE_MODES=("stream")
          fi
          for MODE in "${ACTIVE_MODES[@]}"; do
            if [ "$MODE" = "stream" ]; then
              WSIZES=(0)
            else
              WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
            fi
            for WSIZ in "${WSIZES[@]}"; do
              LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING")"
              for SEED in "${SEEDS[@]}"; do
                PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
                PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
                if [ ! -f "$PT_CKPT" ]; then
                  echo "Skipping missing checkpoint: ${PT_CKPT}"
                  continue
                fi
                TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
                _launch "$TAG" "logs/${TAG}.log" \
                  python run_finetune.py \
                    --data_name "${FINETUNE_DATA}" \
                    --pretrain_data_name "${PRETRAIN_DATA}" \
                    --num_feature "${FINETUNE_NUM_FEATURE}" \
                    --num_target  "${FINETUNE_NUM_TARGET}" \
                    --view2 "${V2}" --view3 "${V3}" \
                    --encoder_type "${ENC}" \
                    --logsig_depth "${LOGSIG_DEPTH}" \
                    --logsig_mode "${MODE}" \
                    --logsig_window_size "${WSIZ}" \
                    --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                    --epochs_pretrain "${EP}" \
                    --epochs_finetune "${EPOCHS_FINETUNE}" \
                    --feature "${FEATURE}" \
                    --loss_type "${LOSS_TYPE}" \
                    --lam "${LAM}" \
                    --seed "${SEED}"
              done
            done
          done
        done
      done
    done
    [ "$PARALLEL" = true ] && { wait; echo "Finetune jobs done."; }
  fi

  # -------------------------------------------------------------------------
  # Stage 3: Probe (transformer encoder only)
  # -------------------------------------------------------------------------
  if ! _skip_stage probe; then
    echo "--- Stage: probe ---"
    for EP in "${EPOCHS_PRETRAIN[@]}"; do
      for i in "${!VIEW2S[@]}"; do
        V2="${VIEW2S[$i]}"; V3="${VIEW3S[$i]}"
        if [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; then
          ACTIVE_MODES=("${LOGSIG_MODES[@]}")
        else
          ACTIVE_MODES=("stream")
        fi
        for MODE in "${ACTIVE_MODES[@]}"; do
          if [ "$MODE" = "stream" ]; then
            WSIZES=(0)
          else
            WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
          fi
          for WSIZ in "${WSIZES[@]}"; do
            LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING")"
            for SEED in "${SEEDS[@]}"; do
              # raw probe (no checkpoint needed — one per view/seed/mode combo)
              TAG_RAW="probe_raw_${FINETUNE_DATA}_v2${V2}_v3${V3}_${SEED}${LS}"
              _launch "$TAG_RAW" "logs/${TAG_RAW}.log" \
                python run_probe.py \
                  --probe_type raw \
                  --data_name "${FINETUNE_DATA}" \
                  --pretrain_data_name "${PRETRAIN_DATA}" \
                  --num_feature "${FINETUNE_NUM_FEATURE}" \
                  --num_target  "${FINETUNE_NUM_TARGET}" \
                  --view2 "${V2}" --view3 "${V3}" \
                  --logsig_depth "${LOGSIG_DEPTH}" \
                  --logsig_mode "${MODE}" \
                  --logsig_window_size "${WSIZ}" \
                  --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                  --epochs_pretrain "${EP}" \
                  --epochs_finetune "${EPOCHS_PROBE}" \
                  --loss_type "${LOSS_TYPE}" \
                  --seed "${SEED}"

              # pretrained probe (transformer checkpoint)
              PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${LS}"
              PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
              TAG_PT="probe_pt_${FINETUNE_DATA}_from_${PT_TAG}"
              if [ ! -f "$PT_CKPT" ]; then
                echo "Skipping pretrained probe — missing checkpoint: ${PT_CKPT}"
              else
                _launch "$TAG_PT" "logs/${TAG_PT}.log" \
                  python run_probe.py \
                    --probe_type pretrained \
                    --data_name "${FINETUNE_DATA}" \
                    --pretrain_data_name "${PRETRAIN_DATA}" \
                    --num_feature "${FINETUNE_NUM_FEATURE}" \
                    --num_target  "${FINETUNE_NUM_TARGET}" \
                    --view2 "${V2}" --view3 "${V3}" \
                    --logsig_depth "${LOGSIG_DEPTH}" \
                    --logsig_mode "${MODE}" \
                    --logsig_window_size "${WSIZ}" \
                    --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                    --epochs_pretrain "${EP}" \
                    --epochs_finetune "${EPOCHS_PROBE}" \
                    --loss_type "${LOSS_TYPE}" \
                    --seed "${SEED}"
              fi
            done
          done
        done
      done
    done
    [ "$PARALLEL" = true ] && { wait; echo "Probe jobs done."; }
  fi

done

echo ""
echo "All sweeps complete."
