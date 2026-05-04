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
# Each pair runs three sections in order:
#   A. 3-view sweep  — xt+dx+xf, xt+logsig+xf, xt+dx+logsig
#                      pretrain → finetune → probe
#   B. 2-view sweep  — xt+logsig only (uses run_{pretrain,finetune}_nview.py)
#                      pretrain → finetune
#
# Set SKIP_STAGES to omit stages, e.g. SKIP_STAGES="probe" bash run_sweep.sh
# Set SKIP_SECTIONS to omit entire sections, e.g. SKIP_SECTIONS="2view"
#
# Logsig modes:
#   stream         — running log-sig of [0,t]; window size irrelevant.
#   window W       — sliding window size W, const-pad early windows.
#   window_smooth W — sliding window + Tukey smoothing.
#
# Checkpoint suffixes (never overwrite each other):
#   stream               →  (no suffix)
#   window W             →  _winW
#   window_smooth W      →  _tukeyW
#   2-view               →  _nview appended to checkpoint name
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Common parameters — edit here to change all pairs at once
# ---------------------------------------------------------------------------
SEEDS=(0)
EPOCHS_PRETRAIN=(2)
EPOCHS_FINETUNE=10
EPOCHS_PROBE=50

# ---------------------------------------------------------------------------
# 3-view combos
# ---------------------------------------------------------------------------
VIEW2S=("dx" "logsig" "dx")
VIEW3S=("xf"  "xf"   "logsig")
ENCODER_TYPES=("transformer" "mlp_logsig")

# ---------------------------------------------------------------------------
# Logsig configuration
#   LOGSIG_MODES:        which modes to run
#   LOGSIG_WINDOW_SIZES: window sizes for window/window_smooth modes
#   LOGSIG_SMOOTHING:    smoothing used for window_smooth mode
# ---------------------------------------------------------------------------
LOGSIG_MODES=("stream" "window" "window_smooth")
LOGSIG_WINDOW_SIZES=(64 128)
LOGSIG_SMOOTHING="tukey"
LOGSIG_DEPTH=2

FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"
RUN_MODES="finetune"   # set to "finetune,freeze,baseline" to re-enable all three

PARALLEL=false
DEFAULT_BATCH=64
LOGSIG_BATCH=32
DISABLE_TQDM=0

SKIP_STAGES="${SKIP_STAGES:-}"     # space-separated: "pretrain" "finetune" "probe"
SKIP_SECTIONS="${SKIP_SECTIONS:-}" # space-separated: "3view" "2view"
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
    TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" | tee "$log" &
  else
    echo "Running: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} "${cmd[@]}" | tee "$log"
  fi
  job_index=$(( job_index + 1 ))
}

_skip_stage()   { [[ " ${SKIP_STAGES}   " == *" $1 "* ]]; }
_skip_section() { [[ " ${SKIP_SECTIONS} " == *" $1 "* ]]; }

# Build the logsig checkpoint suffix that matches _logsig_suffix() in the Python scripts.
#   stream        → ""
#   window W      → "_winW"
#   window_smooth → "_tukeyW" / "_emaW"
#   stride S>1    → append "_sS"
#   global_time   → append "_gt"
_lsig_suffix() {
  local mode="$1" wsiz="$2" smoothing="$3" stride="${4:-1}" gt="${5:-0}"
  local base=""
  if [ "$mode" = "stream" ]; then
    base=""
  elif [ "$mode" = "window" ]; then
    base="_win${wsiz}"
  else
    base="_${smoothing}${wsiz}"
  fi
  [ "$stride" -gt 1 ] 2>/dev/null && base="${base}_s${stride}"
  [ "$gt" = "1" ]              && base="${base}_gt"
  echo "$base"
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

  # =========================================================================
  # SECTION A: 3-view sweep
  # =========================================================================
  if ! _skip_section 3view; then

    # -----------------------------------------------------------------------
    # Stage A1: Pretrain (3-view)
    # -----------------------------------------------------------------------
    if ! _skip_stage pretrain; then
      echo "--- [3-view] Stage: pretrain ---"
      for ENC in "${ENCODER_TYPES[@]}"; do
        ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
        for EP in "${EPOCHS_PRETRAIN[@]}"; do
          for i in "${!VIEW2S[@]}"; do
            V2="${VIEW2S[$i]}"; V3="${VIEW3S[$i]}"
            BATCH=$DEFAULT_BATCH
            { [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; } && BATCH=$LOGSIG_BATCH
            if [ "$V2" = "logsig" ] || [ "$V3" = "logsig" ]; then
              ACTIVE_MODES=("${LOGSIG_MODES[@]}")
            else
              ACTIVE_MODES=("stream")
            fi
            [ "$ENC" = "mlp_logsig" ] && [ "$V2" != "logsig" ] && [ "$V3" != "logsig" ] && continue
            for MODE in "${ACTIVE_MODES[@]}"; do
              if [ "$MODE" = "stream" ]; then
                WSIZES=(0)
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
      [ "$PARALLEL" = true ] && { wait; echo "3-view pretrain jobs done."; }
    fi

    # -----------------------------------------------------------------------
    # Stage A2: Finetune (3-view)
    # -----------------------------------------------------------------------
    if ! _skip_stage finetune; then
      echo "--- [3-view] Stage: finetune ---"
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
            [ "$ENC" = "mlp_logsig" ] && [ "$V2" != "logsig" ] && [ "$V3" != "logsig" ] && continue
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
                      --run_modes "${RUN_MODES}" \
                      --seed "${SEED}"
                done
              done
            done
          done
        done
      done
      [ "$PARALLEL" = true ] && { wait; echo "3-view finetune jobs done."; }
    fi

    # -----------------------------------------------------------------------
    # Stage A3: Probe (3-view, transformer encoder only)
    # -----------------------------------------------------------------------
    if ! _skip_stage probe; then
      echo "--- [3-view] Stage: probe ---"
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

                PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${LS}"
                PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
                TAG_PT="probe_pt_${FINETUNE_DATA}_from_${PT_TAG}"
                if [ ! -f "$PT_CKPT" ]; then
                  echo "Skipping pretrained probe — missing: ${PT_CKPT}"
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
      [ "$PARALLEL" = true ] && { wait; echo "3-view probe jobs done."; }
    fi

  fi  # end section A (3-view)

  # =========================================================================
  # SECTION B: 2-view sweep  (xt + logsig)
  # Only runs stream / window / window_smooth × window sizes × encoder types.
  # No probe stage (run_probe.py is 3-view only).
  # =========================================================================
  if ! _skip_section 2view; then

    echo ""
    echo "--- [2-view xt+logsig] ---"

    for ENC in "${ENCODER_TYPES[@]}"; do
      ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
      for EP in "${EPOCHS_PRETRAIN[@]}"; do
        for MODE in "${LOGSIG_MODES[@]}"; do
          if [ "$MODE" = "stream" ]; then
            WSIZES=(0)
          else
            WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
          fi
          for WSIZ in "${WSIZES[@]}"; do
            LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING")"
            for SEED in "${SEEDS[@]}"; do

              # ------ 2-view pretrain ------
              if ! _skip_stage pretrain; then
                TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
                _launch "$TAG" "logs/${TAG}.log" \
                  python -u run_pretrain_nview.py \
                    --data_name "${PRETRAIN_DATA}" \
                    --num_feature "${PRETRAIN_NUM_FEATURE}" \
                    --num_target  "${PRETRAIN_NUM_TARGET}" \
                    --view2 logsig \
                    --encoder_type "${ENC}" \
                    --batch_size_pretrain "${LOGSIG_BATCH}" \
                    --epochs_pretrain "${EP}" \
                    --logsig_depth "${LOGSIG_DEPTH}" \
                    --logsig_mode "${MODE}" \
                    --logsig_window_size "${WSIZ}" \
                    --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                    --seed "${SEED}"
              fi

              # ------ 2-view finetune ------
              if ! _skip_stage finetune; then
                PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
                PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
                if [ ! -f "$PT_CKPT" ]; then
                  echo "Skipping 2-view finetune — missing: ${PT_CKPT}"
                else
                  TAG="ft2v_${FINETUNE_DATA}_from_${PT_TAG}"
                  _launch "$TAG" "logs/${TAG}.log" \
                    python run_finetune_nview.py \
                      --data_name "${FINETUNE_DATA}" \
                      --pretrain_data_name "${PRETRAIN_DATA}" \
                      --num_feature "${FINETUNE_NUM_FEATURE}" \
                      --num_target  "${FINETUNE_NUM_TARGET}" \
                      --view2 logsig \
                      --encoder_type "${ENC}" \
                      --logsig_depth "${LOGSIG_DEPTH}" \
                      --logsig_mode "${MODE}" \
                      --logsig_window_size "${WSIZ}" \
                      --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                      --epochs_pretrain "${EP}" \
                      --epochs_finetune "${EPOCHS_FINETUNE}" \
                      --feature "${FEATURE}" \
                      --lam "${LAM}" \
                      --run_modes "${RUN_MODES}" \
                      --seed "${SEED}"
                fi
              fi

            done
          done
        done
      done
    done
    [ "$PARALLEL" = true ] && { wait; echo "2-view jobs done."; }

  fi  # end section B (2-view)

done

echo ""
echo "All sweeps complete."
