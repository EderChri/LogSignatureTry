#!/bin/bash
# =============================================================================
# run_sweep_capture24.sh — pretrain on capture24, finetune on all targets
#
# Usage:
#   bash run_sweep_capture24.sh [pair]             # run one pair or all
#   GPUS="0,1" bash run_sweep_capture24.sh         # specific GPUs (sequential)
#   GPUS="0,1" PARALLEL=true bash run_sweep_capture24.sh  # parallel
#   SEEDS="0 1 2" bash run_sweep_capture24.sh      # subset of seeds
#   SKIP_SECTIONS="2view" bash run_sweep_capture24.sh
#
# Capture24 pretrain targets (6 pairs):
#   capture24_har70plus   — 6ch IMU, 7 classes
#   capture24_wisdm       — 3ch accel, 6 classes
#   capture24_wisdm2      — 3ch accel, 6 classes
#   capture24_usc_had     — 6ch motion, 12 classes
#   capture24_opportunity — 113ch → 32 PCA, 18 classes
#   capture24_skoda       — 60ch  →  4 PCA, 11 classes
#
# PCA is applied only during finetuning (capture24 pretrain is always 3ch).
# Components chosen for ≥90% explained variance:
#   Opportunity 113ch → 32 PCs
#   Skoda       60ch  →  4 PCs
#
# Sections:
#   A: 3-view xt+dx+xf, xt+logsig+xf, xt+dx+logsig  (pretrain → finetune)
#   B: 2-view xt+logsig nview                         (pretrain → finetune)
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
IFS=' ' read -ra SEEDS            <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS_PRETRAIN=(2)
EPOCHS_FINETUNE=10

IFS=' ' read -ra INTERACTION_TYPES <<< "${INTERACTION_TYPES:-bilinear}"
IFS=' ' read -ra ENCODER_TYPES     <<< "${ENCODER_TYPES:-transformer mlp_logsig}"

VIEW2S=("dx" "logsig" "dx")
VIEW3S=("xf"  "xf"   "logsig")

IFS=' ' read -ra LOGSIG_MODES        <<< "${LOGSIG_MODES:-stream window window_smooth}"
IFS=' ' read -ra LOGSIG_WINDOW_SIZES <<< "${LOGSIG_WINDOW_SIZES:-64 128}"
LOGSIG_SMOOTHING="tukey"
IFS=' ' read -ra LOGSIG_DEPTHS       <<< "${LOGSIG_DEPTHS:-2}"
LOGSIG_DEPTH="${LOGSIG_DEPTHS[0]}"
IFS=' ' read -ra LOGSIG_SMOOTH_PARAMS <<< "${LOGSIG_SMOOTH_PARAMS:-0.5}"

FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"
RUN_MODES="finetune"

PARALLEL="${PARALLEL:-false}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DEFAULT_BATCH=256
LOGSIG_BATCH=128
DISABLE_TQDM=0

SKIP_STAGES="${SKIP_STAGES:-}"
SKIP_SECTIONS="${SKIP_SECTIONS:-3view pool_ablation stride bilinear long_pretrain multi_smooth depth smooth_param}"

# ---------------------------------------------------------------------------
source datasets.cfg

ALL_PAIRS=(
  "capture24_har70plus"
  "capture24_wisdm"
  "capture24_wisdm2"
  "capture24_usc_had"
  "capture24_opportunity"
  "capture24_skoda"
)

REQUESTED="${1:-all}"
if [ "$REQUESTED" = "all" ]; then
  RUN_PAIRS=("${ALL_PAIRS[@]}")
else
  RUN_PAIRS=("$REQUESTED")
fi

mkdir -p logs

_gpus_raw="${GPUS:-0}"
IFS=',' read -ra GPU_LIST <<< "$_gpus_raw"
num_gpus=${#GPU_LIST[@]}
echo "GPUs: [${GPU_LIST[*]}]  (${num_gpus} total)"

_gpu_has_live_procs() {
  local gid="$1"
  local pids
  pids=$(nvidia-smi -i "$gid" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$pids" ] && return 1
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null && return 0
  done
  return 1
}

_wait_for_gpu() {
  local gid="$1"
  if _gpu_has_live_procs "$gid"; then
    echo "  GPU ${gid} busy — waiting..."
    while _gpu_has_live_procs "$gid"; do sleep 15; done
    sleep 5
  fi
}

_launch() {
  local tag="$1"; local log="$2"; shift 2; local cmd=("$@")
  local gid="${GPU_LIST[$(( job_index % num_gpus ))]}"
  _wait_for_gpu "$gid"
  echo "Running [GPU ${gid}]: ${tag}"
  TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${cmd[@]}" 2>&1 | tee "$log"
  job_index=$(( job_index + 1 ))
}

_launch_if_new() {
  local out_file="$1"; shift
  if [ -f "$out_file" ]; then
    echo "Skip (exists): $(basename "$out_file")"
    return
  fi
  _launch "$@"
}

_skip_stage()   { [[ " ${SKIP_STAGES}   " == *" $1 "* ]]; }
_skip_section() { [[ " ${SKIP_SECTIONS} " == *" $1 "* ]]; }

_il_suffix() {
  local il_type="$1"
  [ "$il_type" = "attention" ] && echo "" || echo "_il${il_type//_/}"
}

_lsig_suffix() {
  local mode="$1" wsiz="$2" smoothing="$3" stride="${4:-1}" gt="${5:-0}" sp="${6:-0.5}" depth="${7:-2}" msp_k="${8:-0}"
  local base=""
  if [ "$mode" = "stream" ]; then
    base=""
  elif [ "$mode" = "window" ]; then
    base="_win${wsiz}"
  else
    base="_${smoothing}${wsiz}"
    if [ "${msp_k}" -gt 0 ] 2>/dev/null; then
      base="${base}_msp${msp_k}"
    elif [ "$mode" = "window_smooth" ] && [ "$sp" != "0.5" ]; then
      base="${base}_sp${sp}"
    fi
  fi
  [ "$stride" -gt 1 ] 2>/dev/null && base="${base}_s${stride}"
  [ "$gt" = "1" ]              && base="${base}_gt"
  [ "$depth" != "2" ]          && base="${base}_d${depth}"
  echo "$base"
}

# ---------------------------------------------------------------------------
# Per-pair PCA args for finetuning on high-dimensional targets.
# Fit on training data only; components chosen for >=90% explained variance:
#   Opportunity 113ch -> 32 PCs
#   Skoda       60ch  ->  4 PCs
# ---------------------------------------------------------------------------
_pca_finetune_args() {
  case "$1" in
    capture24_opportunity) echo "--pca_components 32" ;;
    capture24_skoda)       echo "--pca_components 4"  ;;
    *)                     echo ""                    ;;
  esac
}

# ===========================================================================

for PAIR in "${RUN_PAIRS[@]}"; do
  cfg_var="cfg_${PAIR}[@]"
  for assignment in "${!cfg_var}"; do
    eval "$assignment"
  done

  # Build per-pair PCA finetune args as an array
  PCA_FT_STR="$(_pca_finetune_args "$PAIR")"
  PCA_FT_ARGS=()
  # shellcheck disable=SC2086
  read -ra PCA_FT_ARGS <<< $PCA_FT_STR

  echo ""
  echo "======================================================================="
  echo " Pair: ${PAIR}"
  echo "   Pretrain: ${PRETRAIN_DATA}  (${PRETRAIN_NUM_FEATURE} ch, ${PRETRAIN_NUM_TARGET} cls)"
  echo "   Finetune: ${FINETUNE_DATA}  (${FINETUNE_NUM_FEATURE} ch, ${FINETUNE_NUM_TARGET} cls)"
  [ "${#PCA_FT_ARGS[@]}" -gt 0 ] && echo "   PCA:      ${PCA_FT_ARGS[*]}"
  echo "======================================================================="

  job_index=0

  # =========================================================================
  # SECTION A: 3-view (xt+dx+xf, xt+logsig+xf, xt+dx+logsig)
  # =========================================================================
  if ! _skip_section 3view; then

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
              [ "$MODE" = "stream" ] && WSIZES=(0) || WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
              for WSIZ in "${WSIZES[@]}"; do
                SP="0.5"; DEPTH=2
                LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$SP" "$DEPTH")"
                for SEED in "${SEEDS[@]}"; do
                  for IL_TYPE in "${INTERACTION_TYPES[@]}"; do
                    IL_SFXR="$(_il_suffix "$IL_TYPE")"
                    IL_ARGS=(); [ "$IL_TYPE" != "attention" ] && IL_ARGS=(--interaction_type "${IL_TYPE}")
                    TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${ENC_SUFFIX}${LS}${IL_SFXR}"
                    _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
                      "$TAG" "logs/${TAG}.log" \
                      python -u run_pretrain.py \
                        --data_name "${PRETRAIN_DATA}" \
                        --num_feature "${PRETRAIN_NUM_FEATURE}" \
                        --num_target  "${PRETRAIN_NUM_TARGET}" \
                        --view2 "${V2}" --view3 "${V3}" \
                        --encoder_type "${ENC}" \
                        --batch_size_pretrain "${BATCH}" \
                        --epochs_pretrain "${EP}" \
                        --logsig_depth "${DEPTH}" \
                        --logsig_mode "${MODE}" \
                        --logsig_window_size "${WSIZ}" \
                        --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                        --logsig_smooth_param "${SP}" \
                        --seed "${SEED}" \
                        "${IL_ARGS[@]}"
                  done
                done
              done
            done
          done
        done
      done
    fi

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
              [ "$MODE" = "stream" ] && WSIZES=(0) || WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
              for WSIZ in "${WSIZES[@]}"; do
                SP="0.5"; DEPTH=2
                LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$SP" "$DEPTH")"
                for SEED in "${SEEDS[@]}"; do
                  for IL_TYPE in "${INTERACTION_TYPES[@]}"; do
                    IL_SFXR="$(_il_suffix "$IL_TYPE")"
                    IL_ARGS=(); [ "$IL_TYPE" != "attention" ] && IL_ARGS=(--interaction_type "${IL_TYPE}")
                    PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${ENC_SUFFIX}${LS}${IL_SFXR}"
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
                        --logsig_depth "${DEPTH}" \
                        --logsig_mode "${MODE}" \
                        --logsig_window_size "${WSIZ}" \
                        --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                        --logsig_smooth_param "${SP}" \
                        --epochs_pretrain "${EP}" \
                        --epochs_finetune "${EPOCHS_FINETUNE}" \
                        --feature "${FEATURE}" \
                        --loss_type "${LOSS_TYPE}" \
                        --lam "${LAM}" \
                        --run_modes "${RUN_MODES}" \
                        --seed "${SEED}" \
                        "${IL_ARGS[@]}" \
                        "${PCA_FT_ARGS[@]}"
                  done
                done
              done
            done
          done
        done
      done
    fi

  fi  # end section A

  # =========================================================================
  # SECTION B: 2-view xt+logsig (nview scripts)
  # =========================================================================
  if ! _skip_section 2view; then

    echo ""
    echo "--- [2-view xt+logsig] ---"

    IFS=' ' read -ra B_IL_TYPES <<< "${B_INTERACTION_TYPES:-${INTERACTION_TYPES[*]}}"
    for IL_TYPE_B in "${B_IL_TYPES[@]}"; do
    IL_SFXR_B="$(_il_suffix "$IL_TYPE_B")"
    IL_ARGS_B=(); [ "$IL_TYPE_B" != "attention" ] && IL_ARGS_B=(--interaction_type "${IL_TYPE_B}")
    for ENC in "${ENCODER_TYPES[@]}"; do
      ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
      for EP in "${EPOCHS_PRETRAIN[@]}"; do
        for MODE in "${LOGSIG_MODES[@]}"; do
          [ "$MODE" = "stream" ] && WSIZES=(0) || WSIZES=("${LOGSIG_WINDOW_SIZES[@]}")
          for WSIZ in "${WSIZES[@]}"; do
            [ "$MODE" = "stream" ] && STRIDES_2V=(1) || STRIDES_2V=(1 7)
            for STRIDE_2V in "${STRIDES_2V[@]}"; do
            SP="0.5"; DEPTH=2
            LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING" "$STRIDE_2V" "0" "$SP" "$DEPTH")"
            for SEED in "${SEEDS[@]}"; do

              if ! _skip_stage pretrain; then
                TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}${ENC_SUFFIX}${LS}${IL_SFXR_B}"
                _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
                  "$TAG" "logs/${TAG}.log" \
                  python -u run_pretrain_nview.py \
                    --data_name "${PRETRAIN_DATA}" \
                    --num_feature "${PRETRAIN_NUM_FEATURE}" \
                    --num_target  "${PRETRAIN_NUM_TARGET}" \
                    --view2 logsig \
                    --encoder_type "${ENC}" \
                    --batch_size_pretrain "${LOGSIG_BATCH}" \
                    --epochs_pretrain "${EP}" \
                    --logsig_depth "${DEPTH}" \
                    --logsig_mode "${MODE}" \
                    --logsig_window_size "${WSIZ}" \
                    --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                    --logsig_smooth_param "${SP}" \
                    --logsig_stride "${STRIDE_2V}" \
                    --seed "${SEED}" \
                    "${IL_ARGS_B[@]}"
              fi

              if ! _skip_stage finetune; then
                PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}${ENC_SUFFIX}${LS}${IL_SFXR_B}"
                PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
                FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
                if [ -f "$FT_OUT" ]; then
                  echo "Skip finetune (exists): $(basename "$FT_OUT")"
                elif [ ! -f "$PT_CKPT" ]; then
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
                      --logsig_depth "${DEPTH}" \
                      --logsig_mode "${MODE}" \
                      --logsig_window_size "${WSIZ}" \
                      --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                      --logsig_smooth_param "${SP}" \
                      --logsig_stride "${STRIDE_2V}" \
                      --epochs_pretrain "${EP}" \
                      --epochs_finetune "${EPOCHS_FINETUNE}" \
                      --feature "${FEATURE}" \
                      --lam "${LAM}" \
                      --run_modes "${RUN_MODES}" \
                      --seed "${SEED}" \
                      "${IL_ARGS_B[@]}" \
                      "${PCA_FT_ARGS[@]}"
                fi
              fi

            done
            done  # STRIDE_2V
          done
        done
      done
    done
    done  # IL_TYPE_B

  fi  # end section B

done  # PAIR
