#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
# =============================================================================
# run_norm_win64.sh — 3-view logsig+xf, window-64, mlp_logsig, bilinear, with
#                     per-level logsig normalisation.
#
# Pairs (in order):
#   sleepeeg_epilepsy
#   harth_har70plus
#   capture24_wisdm
#   capture24_wisdm2
#   capture24_har70plus
#
# Usage:
#   bash scripts/run_norm_win64.sh               # default GPUs 5,7
#   GPUS="5,7" bash scripts/run_norm_win64.sh
#   SEEDS="0 1 2" bash scripts/run_norm_win64.sh  # subset of seeds
# =============================================================================

set -uo pipefail

export PYTHONPATH=".:${PYTHONPATH:-}"

IFS=' ' read -ra SEEDS <<< "${SEEDS:-0 1 2 3 4}"

ENC="mlp_logsig"
ENC_SUFFIX="_mlp_logsig"
MODE="window"
WSIZ=64
DEPTH=2
SP="0.5"
SMOOTHING="tukey"
EP=2
EPOCHS_FINETUNE=10
FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"
RUN_MODES="finetune"
BATCH=128
IL_ARGS=(--interaction_type bilinear)
IL_SFXR="_ilbilinear"
# Suffix order must match Python's _logsig_suffix (lsig part) then _il_suffix:
#   _win64        ← logsig mode/window
#   _norm         ← logsig normalize (appended last inside _logsig_suffix)
#   _ilbilinear   ← interaction layer (separate suffix in the run scripts)
LS_SFXR="_win${WSIZ}"
NORM_SFXR="_norm"

source datasets.cfg

PAIRS=(
  "sleepeeg_epilepsy"
  "harth_har70plus"
  "capture24_wisdm"
  "capture24_wisdm2"
  "capture24_har70plus"
)

_gpus_raw="${GPUS:-5,7}"
IFS=',' read -ra GPU_LIST <<< "$_gpus_raw"
num_gpus=${#GPU_LIST[@]}
echo "GPUs: [${GPU_LIST[*]}]  (${num_gpus} total)"
mkdir -p logs

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

job_index=0

_launch() {
  local tag="$1"; local log="$2"; shift 2; local cmd=("$@")
  local gid="${GPU_LIST[$(( job_index % num_gpus ))]}"
  _wait_for_gpu "$gid"
  echo "Running [GPU ${gid}]: ${tag}"
  CUDA_VISIBLE_DEVICES=$gid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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

# ---------------------------------------------------------------------------

for PAIR in "${PAIRS[@]}"; do
  cfg_var="cfg_${PAIR}[@]"
  for assignment in "${!cfg_var}"; do eval "$assignment"; done

  echo ""
  echo "======================================================================="
  echo " Pair: ${PAIR}"
  echo "   Pretrain: ${PRETRAIN_DATA}  (${PRETRAIN_NUM_FEATURE} ch, ${PRETRAIN_NUM_TARGET} cls)"
  echo "   Finetune: ${FINETUNE_DATA}  (${FINETUNE_NUM_FEATURE} ch, ${FINETUNE_NUM_TARGET} cls)"
  echo "======================================================================="

  # --- pretrain ---
  echo "--- pretrain ---"
  for SEED in "${SEEDS[@]}"; do
    TAG="${PRETRAIN_DATA}_v2logsig_v3xf_ep${EP}_${SEED}${ENC_SUFFIX}${LS_SFXR}${NORM_SFXR}${IL_SFXR}"
    _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
      "$TAG" "logs/${TAG}.log" \
      python -u scripts/run_pretrain.py \
        --data_name "${PRETRAIN_DATA}" \
        --num_feature "${PRETRAIN_NUM_FEATURE}" \
        --num_target  "${PRETRAIN_NUM_TARGET}" \
        --view2 logsig --view3 xf \
        --encoder_type "${ENC}" \
        --batch_size_pretrain "${BATCH}" \
        --epochs_pretrain "${EP}" \
        --logsig_depth "${DEPTH}" \
        --logsig_mode "${MODE}" \
        --logsig_window_size "${WSIZ}" \
        --logsig_smoothing "${SMOOTHING}" \
        --logsig_smooth_param "${SP}" \
        --logsig_normalize \
        --seed "${SEED}" \
        "${IL_ARGS[@]}"
  done

  # --- finetune ---
  echo "--- finetune ---"
  for SEED in "${SEEDS[@]}"; do
    PT_TAG="${PRETRAIN_DATA}_v2logsig_v3xf_ep${EP}_${SEED}${ENC_SUFFIX}${LS_SFXR}${NORM_SFXR}${IL_SFXR}"
    PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
    if [ ! -f "$PT_CKPT" ]; then
      echo "Skipping finetune — missing checkpoint: ${PT_CKPT}"
      continue
    fi
    TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
    _launch "$TAG" "logs/${TAG}.log" \
      python scripts/run_finetune.py \
        --data_name "${FINETUNE_DATA}" \
        --pretrain_data_name "${PRETRAIN_DATA}" \
        --num_feature "${FINETUNE_NUM_FEATURE}" \
        --num_target  "${FINETUNE_NUM_TARGET}" \
        --view2 logsig --view3 xf \
        --encoder_type "${ENC}" \
        --logsig_depth "${DEPTH}" \
        --logsig_mode "${MODE}" \
        --logsig_window_size "${WSIZ}" \
        --logsig_smoothing "${SMOOTHING}" \
        --logsig_smooth_param "${SP}" \
        --logsig_normalize \
        --epochs_pretrain "${EP}" \
        --epochs_finetune "${EPOCHS_FINETUNE}" \
        --feature "${FEATURE}" \
        --loss_type "${LOSS_TYPE}" \
        --lam "${LAM}" \
        --run_modes "${RUN_MODES}" \
        --seed "${SEED}" \
        "${IL_ARGS[@]}"
  done

done

echo ""
echo "All runs complete."
