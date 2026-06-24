#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
# run_missing_finetunes.sh
# Run the finetune jobs missing multi-seed coverage (seeds 1-9).
#
# Covers:
#   1. Pooling ablation (_plast): logsig+xf, dx+logsig, logsig_nview — seeds 1-9
#   2. Stride ablation (_win128_s7, _tukey128_s7): logsig+xf, dx+logsig
#      × {transformer, mlp_logsig} — seeds 1-9
#
# All pretrain checkpoints already exist; this is finetune-only.
# Total: 22 config groups × 9 seeds = 198 finetune jobs.
#
# Usage:
#   bash run_missing_finetunes.sh
#   GPUS="6,7,8" PARALLEL=true MAX_PARALLEL=6 bash run_missing_finetunes.sh

set -uo pipefail

EPOCHS_PRETRAIN=2
EPOCHS_FINETUNE=10
LOGSIG_DEPTH=2
FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"
RUN_MODES="finetune"

PARALLEL="${PARALLEL:-false}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DISABLE_TQDM="${DISABLE_TQDM:-0}"

_gpus_raw="${GPUS:-0}"
IFS=',' read -ra GPU_LIST <<< "$_gpus_raw"
num_gpus=${#GPU_LIST[@]}
echo "GPUs: [${GPU_LIST[*]}]  (${num_gpus} total)"
job_index=0

mkdir -p logs

_launch() {
  local tag="$1"; local log="$2"; shift 2; local cmd=("$@")
  local gid="${GPU_LIST[$(( job_index % num_gpus ))]}"
  if [ "$PARALLEL" = true ]; then
    if [ "${MAX_PARALLEL}" -gt 0 ] 2>/dev/null; then
      while [ "$(jobs -r | wc -l)" -ge "${MAX_PARALLEL}" ]; do sleep 2; done
    fi
    echo "Launching [GPU ${gid}]: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" > "$log" 2>&1 &
  else
    echo "Running [GPU ${gid}]: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" 2>&1 | tee "$log"
  fi
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
# Dataset pairs
# ---------------------------------------------------------------------------
declare -A PT_DATA=( [sleepeeg_epilepsy]="_DA_SleepEEG_256_00"  [harth_har70plus]="_DA_HARTH_256_00" )
declare -A FT_DATA=( [sleepeeg_epilepsy]="_DA_Epilepsy_256_00"  [harth_har70plus]="_DA_HAR70plus_256_00" )
declare -A FT_FEAT=( [sleepeeg_epilepsy]=1                       [harth_har70plus]=6 )
declare -A FT_TARG=( [sleepeeg_epilepsy]=2                       [harth_har70plus]=7 )

SEEDS=(1 2 3 4 5 6 7 8 9)

for PAIR in sleepeeg_epilepsy harth_har70plus; do
  PT="${PT_DATA[$PAIR]}"
  FT="${FT_DATA[$PAIR]}"
  FT_NF="${FT_FEAT[$PAIR]}"
  FT_NT="${FT_TARG[$PAIR]}"

  echo ""
  echo "======================================================================="
  echo " Pair: ${PAIR}  (pretrain: ${PT}  finetune: ${FT})"
  echo "======================================================================="

  # -------------------------------------------------------------------------
  # 1. Pooling ablation — transformer + stream + last-token (_plast), 3-view
  # -------------------------------------------------------------------------
  echo "--- [plast 3-view] ---"
  for V2V3 in "logsig xf" "dx logsig"; do
    V2="${V2V3% *}"; V3="${V2V3#* }"
    for SEED in "${SEEDS[@]}"; do
      PT_TAG="${PT}_v2${V2}_v3${V3}_ep${EPOCHS_PRETRAIN}_${SEED}_plast"
      PT_CKPT="model_pretrain/${PT}/${PT_TAG}.pth"
      FT_OUT="out_finetune/${FT}/${FT}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
      if [ ! -f "$PT_CKPT" ]; then
        echo "  Skipping — missing checkpoint: ${PT_CKPT}"; continue
      fi
      TAG="ft_${FT}_from_${PT_TAG}"
      _launch_if_new "$FT_OUT" "$TAG" "logs/${TAG}.log" \
        python scripts/run_finetune.py \
          --data_name "${FT}" \
          --pretrain_data_name "${PT}" \
          --num_feature "${FT_NF}" \
          --num_target  "${FT_NT}" \
          --view2 "${V2}" --view3 "${V3}" \
          --encoder_type transformer \
          --logsig_depth "${LOGSIG_DEPTH}" \
          --logsig_mode stream \
          --logsig_pool last \
          --epochs_pretrain "${EPOCHS_PRETRAIN}" \
          --epochs_finetune "${EPOCHS_FINETUNE}" \
          --feature "${FEATURE}" \
          --loss_type "${LOSS_TYPE}" \
          --lam "${LAM}" \
          --run_modes "${RUN_MODES}" \
          --seed "${SEED}"
    done
  done

  # -------------------------------------------------------------------------
  # 2. Pooling ablation — 2-view nview + stream + last-token (_plast)
  # -------------------------------------------------------------------------
  echo "--- [plast nview] ---"
  for SEED in "${SEEDS[@]}"; do
    PT_TAG="${PT}_v2logsig_nview_ep${EPOCHS_PRETRAIN}_${SEED}_plast"
    PT_CKPT="model_pretrain/${PT}/${PT_TAG}.pth"
    FT_OUT="out_finetune/${FT}/${FT}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
    if [ ! -f "$PT_CKPT" ]; then
      echo "  Skipping — missing checkpoint: ${PT_CKPT}"; continue
    fi
    TAG="ft2v_${FT}_from_${PT_TAG}"
    _launch_if_new "$FT_OUT" "$TAG" "logs/${TAG}.log" \
      python scripts/run_finetune_nview.py \
        --data_name "${FT}" \
        --pretrain_data_name "${PT}" \
        --num_feature "${FT_NF}" \
        --num_target  "${FT_NT}" \
        --view2 logsig \
        --encoder_type transformer \
        --logsig_depth "${LOGSIG_DEPTH}" \
        --logsig_mode stream \
        --logsig_pool last \
        --epochs_pretrain "${EPOCHS_PRETRAIN}" \
        --epochs_finetune "${EPOCHS_FINETUNE}" \
        --feature "${FEATURE}" \
        --lam "${LAM}" \
        --run_modes "${RUN_MODES}" \
        --seed "${SEED}"
  done

  # -------------------------------------------------------------------------
  # 3. Stride ablation — win128_s7 and tukey128_s7 × {transformer, mlp_logsig}
  #    3-view: logsig+xf and dx+logsig
  # -------------------------------------------------------------------------
  echo "--- [stride win128_s7 + tukey128_s7, 3-view] ---"
  for V2V3 in "logsig xf" "dx logsig"; do
    V2="${V2V3% *}"; V3="${V2V3#* }"
    for ENC in transformer mlp_logsig; do
      ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
      for MODE_LS in "window _win128_s7" "window_smooth _tukey128_s7"; do
        MODE="${MODE_LS% *}"; LS="${MODE_LS#* }"
        for SEED in "${SEEDS[@]}"; do
          PT_TAG="${PT}_v2${V2}_v3${V3}_ep${EPOCHS_PRETRAIN}_${SEED}${ENC_SUFFIX}${LS}"
          PT_CKPT="model_pretrain/${PT}/${PT_TAG}.pth"
          FT_OUT="out_finetune/${FT}/${FT}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
          if [ ! -f "$PT_CKPT" ]; then
            echo "  Skipping — missing checkpoint: ${PT_CKPT}"; continue
          fi
          TAG="ft_${FT}_from_${PT_TAG}"
          _launch_if_new "$FT_OUT" "$TAG" "logs/${TAG}.log" \
            python scripts/run_finetune.py \
              --data_name "${FT}" \
              --pretrain_data_name "${PT}" \
              --num_feature "${FT_NF}" \
              --num_target  "${FT_NT}" \
              --view2 "${V2}" --view3 "${V3}" \
              --encoder_type "${ENC}" \
              --logsig_depth "${LOGSIG_DEPTH}" \
              --logsig_mode "${MODE}" \
              --logsig_window_size 128 \
              --logsig_smoothing tukey \
              --logsig_stride 7 \
              --epochs_pretrain "${EPOCHS_PRETRAIN}" \
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

wait
echo ""
echo "All 198 missing finetune jobs complete."
echo "Run:  python scripts/aggregate_results.py && python vis/visualize_results.py"
