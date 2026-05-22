#!/bin/bash
# =============================================================================
# run_sweep.sh — unified pretrain + finetune + probe sweep for any dataset pair
#
# Usage:
#   bash run_sweep.sh [pair]                        # run one or all pairs
#   GPUS="0,1" bash run_sweep.sh                    # use GPUs 0 and 1 (sequential, round-robin)
#   GPUS="0,1" PARALLEL=true bash run_sweep.sh      # launch jobs in parallel across GPUs 0,1
#   SKIP_SECTIONS="3view" bash run_sweep.sh         # skip section A
#   GPUS="2" bash run_sweep.sh harth_har70plus      # run one pair on GPU 2
#
# Each pair runs sections in order (some skipped by default):
#   A. 3-view sweep  — xt+dx+xf, xt+logsig+xf, xt+dx+logsig
#                      pretrain → finetune → probe
#   B. 2-view sweep  — xt+logsig only (uses run_{pretrain,finetune}_nview.py)
#                      pretrain → finetune
#   C. Pool ablation — transformer + stream + last-token pooling
#   D. Stride ablation — win128 × stride {1, 7}
#   E. Bilinear sweep — E1: 2v nview mlp_logsig win64/128/tukey128
#                       E2: 3v dx+logsig, logsig+xf mlp_logsig bilinear
#                       E3: 3v dx+xf transformer bilinear
#                       Launch: GPUS="5,6,8" PARALLEL=true \
#                         SKIP_SECTIONS="3view 2view pool_ablation stride" bash run_sweep.sh
#   F. Long pretrain — 200-epoch focused runs on logsig combos
#                      mlp_logsig, window_smooth, win128, bilinear, all seeds
#                      (logsig+xf, dx+logsig, 2-view nview)
#                      Override: F_EP=100 F_ENC=transformer
#                      Launch: SKIP_SECTIONS="3view 2view pool_ablation stride bilinear" bash run_sweep.sh
#   G. Multi-smooth   — multi-param Tukey (0.25,0.5 and 0.25,0.5,0.75) vs single-param baseline
#                      mlp_logsig, window_smooth, win128, bilinear, ep2, all seeds
#                      Baseline already in Section B (sp=0.5, _tukey128_ilbilinear)
#                      Launch: SKIP_SECTIONS="3view 2view pool_ablation stride bilinear long_pretrain" bash run_sweep.sh
#   H. Depth ablation — logsig truncation depth {2,3,4} at fixed best config
#                      mlp_logsig, window_smooth, win128, sp=0.5, bilinear, ep2, all seeds
#                      Override: H_DEPTHS="2 3 4"
#                      Launch: SKIP_SECTIONS="3view 2view pool_ablation stride bilinear long_pretrain multi_smooth" bash run_sweep.sh
#   I. Smooth-param   — Tukey alpha {0.1,0.25,0.5,0.75,0.9} at fixed best config
#                      mlp_logsig, window_smooth, win128, depth=2, bilinear, ep2, all seeds
#                      Override: I_SMOOTH_PARAMS="0.1 0.25 0.5 0.75 0.9"
#                      Baseline (sp=0.5) already in Section B (_tukey128_ilbilinear)
#                      Launch: SKIP_SECTIONS="3view 2view pool_ablation stride bilinear long_pretrain multi_smooth depth" bash run_sweep.sh
#
# Combined G+H+I launch (ablation-only, skips main sweep):
#   SKIP_SECTIONS="3view 2view pool_ablation stride bilinear long_pretrain" bash run_sweep.sh
#
# Set SKIP_STAGES to omit stages, e.g. SKIP_STAGES="pretrain" bash run_sweep.sh
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
IFS=' ' read -ra SEEDS            <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS_PRETRAIN=(2)
EPOCHS_FINETUNE=10

# ---------------------------------------------------------------------------
# Interaction layer variants (Section A only — 3-view scripts)
#   Default: attention only (existing behaviour, no tag change).
#   Full sweep: set env var  INTERACTION_TYPES="attention view_embed bilinear"
# ---------------------------------------------------------------------------
IFS=' ' read -ra INTERACTION_TYPES <<< "${INTERACTION_TYPES:-bilinear}"

# ---------------------------------------------------------------------------
# 3-view combos
# ---------------------------------------------------------------------------
VIEW2S=("dx" "logsig" "dx")
VIEW3S=("xf"  "xf"   "logsig")
IFS=' ' read -ra ENCODER_TYPES <<< "${ENCODER_TYPES:-transformer mlp_logsig}"

# ---------------------------------------------------------------------------
# Logsig configuration
#   LOGSIG_MODES:        which modes to run
#   LOGSIG_WINDOW_SIZES: window sizes for window/window_smooth modes
#   LOGSIG_SMOOTHING:    smoothing used for window_smooth mode
# ---------------------------------------------------------------------------
IFS=' ' read -ra LOGSIG_MODES        <<< "${LOGSIG_MODES:-stream window window_smooth}"
IFS=' ' read -ra LOGSIG_WINDOW_SIZES <<< "${LOGSIG_WINDOW_SIZES:-64 128}"
LOGSIG_SMOOTHING="tukey"
IFS=' ' read -ra LOGSIG_DEPTHS        <<< "${LOGSIG_DEPTHS:-2}"
IFS=' ' read -ra LOGSIG_SMOOTH_PARAMS <<< "${LOGSIG_SMOOTH_PARAMS:-0.5}"
LOGSIG_DEPTH="${LOGSIG_DEPTHS[0]}"   # scalar alias used by ablation sections C/D/E

FEATURE="hidden"
LOSS_TYPE="ALL"
LAM="0.0"
RUN_MODES="finetune"   # set to "finetune,freeze,baseline" to re-enable all three

PARALLEL="${PARALLEL:-false}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"   # max concurrent jobs in PARALLEL mode; 0 = unlimited
DEFAULT_BATCH=256
LOGSIG_BATCH=128
DISABLE_TQDM=0

SKIP_STAGES="${SKIP_STAGES:-}"     # space-separated: "pretrain" "finetune" "probe"
SKIP_SECTIONS="${SKIP_SECTIONS:-3view pool_ablation stride bilinear long_pretrain multi_smooth depth smooth_param}" # default: 2view only
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

# GPU assignment: GPUS="0,1,2,3" bash run_sweep.sh  (default: GPU 0)
# Round-robins across the list for both parallel and sequential modes.
_gpus_raw="${GPUS:-0}"
IFS=',' read -ra GPU_LIST <<< "$_gpus_raw"
num_gpus=${#GPU_LIST[@]}
echo "GPUs: [${GPU_LIST[*]}]  (${num_gpus} total)"

# Wait for a GPU to have no active (live) compute processes before starting a job.
# Ignores ghost PIDs that nvidia-smi still reports after a process was killed.
_gpu_has_live_procs() {
  local gid="$1"
  local pids
  pids=$(nvidia-smi -i "$gid" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$pids" ] && return 1   # no PIDs at all → free
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null && return 0   # at least one live PID → busy
  done
  return 1  # all PIDs are ghosts → treat as free
}

_wait_for_gpu() {
  local gid="$1"
  if _gpu_has_live_procs "$gid"; then
    echo "  GPU ${gid} busy — waiting for it to free up..."
    while _gpu_has_live_procs "$gid"; do
      sleep 15
    done
    sleep 5  # brief grace period for memory release
  fi
}

_launch() {
  local tag="$1"; local log="$2"; shift 2; local cmd=("$@")
  local gid="${GPU_LIST[$(( job_index % num_gpus ))]}"
  _wait_for_gpu "$gid"
  echo "Running [GPU ${gid}]: ${tag}"
  TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" 2>&1 | tee "$log"
  job_index=$(( job_index + 1 ))
}

# Like _launch but skips immediately (no Python startup) if OUT_FILE already exists.
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

# Build the interaction-layer suffix that matches _il_suffix in the Python scripts.
#   attention → ""   view_embed → "_ilviewembed"   bilinear → "_ilbilinear"
_il_suffix() {
  local il_type="$1"
  [ "$il_type" = "attention" ] && echo "" || echo "_il${il_type//_/}"
}

# Build the logsig checkpoint suffix that matches _logsig_suffix() in the Python scripts.
#   stream        → ""
#   window W      → "_winW"
#   window_smooth → "_tukeyW" / "_emaW"
#   stride S>1    → append "_sS"
#   global_time   → append "_gt"
#   smooth_param  → append "_sp{val}" when != 0.5 (window_smooth only)
#   depth         → append "_d{N}" when != 2
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
                        "${IL_ARGS[@]}"
                  done
                done
              done
            done
          done
        done
      done
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

    IFS=' ' read -ra B_IL_TYPES <<< "${B_INTERACTION_TYPES:-${INTERACTION_TYPES[*]}}"
    for IL_TYPE_B in "${B_IL_TYPES[@]}"; do
    IL_SFXR_B="$(_il_suffix "$IL_TYPE_B")"
    IL_ARGS_B=(); [ "$IL_TYPE_B" != "attention" ] && IL_ARGS_B=(--interaction_type "${IL_TYPE_B}")
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
            # stride=1 is the baseline; stride>1 only makes sense for window modes
            if [ "$MODE" = "stream" ]; then
              STRIDES_2V=(1)
            else
              STRIDES_2V=(1 7)
            fi
            for STRIDE_2V in "${STRIDES_2V[@]}"; do
            SP="0.5"; DEPTH=2
            LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING" "$STRIDE_2V" "0" "$SP" "$DEPTH")"
            for SEED in "${SEEDS[@]}"; do

              # ------ 2-view pretrain ------
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

              # ------ 2-view finetune ------
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
                      "${IL_ARGS_B[@]}"
                fi
              fi

            done
            done  # STRIDE_2V
          done
        done
      done
    done
    done  # IL_TYPE_B

  fi  # end section B (2-view)

  # =========================================================================
  # SECTION C: Pooling ablation — transformer + stream + last-token
  #
  # Compares transformer+mean (auto, already in section A) vs
  # transformer+last-token on global/stream logsig, to isolate whether
  # the transformer's advantage over MLP-LogSig comes from the attention
  # mechanism or from mean-pooling vs last-token.
  # Skip with: SKIP_SECTIONS="pool_ablation" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section pool_ablation; then

    echo ""
    echo "--- [pool ablation] transformer + stream + last-token ---"

    for EP in "${EPOCHS_PRETRAIN[@]}"; do
      for i in "${!VIEW2S[@]}"; do
        V2="${VIEW2S[$i]}"; V3="${VIEW3S[$i]}"
        [ "$V2" != "logsig" ] && [ "$V3" != "logsig" ] && continue
        BATCH=$LOGSIG_BATCH
        for SEED in "${SEEDS[@]}"; do
          TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}_plast"

          if ! _skip_stage pretrain; then
            _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
              "$TAG" "logs/${TAG}.log" \
              python -u run_pretrain.py \
                --data_name "${PRETRAIN_DATA}" \
                --num_feature "${PRETRAIN_NUM_FEATURE}" \
                --num_target  "${PRETRAIN_NUM_TARGET}" \
                --view2 "${V2}" --view3 "${V3}" \
                --encoder_type transformer \
                --batch_size_pretrain "${BATCH}" \
                --epochs_pretrain "${EP}" \
                --logsig_depth "${LOGSIG_DEPTH}" \
                --logsig_mode stream \
                --logsig_pool last \
                --seed "${SEED}"
          fi

          if ! _skip_stage finetune; then
            PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${TAG}.pth"
            if [ ! -f "$PT_CKPT" ]; then
              echo "Skipping ablation finetune — missing: ${PT_CKPT}"
            else
              FT_TAG="ft_${FINETUNE_DATA}_from_${TAG}"
              _launch "$FT_TAG" "logs/${FT_TAG}.log" \
                python run_finetune.py \
                  --data_name "${FINETUNE_DATA}" \
                  --pretrain_data_name "${PRETRAIN_DATA}" \
                  --num_feature "${FINETUNE_NUM_FEATURE}" \
                  --num_target  "${FINETUNE_NUM_TARGET}" \
                  --view2 "${V2}" --view3 "${V3}" \
                  --encoder_type transformer \
                  --logsig_depth "${LOGSIG_DEPTH}" \
                  --logsig_mode stream \
                  --logsig_pool last \
                  --epochs_pretrain "${EP}" \
                  --epochs_finetune "${EPOCHS_FINETUNE}" \
                  --feature "${FEATURE}" \
                  --loss_type "${LOSS_TYPE}" \
                  --lam "${LAM}" \
                  --run_modes "${RUN_MODES}" \
                  --seed "${SEED}"
            fi
          fi

        done
      done
    done

    # ------ 2-view (nview) plast ablation ------
    echo ""
    echo "--- [pool ablation] 2-view nview + stream + last-token ---"

    for EP in "${EPOCHS_PRETRAIN[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}_plast"

        if ! _skip_stage pretrain; then
          _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
            "$TAG" "logs/${TAG}.log" \
            python -u run_pretrain_nview.py \
              --data_name "${PRETRAIN_DATA}" \
              --num_feature "${PRETRAIN_NUM_FEATURE}" \
              --num_target  "${PRETRAIN_NUM_TARGET}" \
              --view2 logsig \
              --encoder_type transformer \
              --batch_size_pretrain "${LOGSIG_BATCH}" \
              --epochs_pretrain "${EP}" \
              --logsig_depth "${LOGSIG_DEPTH}" \
              --logsig_mode stream \
              --logsig_pool last \
              --seed "${SEED}"
        fi

        if ! _skip_stage finetune; then
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${TAG}.pth"
          if [ ! -f "$PT_CKPT" ]; then
            echo "Skipping nview ablation finetune — missing: ${PT_CKPT}"
          else
            FT_TAG="ft2v_${FINETUNE_DATA}_from_${TAG}"
            _launch "$FT_TAG" "logs/${FT_TAG}.log" \
              python run_finetune_nview.py \
                --data_name "${FINETUNE_DATA}" \
                --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature "${FINETUNE_NUM_FEATURE}" \
                --num_target  "${FINETUNE_NUM_TARGET}" \
                --view2 logsig \
                --encoder_type transformer \
                --logsig_depth "${LOGSIG_DEPTH}" \
                --logsig_mode stream \
                --logsig_pool last \
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

  fi  # end section C (pool ablation)

  # =========================================================================
  # SECTION E: Bilinear attention sweep
  #
  # Runs bilinear interaction layer on focused combos:
  #   E1. 2-view xt+logsig (nview):   mlp_logsig, win64 / win128 / tukey128
  #   E2. 3-view dx+logsig, logsig+xf: mlp_logsig, win64 / win128 / tukey128, bilinear IL
  #   E3. 3-view dx+xf:                transformer, bilinear IL
  # All: seeds 0-9, ep2 pretrain, ep10 finetune, both dataset pairs.
  #
  # Invoke with:
  #   GPUS="5,6,8" PARALLEL=true SKIP_SECTIONS="3view 2view pool_ablation stride" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section bilinear; then

    echo ""
    echo "--- [bilinear] Section E ---"

    BIL_EP=2
    BIL_ENC="mlp_logsig"
    BIL_ENC_SUFFIX="_mlp_logsig"
    # Ordered: window64, window128, window_smooth(tukey)128
    BIL_MODES_WSIZES=("window:64" "window:128" "window_smooth:128")
    IL_BILINEAR=(--interaction_type bilinear)
    IL_SFXR_BILINEAR="_ilbilinear"
    BIL_3VIEW_COMBOS=("dx:logsig" "logsig:xf")

    # --- E pretrain pass (all sub-sections) ---
    if ! _skip_stage pretrain; then
      echo "  [E pretrain] E1/E2/E3"

      # E1: 2-view nview
      for MW in "${BIL_MODES_WSIZES[@]}"; do
        BIL_MODE="${MW%%:*}"; BIL_WSIZ="${MW##*:}"
        LS="$(_lsig_suffix "$BIL_MODE" "$BIL_WSIZ" "$LOGSIG_SMOOTHING")"
        for SEED in "${SEEDS[@]}"; do
          TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${BIL_EP}_${SEED}${BIL_ENC_SUFFIX}${LS}"
          _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
            "$TAG" "logs/${TAG}.log" \
            python -u run_pretrain_nview.py \
              --data_name "${PRETRAIN_DATA}" \
              --num_feature "${PRETRAIN_NUM_FEATURE}" \
              --num_target  "${PRETRAIN_NUM_TARGET}" \
              --view2 logsig \
              --encoder_type "${BIL_ENC}" \
              --batch_size_pretrain "${LOGSIG_BATCH}" \
              --epochs_pretrain "${BIL_EP}" \
              --logsig_depth "${LOGSIG_DEPTH}" \
              --logsig_mode "${BIL_MODE}" \
              --logsig_window_size "${BIL_WSIZ}" \
              --logsig_smoothing "${LOGSIG_SMOOTHING}" \
              --seed "${SEED}"
        done
      done

      # E2: 3-view dx+logsig and logsig+xf with bilinear
      for COMBO in "${BIL_3VIEW_COMBOS[@]}"; do
        V2="${COMBO%%:*}"; V3="${COMBO##*:}"
        for MW in "${BIL_MODES_WSIZES[@]}"; do
          BIL_MODE="${MW%%:*}"; BIL_WSIZ="${MW##*:}"
          LS="$(_lsig_suffix "$BIL_MODE" "$BIL_WSIZ" "$LOGSIG_SMOOTHING")"
          for SEED in "${SEEDS[@]}"; do
            TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${BIL_EP}_${SEED}${BIL_ENC_SUFFIX}${LS}${IL_SFXR_BILINEAR}"
            _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
              "$TAG" "logs/${TAG}.log" \
              python -u run_pretrain.py \
                --data_name "${PRETRAIN_DATA}" \
                --num_feature "${PRETRAIN_NUM_FEATURE}" \
                --num_target  "${PRETRAIN_NUM_TARGET}" \
                --view2 "${V2}" --view3 "${V3}" \
                --encoder_type "${BIL_ENC}" \
                --batch_size_pretrain "${LOGSIG_BATCH}" \
                --epochs_pretrain "${BIL_EP}" \
                --logsig_depth "${LOGSIG_DEPTH}" \
                --logsig_mode "${BIL_MODE}" \
                --logsig_window_size "${BIL_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --seed "${SEED}" \
                "${IL_BILINEAR[@]}"
          done
        done
      done

      # E3: 3-view dx+xf transformer bilinear
      for SEED in "${SEEDS[@]}"; do
        TAG="${PRETRAIN_DATA}_v2dx_v3xf_ep${BIL_EP}_${SEED}${IL_SFXR_BILINEAR}"
        _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
          "$TAG" "logs/${TAG}.log" \
          python -u run_pretrain.py \
            --data_name "${PRETRAIN_DATA}" \
            --num_feature "${PRETRAIN_NUM_FEATURE}" \
            --num_target  "${PRETRAIN_NUM_TARGET}" \
            --view2 dx --view3 xf \
            --encoder_type transformer \
            --batch_size_pretrain "${DEFAULT_BATCH}" \
            --epochs_pretrain "${BIL_EP}" \
            --logsig_depth "${LOGSIG_DEPTH}" \
            --seed "${SEED}" \
            "${IL_BILINEAR[@]}"
      done

    fi

    # --- E finetune pass (all sub-sections) ---
    if ! _skip_stage finetune; then
      echo "  [E finetune] E1/E2/E3"

      # E1: 2-view nview
      for MW in "${BIL_MODES_WSIZES[@]}"; do
        BIL_MODE="${MW%%:*}"; BIL_WSIZ="${MW##*:}"
        LS="$(_lsig_suffix "$BIL_MODE" "$BIL_WSIZ" "$LOGSIG_SMOOTHING")"
        for SEED in "${SEEDS[@]}"; do
          PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${BIL_EP}_${SEED}${BIL_ENC_SUFFIX}${LS}"
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
          FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
          if [ -f "$FT_OUT" ]; then
            echo "Skip finetune (exists): $(basename "$FT_OUT")"
          elif [ ! -f "$PT_CKPT" ]; then
            echo "Skipping E1 finetune — missing: ${PT_CKPT}"
          else
            FT_TAG="ft2v_${FINETUNE_DATA}_from_${PT_TAG}"
            _launch "$FT_TAG" "logs/${FT_TAG}.log" \
              python run_finetune_nview.py \
                --data_name "${FINETUNE_DATA}" \
                --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature "${FINETUNE_NUM_FEATURE}" \
                --num_target  "${FINETUNE_NUM_TARGET}" \
                --view2 logsig \
                --encoder_type "${BIL_ENC}" \
                --logsig_depth "${LOGSIG_DEPTH}" \
                --logsig_mode "${BIL_MODE}" \
                --logsig_window_size "${BIL_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --epochs_pretrain "${BIL_EP}" \
                --epochs_finetune "${EPOCHS_FINETUNE}" \
                --feature "${FEATURE}" \
                --lam "${LAM}" \
                --run_modes "${RUN_MODES}" \
                --seed "${SEED}"
          fi
        done
      done

      # E2: 3-view dx+logsig and logsig+xf with bilinear
      for COMBO in "${BIL_3VIEW_COMBOS[@]}"; do
        V2="${COMBO%%:*}"; V3="${COMBO##*:}"
        for MW in "${BIL_MODES_WSIZES[@]}"; do
          BIL_MODE="${MW%%:*}"; BIL_WSIZ="${MW##*:}"
          LS="$(_lsig_suffix "$BIL_MODE" "$BIL_WSIZ" "$LOGSIG_SMOOTHING")"
          for SEED in "${SEEDS[@]}"; do
            PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${BIL_EP}_${SEED}${BIL_ENC_SUFFIX}${LS}${IL_SFXR_BILINEAR}"
            PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
            if [ ! -f "$PT_CKPT" ]; then
              echo "Skipping E2 finetune — missing: ${PT_CKPT}"
            else
              FT_TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
              _launch "$FT_TAG" "logs/${FT_TAG}.log" \
                python run_finetune.py \
                  --data_name "${FINETUNE_DATA}" \
                  --pretrain_data_name "${PRETRAIN_DATA}" \
                  --num_feature "${FINETUNE_NUM_FEATURE}" \
                  --num_target  "${FINETUNE_NUM_TARGET}" \
                  --view2 "${V2}" --view3 "${V3}" \
                  --encoder_type "${BIL_ENC}" \
                  --logsig_depth "${LOGSIG_DEPTH}" \
                  --logsig_mode "${BIL_MODE}" \
                  --logsig_window_size "${BIL_WSIZ}" \
                  --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                  --epochs_pretrain "${BIL_EP}" \
                  --epochs_finetune "${EPOCHS_FINETUNE}" \
                  --feature "${FEATURE}" \
                  --loss_type "${LOSS_TYPE}" \
                  --lam "${LAM}" \
                  --run_modes "${RUN_MODES}" \
                  --seed "${SEED}" \
                  "${IL_BILINEAR[@]}"
            fi
          done
        done
      done

      # E3: 3-view dx+xf transformer bilinear
      for SEED in "${SEEDS[@]}"; do
        PT_TAG="${PRETRAIN_DATA}_v2dx_v3xf_ep${BIL_EP}_${SEED}${IL_SFXR_BILINEAR}"
        PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
        if [ ! -f "$PT_CKPT" ]; then
          echo "Skipping E3 finetune — missing: ${PT_CKPT}"
        else
          FT_TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
          _launch "$FT_TAG" "logs/${FT_TAG}.log" \
            python run_finetune.py \
              --data_name "${FINETUNE_DATA}" \
              --pretrain_data_name "${PRETRAIN_DATA}" \
              --num_feature "${FINETUNE_NUM_FEATURE}" \
              --num_target  "${FINETUNE_NUM_TARGET}" \
              --view2 dx --view3 xf \
              --encoder_type transformer \
              --epochs_pretrain "${BIL_EP}" \
              --epochs_finetune "${EPOCHS_FINETUNE}" \
              --feature "${FEATURE}" \
              --loss_type "${LOSS_TYPE}" \
              --lam "${LAM}" \
              --run_modes "${RUN_MODES}" \
              --seed "${SEED}" \
              "${IL_BILINEAR[@]}"
        fi
      done

    fi

  fi  # end section E (bilinear)

  # =========================================================================
  # SECTION D: stride ablation — win128 × stride {1, 7}
  #
  # Tests overlapping windows: stride=1 (every step, dense) vs stride=7
  # (~18% step fraction on win128).  win128+stride=1 already exists from
  # section A so those runs will be skipped; only stride=7 is new.
  # Skip with: SKIP_SECTIONS="stride" bash run_sweep.sh
  # Run only:  SKIP_SECTIONS="3view 2view pool_ablation" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section stride; then

    echo ""
    echo "--- [stride ablation] win128 × stride 1,7 ---"

    for ENC in "${ENCODER_TYPES[@]}"; do
      ENC_SUFFIX=""; [ "$ENC" != "transformer" ] && ENC_SUFFIX="_${ENC}"
      for EP in "${EPOCHS_PRETRAIN[@]}"; do
        for i in "${!VIEW2S[@]}"; do
          V2="${VIEW2S[$i]}"; V3="${VIEW3S[$i]}"
          [ "$V2" != "logsig" ] && [ "$V3" != "logsig" ] && continue
          [ "$ENC" = "mlp_logsig" ] && [ "$V2" != "logsig" ] && [ "$V3" != "logsig" ] && continue
          for MODE in window window_smooth; do
            for STRIDE in 1 7; do
              LS="$(_lsig_suffix "$MODE" "128" "$LOGSIG_SMOOTHING" "$STRIDE")"
              for SEED in "${SEEDS[@]}"; do

                if ! _skip_stage pretrain; then
                  TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
                  _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
                    "$TAG" "logs/${TAG}.log" \
                    python -u run_pretrain.py \
                      --data_name "${PRETRAIN_DATA}" \
                      --num_feature "${PRETRAIN_NUM_FEATURE}" \
                      --num_target  "${PRETRAIN_NUM_TARGET}" \
                      --view2 "${V2}" --view3 "${V3}" \
                      --encoder_type "${ENC}" \
                      --batch_size_pretrain "${LOGSIG_BATCH}" \
                      --epochs_pretrain "${EP}" \
                      --logsig_depth "${LOGSIG_DEPTH}" \
                      --logsig_mode "${MODE}" \
                      --logsig_window_size 128 \
                      --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                      --logsig_stride "${STRIDE}" \
                      --seed "${SEED}"
                fi

                if ! _skip_stage finetune; then
                  TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
                  PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${TAG}.pth"
                  if [ ! -f "$PT_CKPT" ]; then
                    echo "Skipping stride finetune — missing: ${PT_CKPT}"
                  else
                    FT_TAG="ft_${FINETUNE_DATA}_from_${TAG}"
                    _launch "$FT_TAG" "logs/${FT_TAG}.log" \
                      python run_finetune.py \
                        --data_name "${FINETUNE_DATA}" \
                        --pretrain_data_name "${PRETRAIN_DATA}" \
                        --num_feature "${FINETUNE_NUM_FEATURE}" \
                        --num_target  "${FINETUNE_NUM_TARGET}" \
                        --view2 "${V2}" --view3 "${V3}" \
                        --encoder_type "${ENC}" \
                        --logsig_depth "${LOGSIG_DEPTH}" \
                        --logsig_mode "${MODE}" \
                        --logsig_window_size 128 \
                        --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                        --logsig_stride "${STRIDE}" \
                        --epochs_pretrain "${EP}" \
                        --epochs_finetune "${EPOCHS_FINETUNE}" \
                        --feature "${FEATURE}" \
                        --loss_type "${LOSS_TYPE}" \
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
    done
  fi

  # =========================================================================
  # SECTION F: 200-epoch focused runs
  #
  # Long pretrain for the most promising logsig configs (excluding dx+xf).
  # Hardcoded to: mlp_logsig encoder, window_smooth mode, win128, tukey=0.5,
  # depth=2, for all three logsig view-settings.
  # Override with env vars, e.g.: F_EP=100 F_ENC=transformer bash run_sweep.sh
  #
  # Invoke: SKIP_SECTIONS="3view 2view pool_ablation stride bilinear" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section long_pretrain; then

    echo ""
    echo "--- [Section F] 200-epoch focused runs ---"

    F_EP="${F_EP:-200}"
    F_ENC="${F_ENC:-mlp_logsig}"
    F_ENC_SUFFIX=""; [ "$F_ENC" != "transformer" ] && F_ENC_SUFFIX="_${F_ENC}"
    F_MODE="window_smooth"
    F_WSIZ=128
    F_SP=0.5
    F_DEPTH=2
    F_LS="$(_lsig_suffix "$F_MODE" "$F_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$F_SP" "$F_DEPTH")"
    F_IL_SFXR="_ilbilinear"
    F_IL_ARGS=(--interaction_type bilinear)

    # View combos: xt+logsig+xf, xt+dx+logsig, and 2-view xt+logsig (nview)
    F_3VIEW_COMBOS=("logsig:xf" "dx:logsig")

    if ! _skip_stage pretrain; then
      echo "  [F pretrain]"

      # 3-view combos
      for COMBO in "${F_3VIEW_COMBOS[@]}"; do
        V2="${COMBO%%:*}"; V3="${COMBO##*:}"
        for SEED in "${SEEDS[@]}"; do
          TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${F_EP}_${SEED}${F_ENC_SUFFIX}${F_LS}${F_IL_SFXR}"
          _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
            "$TAG" "logs/${TAG}.log" \
            python -u run_pretrain.py \
              --data_name "${PRETRAIN_DATA}" \
              --num_feature "${PRETRAIN_NUM_FEATURE}" \
              --num_target  "${PRETRAIN_NUM_TARGET}" \
              --view2 "${V2}" --view3 "${V3}" \
              --encoder_type "${F_ENC}" \
              --batch_size_pretrain "${LOGSIG_BATCH}" \
              --epochs_pretrain "${F_EP}" \
              --logsig_depth "${F_DEPTH}" \
              --logsig_mode "${F_MODE}" \
              --logsig_window_size "${F_WSIZ}" \
              --logsig_smoothing "${LOGSIG_SMOOTHING}" \
              --logsig_smooth_param "${F_SP}" \
              --seed "${SEED}" \
              "${F_IL_ARGS[@]}"
        done
      done

      # 2-view nview
      for SEED in "${SEEDS[@]}"; do
        TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${F_EP}_${SEED}${F_ENC_SUFFIX}${F_LS}${F_IL_SFXR}"
        _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
          "$TAG" "logs/${TAG}.log" \
          python -u run_pretrain_nview.py \
            --data_name "${PRETRAIN_DATA}" \
            --num_feature "${PRETRAIN_NUM_FEATURE}" \
            --num_target  "${PRETRAIN_NUM_TARGET}" \
            --view2 logsig \
            --encoder_type "${F_ENC}" \
            --batch_size_pretrain "${LOGSIG_BATCH}" \
            --epochs_pretrain "${F_EP}" \
            --logsig_depth "${F_DEPTH}" \
            --logsig_mode "${F_MODE}" \
            --logsig_window_size "${F_WSIZ}" \
            --logsig_smoothing "${LOGSIG_SMOOTHING}" \
            --logsig_smooth_param "${F_SP}" \
            --seed "${SEED}" \
            "${F_IL_ARGS[@]}"
      done

    fi

    if ! _skip_stage finetune; then
      echo "  [F finetune]"

      # 3-view combos
      for COMBO in "${F_3VIEW_COMBOS[@]}"; do
        V2="${COMBO%%:*}"; V3="${COMBO##*:}"
        for SEED in "${SEEDS[@]}"; do
          PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${F_EP}_${SEED}${F_ENC_SUFFIX}${F_LS}${F_IL_SFXR}"
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
          if [ ! -f "$PT_CKPT" ]; then
            echo "Skipping F finetune — missing: ${PT_CKPT}"
          else
            FT_TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
            _launch "$FT_TAG" "logs/${FT_TAG}.log" \
              python run_finetune.py \
                --data_name "${FINETUNE_DATA}" \
                --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature "${FINETUNE_NUM_FEATURE}" \
                --num_target  "${FINETUNE_NUM_TARGET}" \
                --view2 "${V2}" --view3 "${V3}" \
                --encoder_type "${F_ENC}" \
                --logsig_depth "${F_DEPTH}" \
                --logsig_mode "${F_MODE}" \
                --logsig_window_size "${F_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_smooth_param "${F_SP}" \
                --epochs_pretrain "${F_EP}" \
                --epochs_finetune "${EPOCHS_FINETUNE}" \
                --feature "${FEATURE}" \
                --loss_type "${LOSS_TYPE}" \
                --lam "${LAM}" \
                --run_modes "${RUN_MODES}" \
                --seed "${SEED}" \
                "${F_IL_ARGS[@]}"
          fi
        done
      done

      # 2-view nview
      for SEED in "${SEEDS[@]}"; do
        PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${F_EP}_${SEED}${F_ENC_SUFFIX}${F_LS}${F_IL_SFXR}"
        PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
        FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
        if [ -f "$FT_OUT" ]; then
          echo "Skip finetune (exists): $(basename "$FT_OUT")"
        elif [ ! -f "$PT_CKPT" ]; then
          echo "Skipping F 2-view finetune — missing: ${PT_CKPT}"
        else
          FT_TAG="ft2v_${FINETUNE_DATA}_from_${PT_TAG}"
          _launch "$FT_TAG" "logs/${FT_TAG}.log" \
            python run_finetune_nview.py \
              --data_name "${FINETUNE_DATA}" \
              --pretrain_data_name "${PRETRAIN_DATA}" \
              --num_feature "${FINETUNE_NUM_FEATURE}" \
              --num_target  "${FINETUNE_NUM_TARGET}" \
              --view2 logsig \
              --encoder_type "${F_ENC}" \
              --logsig_depth "${F_DEPTH}" \
              --logsig_mode "${F_MODE}" \
              --logsig_window_size "${F_WSIZ}" \
              --logsig_smoothing "${LOGSIG_SMOOTHING}" \
              --logsig_smooth_param "${F_SP}" \
              --epochs_pretrain "${F_EP}" \
              --epochs_finetune "${EPOCHS_FINETUNE}" \
              --feature "${FEATURE}" \
              --lam "${LAM}" \
              --run_modes "${RUN_MODES}" \
              --seed "${SEED}" \
              "${F_IL_ARGS[@]}"
        fi
      done

    fi

  fi  # end section F (long pretrain)

  # =========================================================================
  # SECTION G: Multi-param Tukey sweep (Q2)
  #
  # Tests combining multiple Tukey-smoothed log signatures simultaneously.
  # Compares single-param baseline (sp=0.5, from Section B) with multi-param
  # variants: 2 params (0.25,0.5) and 3 params (0.25,0.5,0.75).
  # All: mlp_logsig, window_smooth, win128, bilinear, ep2 pretrain, ep10 finetune.
  # View combos: xt+logsig+xf, xt+dx+logsig, 2-view nview.
  # Single-param baseline for comparison already produced by Section B.
  #
  # Invoke: SKIP_SECTIONS="3view 2view pool_ablation stride bilinear long_pretrain" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section multi_smooth; then

    echo ""
    echo "--- [Section G] Multi-param Tukey sweep ---"

    G_EP=2
    G_ENC="mlp_logsig"
    G_ENC_SUFFIX="_mlp_logsig"
    G_MODE="window_smooth"
    G_WSIZ=128
    G_DEPTH=2
    G_IL_SFXR="_ilbilinear"
    G_IL_ARGS=(--interaction_type bilinear)
    G_3VIEW_COMBOS=("logsig:xf" "dx:logsig")
    G_MSP_LIST=("0.25,0.5" "0.25,0.5,0.75")

    if ! _skip_stage pretrain; then
      echo "  [G pretrain]"

      for G_MSP in "${G_MSP_LIST[@]}"; do
        G_MSP_K=$(echo "$G_MSP" | tr -cd ',' | wc -c); G_MSP_K=$((G_MSP_K + 1))
        G_LS="$(_lsig_suffix "$G_MODE" "$G_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "0.5" "$G_DEPTH" "$G_MSP_K")"

        for COMBO in "${G_3VIEW_COMBOS[@]}"; do
          V2="${COMBO%%:*}"; V3="${COMBO##*:}"
          for SEED in "${SEEDS[@]}"; do
            TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${G_EP}_${SEED}${G_ENC_SUFFIX}${G_LS}${G_IL_SFXR}"
            _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
              "$TAG" "logs/${TAG}.log" \
              python -u run_pretrain.py \
                --data_name "${PRETRAIN_DATA}" \
                --num_feature "${PRETRAIN_NUM_FEATURE}" \
                --num_target  "${PRETRAIN_NUM_TARGET}" \
                --view2 "${V2}" --view3 "${V3}" \
                --encoder_type "${G_ENC}" \
                --batch_size_pretrain "${LOGSIG_BATCH}" \
                --epochs_pretrain "${G_EP}" \
                --logsig_depth "${G_DEPTH}" \
                --logsig_mode "${G_MODE}" \
                --logsig_window_size "${G_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_multi_smooth_params "${G_MSP}" \
                --seed "${SEED}" \
                "${G_IL_ARGS[@]}"
          done
        done

        for SEED in "${SEEDS[@]}"; do
          TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${G_EP}_${SEED}${G_ENC_SUFFIX}${G_LS}${G_IL_SFXR}"
          _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
            "$TAG" "logs/${TAG}.log" \
            python -u run_pretrain_nview.py \
              --data_name "${PRETRAIN_DATA}" \
              --num_feature "${PRETRAIN_NUM_FEATURE}" \
              --num_target  "${PRETRAIN_NUM_TARGET}" \
              --view2 logsig \
              --encoder_type "${G_ENC}" \
              --batch_size_pretrain "${LOGSIG_BATCH}" \
              --epochs_pretrain "${G_EP}" \
              --logsig_depth "${G_DEPTH}" \
              --logsig_mode "${G_MODE}" \
              --logsig_window_size "${G_WSIZ}" \
              --logsig_smoothing "${LOGSIG_SMOOTHING}" \
              --logsig_multi_smooth_params "${G_MSP}" \
              --seed "${SEED}" \
              "${G_IL_ARGS[@]}"
        done

      done  # G_MSP

    fi

    if ! _skip_stage finetune; then
      echo "  [G finetune]"

      for G_MSP in "${G_MSP_LIST[@]}"; do
        G_MSP_K=$(echo "$G_MSP" | tr -cd ',' | wc -c); G_MSP_K=$((G_MSP_K + 1))
        G_LS="$(_lsig_suffix "$G_MODE" "$G_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "0.5" "$G_DEPTH" "$G_MSP_K")"

        for COMBO in "${G_3VIEW_COMBOS[@]}"; do
          V2="${COMBO%%:*}"; V3="${COMBO##*:}"
          for SEED in "${SEEDS[@]}"; do
            PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${G_EP}_${SEED}${G_ENC_SUFFIX}${G_LS}${G_IL_SFXR}"
            PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
            if [ ! -f "$PT_CKPT" ]; then
              echo "Skipping G finetune — missing: ${PT_CKPT}"
            else
              FT_TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
              _launch "$FT_TAG" "logs/${FT_TAG}.log" \
                python run_finetune.py \
                  --data_name "${FINETUNE_DATA}" \
                  --pretrain_data_name "${PRETRAIN_DATA}" \
                  --num_feature "${FINETUNE_NUM_FEATURE}" \
                  --num_target  "${FINETUNE_NUM_TARGET}" \
                  --view2 "${V2}" --view3 "${V3}" \
                  --encoder_type "${G_ENC}" \
                  --logsig_depth "${G_DEPTH}" \
                  --logsig_mode "${G_MODE}" \
                  --logsig_window_size "${G_WSIZ}" \
                  --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                  --logsig_multi_smooth_params "${G_MSP}" \
                  --epochs_pretrain "${G_EP}" \
                  --epochs_finetune "${EPOCHS_FINETUNE}" \
                  --feature "${FEATURE}" \
                  --loss_type "${LOSS_TYPE}" \
                  --lam "${LAM}" \
                  --run_modes "${RUN_MODES}" \
                  --seed "${SEED}" \
                  "${G_IL_ARGS[@]}"
            fi
          done
        done

        for SEED in "${SEEDS[@]}"; do
          PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${G_EP}_${SEED}${G_ENC_SUFFIX}${G_LS}${G_IL_SFXR}"
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
          FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
          if [ -f "$FT_OUT" ]; then
            echo "Skip finetune (exists): $(basename "$FT_OUT")"
          elif [ ! -f "$PT_CKPT" ]; then
            echo "Skipping G 2-view finetune — missing: ${PT_CKPT}"
          else
            FT_TAG="ft2v_${FINETUNE_DATA}_from_${PT_TAG}"
            _launch "$FT_TAG" "logs/${FT_TAG}.log" \
              python run_finetune_nview.py \
                --data_name "${FINETUNE_DATA}" \
                --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature "${FINETUNE_NUM_FEATURE}" \
                --num_target  "${FINETUNE_NUM_TARGET}" \
                --view2 logsig \
                --encoder_type "${G_ENC}" \
                --logsig_depth "${G_DEPTH}" \
                --logsig_mode "${G_MODE}" \
                --logsig_window_size "${G_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_multi_smooth_params "${G_MSP}" \
                --epochs_pretrain "${G_EP}" \
                --epochs_finetune "${EPOCHS_FINETUNE}" \
                --feature "${FEATURE}" \
                --lam "${LAM}" \
                --run_modes "${RUN_MODES}" \
                --seed "${SEED}" \
                "${G_IL_ARGS[@]}"
          fi
        done

      done  # G_MSP

    fi

  fi  # end section G (multi-param Tukey)

  # =========================================================================
  # SECTION H: Depth ablation — logsig truncation depth {2,3,4}
  #
  # Fixed config: mlp_logsig, window_smooth, win128, sp=0.5, bilinear IL, ep2.
  # Runs nview (xt+logsig) and 3-view combos (logsig+xf, dx+logsig).
  # Override depths: H_DEPTHS="2 3 4" bash run_sweep.sh
  # Skip with: SKIP_SECTIONS="depth" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section depth; then

    echo ""
    echo "--- [Section H] Depth ablation ---"

    H_EP=2
    H_ENC="mlp_logsig"
    H_ENC_SUFFIX="_mlp_logsig"
    H_MODE="window_smooth"
    H_WSIZ=128
    H_SP="0.5"
    H_IL_SFXR="_ilbilinear"
    H_IL_ARGS=(--interaction_type bilinear)
    H_3VIEW_COMBOS=("logsig:xf" "dx:logsig")
    IFS=' ' read -ra H_DEPTHS <<< "${H_DEPTHS:-2 3 4}"

    if ! _skip_stage pretrain; then
      echo "  [H pretrain]"

      for H_DEPTH in "${H_DEPTHS[@]}"; do
        H_LS="$(_lsig_suffix "$H_MODE" "$H_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$H_SP" "$H_DEPTH")"

        for COMBO in "${H_3VIEW_COMBOS[@]}"; do
          V2="${COMBO%%:*}"; V3="${COMBO##*:}"
          for SEED in "${SEEDS[@]}"; do
            TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${H_EP}_${SEED}${H_ENC_SUFFIX}${H_LS}${H_IL_SFXR}"
            _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
              "$TAG" "logs/${TAG}.log" \
              python -u run_pretrain.py \
                --data_name "${PRETRAIN_DATA}" \
                --num_feature "${PRETRAIN_NUM_FEATURE}" \
                --num_target  "${PRETRAIN_NUM_TARGET}" \
                --view2 "${V2}" --view3 "${V3}" \
                --encoder_type "${H_ENC}" \
                --batch_size_pretrain "${LOGSIG_BATCH}" \
                --epochs_pretrain "${H_EP}" \
                --logsig_depth "${H_DEPTH}" \
                --logsig_mode "${H_MODE}" \
                --logsig_window_size "${H_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_smooth_param "${H_SP}" \
                --seed "${SEED}" \
                "${H_IL_ARGS[@]}"
          done
        done

        for SEED in "${SEEDS[@]}"; do
          TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${H_EP}_${SEED}${H_ENC_SUFFIX}${H_LS}${H_IL_SFXR}"
          _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
            "$TAG" "logs/${TAG}.log" \
            python -u run_pretrain_nview.py \
              --data_name "${PRETRAIN_DATA}" \
              --num_feature "${PRETRAIN_NUM_FEATURE}" \
              --num_target  "${PRETRAIN_NUM_TARGET}" \
              --view2 logsig \
              --encoder_type "${H_ENC}" \
              --batch_size_pretrain "${LOGSIG_BATCH}" \
              --epochs_pretrain "${H_EP}" \
              --logsig_depth "${H_DEPTH}" \
              --logsig_mode "${H_MODE}" \
              --logsig_window_size "${H_WSIZ}" \
              --logsig_smoothing "${LOGSIG_SMOOTHING}" \
              --logsig_smooth_param "${H_SP}" \
              --seed "${SEED}" \
              "${H_IL_ARGS[@]}"
        done

      done  # H_DEPTH

    fi

    if ! _skip_stage finetune; then
      echo "  [H finetune]"

      for H_DEPTH in "${H_DEPTHS[@]}"; do
        H_LS="$(_lsig_suffix "$H_MODE" "$H_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$H_SP" "$H_DEPTH")"

        for COMBO in "${H_3VIEW_COMBOS[@]}"; do
          V2="${COMBO%%:*}"; V3="${COMBO##*:}"
          for SEED in "${SEEDS[@]}"; do
            PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${H_EP}_${SEED}${H_ENC_SUFFIX}${H_LS}${H_IL_SFXR}"
            PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
            FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
            if [ -f "$FT_OUT" ]; then
              echo "Skip finetune (exists): $(basename "$FT_OUT")"
            elif [ ! -f "$PT_CKPT" ]; then
              echo "Skipping 3-view H finetune — missing: ${PT_CKPT}"
            else
              TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
              _launch "$TAG" "logs/${TAG}.log" \
                python run_finetune.py \
                  --data_name "${FINETUNE_DATA}" \
                  --pretrain_data_name "${PRETRAIN_DATA}" \
                  --num_feature "${FINETUNE_NUM_FEATURE}" \
                  --num_target  "${FINETUNE_NUM_TARGET}" \
                  --view2 "${V2}" --view3 "${V3}" \
                  --encoder_type "${H_ENC}" \
                  --logsig_depth "${H_DEPTH}" \
                  --logsig_mode "${H_MODE}" \
                  --logsig_window_size "${H_WSIZ}" \
                  --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                  --logsig_smooth_param "${H_SP}" \
                  --epochs_pretrain "${H_EP}" \
                  --epochs_finetune "${EPOCHS_FINETUNE}" \
                  --feature "${FEATURE}" \
                  --loss_type "${LOSS_TYPE}" \
                  --lam "${LAM}" \
                  --run_modes "${RUN_MODES}" \
                  --seed "${SEED}" \
                  "${H_IL_ARGS[@]}"
            fi
          done
        done

        for SEED in "${SEEDS[@]}"; do
          PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${H_EP}_${SEED}${H_ENC_SUFFIX}${H_LS}${H_IL_SFXR}"
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
          FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_ALL_${LAM}_0_finetune"
          if [ -f "$FT_OUT" ]; then
            echo "Skip finetune (exists): $(basename "$FT_OUT")"
          elif [ ! -f "$PT_CKPT" ]; then
            echo "Skipping nview H finetune — missing: ${PT_CKPT}"
          else
            TAG="ft2v_${FINETUNE_DATA}_from_${PT_TAG}"
            _launch "$TAG" "logs/${TAG}.log" \
              python run_finetune_nview.py \
                --data_name "${FINETUNE_DATA}" \
                --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature "${FINETUNE_NUM_FEATURE}" \
                --num_target  "${FINETUNE_NUM_TARGET}" \
                --view2 logsig \
                --encoder_type "${H_ENC}" \
                --logsig_depth "${H_DEPTH}" \
                --logsig_mode "${H_MODE}" \
                --logsig_window_size "${H_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_smooth_param "${H_SP}" \
                --epochs_pretrain "${H_EP}" \
                --epochs_finetune "${EPOCHS_FINETUNE}" \
                --feature "${FEATURE}" \
                --lam "${LAM}" \
                --run_modes "${RUN_MODES}" \
                --seed "${SEED}" \
                "${H_IL_ARGS[@]}"
          fi
        done

      done  # H_DEPTH

    fi

  fi  # end section H (depth ablation)

  # =========================================================================
  # SECTION I: Smooth-param ablation — Tukey alpha {0.1,0.25,0.5,0.75,0.9}
  #
  # Fixed config: mlp_logsig, window_smooth, win128, depth=2, bilinear IL, ep2.
  # Runs nview (xt+logsig) and 3-view combos (logsig+xf, dx+logsig).
  # Note: sp=0.5 baseline is already covered by Section B (_tukey128_ilbilinear).
  # Override params: I_SMOOTH_PARAMS="0.1 0.25 0.5 0.75 0.9" bash run_sweep.sh
  # Skip with: SKIP_SECTIONS="smooth_param" bash run_sweep.sh
  # =========================================================================
  if ! _skip_section smooth_param; then

    echo ""
    echo "--- [Section I] Smooth-param ablation ---"

    I_EP=2
    I_ENC="mlp_logsig"
    I_ENC_SUFFIX="_mlp_logsig"
    I_MODE="window_smooth"
    I_WSIZ=128
    I_DEPTH=2
    I_IL_SFXR="_ilbilinear"
    I_IL_ARGS=(--interaction_type bilinear)
    I_3VIEW_COMBOS=("logsig:xf" "dx:logsig")
    IFS=' ' read -ra I_SMOOTH_PARAMS <<< "${I_SMOOTH_PARAMS:-0.1 0.25 0.5 0.75 0.9}"

    if ! _skip_stage pretrain; then
      echo "  [I pretrain]"

      for I_SP in "${I_SMOOTH_PARAMS[@]}"; do
        I_LS="$(_lsig_suffix "$I_MODE" "$I_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$I_SP" "$I_DEPTH")"

        for COMBO in "${I_3VIEW_COMBOS[@]}"; do
          V2="${COMBO%%:*}"; V3="${COMBO##*:}"
          for SEED in "${SEEDS[@]}"; do
            TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${I_EP}_${SEED}${I_ENC_SUFFIX}${I_LS}${I_IL_SFXR}"
            _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
              "$TAG" "logs/${TAG}.log" \
              python -u run_pretrain.py \
                --data_name "${PRETRAIN_DATA}" \
                --num_feature "${PRETRAIN_NUM_FEATURE}" \
                --num_target  "${PRETRAIN_NUM_TARGET}" \
                --view2 "${V2}" --view3 "${V3}" \
                --encoder_type "${I_ENC}" \
                --batch_size_pretrain "${LOGSIG_BATCH}" \
                --epochs_pretrain "${I_EP}" \
                --logsig_depth "${I_DEPTH}" \
                --logsig_mode "${I_MODE}" \
                --logsig_window_size "${I_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_smooth_param "${I_SP}" \
                --seed "${SEED}" \
                "${I_IL_ARGS[@]}"
          done
        done

        for SEED in "${SEEDS[@]}"; do
          TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${I_EP}_${SEED}${I_ENC_SUFFIX}${I_LS}${I_IL_SFXR}"
          _launch_if_new "out_pretrain/${PRETRAIN_DATA}/${TAG}" \
            "$TAG" "logs/${TAG}.log" \
            python -u run_pretrain_nview.py \
              --data_name "${PRETRAIN_DATA}" \
              --num_feature "${PRETRAIN_NUM_FEATURE}" \
              --num_target  "${PRETRAIN_NUM_TARGET}" \
              --view2 logsig \
              --encoder_type "${I_ENC}" \
              --batch_size_pretrain "${LOGSIG_BATCH}" \
              --epochs_pretrain "${I_EP}" \
              --logsig_depth "${I_DEPTH}" \
              --logsig_mode "${I_MODE}" \
              --logsig_window_size "${I_WSIZ}" \
              --logsig_smoothing "${LOGSIG_SMOOTHING}" \
              --logsig_smooth_param "${I_SP}" \
              --seed "${SEED}" \
              "${I_IL_ARGS[@]}"
        done

      done  # I_SP

    fi

    if ! _skip_stage finetune; then
      echo "  [I finetune]"

      for I_SP in "${I_SMOOTH_PARAMS[@]}"; do
        I_LS="$(_lsig_suffix "$I_MODE" "$I_WSIZ" "$LOGSIG_SMOOTHING" "1" "0" "$I_SP" "$I_DEPTH")"

        for COMBO in "${I_3VIEW_COMBOS[@]}"; do
          V2="${COMBO%%:*}"; V3="${COMBO##*:}"
          for SEED in "${SEEDS[@]}"; do
            PT_TAG="${PRETRAIN_DATA}_v2${V2}_v3${V3}_ep${I_EP}_${SEED}${I_ENC_SUFFIX}${I_LS}${I_IL_SFXR}"
            PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
            FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_${LOSS_TYPE}_${LAM}_0_finetune"
            if [ -f "$FT_OUT" ]; then
              echo "Skip finetune (exists): $(basename "$FT_OUT")"
            elif [ ! -f "$PT_CKPT" ]; then
              echo "Skipping 3-view I finetune — missing: ${PT_CKPT}"
            else
              TAG="ft_${FINETUNE_DATA}_from_${PT_TAG}"
              _launch "$TAG" "logs/${TAG}.log" \
                python run_finetune.py \
                  --data_name "${FINETUNE_DATA}" \
                  --pretrain_data_name "${PRETRAIN_DATA}" \
                  --num_feature "${FINETUNE_NUM_FEATURE}" \
                  --num_target  "${FINETUNE_NUM_TARGET}" \
                  --view2 "${V2}" --view3 "${V3}" \
                  --encoder_type "${I_ENC}" \
                  --logsig_depth "${I_DEPTH}" \
                  --logsig_mode "${I_MODE}" \
                  --logsig_window_size "${I_WSIZ}" \
                  --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                  --logsig_smooth_param "${I_SP}" \
                  --epochs_pretrain "${I_EP}" \
                  --epochs_finetune "${EPOCHS_FINETUNE}" \
                  --feature "${FEATURE}" \
                  --loss_type "${LOSS_TYPE}" \
                  --lam "${LAM}" \
                  --run_modes "${RUN_MODES}" \
                  --seed "${SEED}" \
                  "${I_IL_ARGS[@]}"
            fi
          done
        done

        for SEED in "${SEEDS[@]}"; do
          PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${I_EP}_${SEED}${I_ENC_SUFFIX}${I_LS}${I_IL_SFXR}"
          PT_CKPT="model_pretrain/${PRETRAIN_DATA}/${PT_TAG}.pth"
          FT_OUT="out_finetune/${FINETUNE_DATA}/${FINETUNE_DATA}_pt-${PT_TAG}_${FEATURE}_ALL_${LAM}_0_finetune"
          if [ -f "$FT_OUT" ]; then
            echo "Skip finetune (exists): $(basename "$FT_OUT")"
          elif [ ! -f "$PT_CKPT" ]; then
            echo "Skipping nview I finetune — missing: ${PT_CKPT}"
          else
            TAG="ft2v_${FINETUNE_DATA}_from_${PT_TAG}"
            _launch "$TAG" "logs/${TAG}.log" \
              python run_finetune_nview.py \
                --data_name "${FINETUNE_DATA}" \
                --pretrain_data_name "${PRETRAIN_DATA}" \
                --num_feature "${FINETUNE_NUM_FEATURE}" \
                --num_target  "${FINETUNE_NUM_TARGET}" \
                --view2 logsig \
                --encoder_type "${I_ENC}" \
                --logsig_depth "${I_DEPTH}" \
                --logsig_mode "${I_MODE}" \
                --logsig_window_size "${I_WSIZ}" \
                --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                --logsig_smooth_param "${I_SP}" \
                --epochs_pretrain "${I_EP}" \
                --epochs_finetune "${EPOCHS_FINETUNE}" \
                --feature "${FEATURE}" \
                --lam "${LAM}" \
                --run_modes "${RUN_MODES}" \
                --seed "${SEED}" \
                "${I_IL_ARGS[@]}"
          fi
        done

      done  # I_SP

    fi

  fi  # end section I (smooth-param ablation)

done


echo ""
echo "All sweeps complete."
