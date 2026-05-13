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
# Each pair runs three sections in order:
#   A. 3-view sweep  — xt+dx+xf, xt+logsig+xf, xt+dx+logsig
#                      pretrain → finetune → probe
#   B. 2-view sweep  — xt+logsig only (uses run_{pretrain,finetune}_nview.py)
#                      pretrain → finetune
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
SEEDS=(0 1 2 3 4 5 6 7 8 9)
EPOCHS_PRETRAIN=(2)
EPOCHS_FINETUNE=10

# ---------------------------------------------------------------------------
# Interaction layer variants (Section A only — 3-view scripts)
#   Default: attention only (existing behaviour, no tag change).
#   Full sweep: set env var  INTERACTION_TYPES="attention view_embed bilinear"
# ---------------------------------------------------------------------------
IFS=' ' read -ra INTERACTION_TYPES <<< "${INTERACTION_TYPES:-attention}"

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

PARALLEL="${PARALLEL:-false}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"   # max concurrent jobs in PARALLEL mode; 0 = unlimited
DEFAULT_BATCH=256
LOGSIG_BATCH=128
DISABLE_TQDM=0

SKIP_STAGES="${SKIP_STAGES:-}"     # space-separated: "pretrain" "finetune" "probe"
SKIP_SECTIONS="${SKIP_SECTIONS:-3view pool_ablation stride}" # default: 2view only
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

_launch() {
  local tag="$1"; local log="$2"; shift 2; local cmd=("$@")
  local gid="${GPU_LIST[$(( job_index % num_gpus ))]}"
  if [ "$PARALLEL" = true ]; then
    # Honour MAX_PARALLEL: wait until a slot is free before launching
    if [ "${MAX_PARALLEL}" -gt 0 ] 2>/dev/null; then
      while [ "$(jobs -r | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        sleep 2
      done
    fi
    echo "Launching [GPU ${gid}]: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" > "$log" 2>&1 &
  else
    echo "Running [GPU ${gid}]: ${tag}"
    TQDM_DISABLE=${DISABLE_TQDM} CUDA_VISIBLE_DEVICES=$gid "${cmd[@]}" 2>&1 | tee "$log"
  fi
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
                        --logsig_depth "${LOGSIG_DEPTH}" \
                        --logsig_mode "${MODE}" \
                        --logsig_window_size "${WSIZ}" \
                        --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                        --seed "${SEED}" \
                        "${IL_ARGS[@]}"
                  done
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
                        --seed "${SEED}" \
                        "${IL_ARGS[@]}"
                  done
                done
              done
            done
          done
        done
      done
      [ "$PARALLEL" = true ] && { wait; echo "3-view finetune jobs done."; }
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
            # stride=1 is the baseline; stride>1 only makes sense for window modes
            if [ "$MODE" = "stream" ]; then
              STRIDES_2V=(1)
            else
              STRIDES_2V=(1 7)
            fi
            for STRIDE_2V in "${STRIDES_2V[@]}"; do
            LS="$(_lsig_suffix "$MODE" "$WSIZ" "$LOGSIG_SMOOTHING" "$STRIDE_2V")"
            for SEED in "${SEEDS[@]}"; do

              # ------ 2-view pretrain ------
              if ! _skip_stage pretrain; then
                TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
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
                    --logsig_depth "${LOGSIG_DEPTH}" \
                    --logsig_mode "${MODE}" \
                    --logsig_window_size "${WSIZ}" \
                    --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                    --logsig_stride "${STRIDE_2V}" \
                    --seed "${SEED}"
              fi

              # ------ 2-view finetune ------
              if ! _skip_stage finetune; then
                PT_TAG="${PRETRAIN_DATA}_v2logsig_nview_ep${EP}_${SEED}${ENC_SUFFIX}${LS}"
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
                      --logsig_depth "${LOGSIG_DEPTH}" \
                      --logsig_mode "${MODE}" \
                      --logsig_window_size "${WSIZ}" \
                      --logsig_smoothing "${LOGSIG_SMOOTHING}" \
                      --logsig_stride "${STRIDE_2V}" \
                      --epochs_pretrain "${EP}" \
                      --epochs_finetune "${EPOCHS_FINETUNE}" \
                      --feature "${FEATURE}" \
                      --lam "${LAM}" \
                      --run_modes "${RUN_MODES}" \
                      --seed "${SEED}"
                fi
              fi

            done
            done  # STRIDE_2V
          done
        done
      done
    done
    [ "$PARALLEL" = true ] && { wait; echo "2-view jobs done."; }

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
    [ "$PARALLEL" = true ] && { wait; echo "pool ablation jobs done."; }

  fi  # end section C (pool ablation)

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
done

wait   # collect any remaining background jobs (PARALLEL=true)

echo ""
echo "All sweeps complete."
