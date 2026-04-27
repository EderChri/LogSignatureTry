#!/bin/bash
# =============================================================================
# End-to-end pipeline: pretrain → finetune → probe → aggregate for all pairs.
#
# Uses the unified run_sweep.sh with datasets.cfg for all dataset pairs.
# Each pair runs: pretrain → finetune → probe (all 3 stages sequentially).
#
# Usage:
#   bash run_full_pipeline.sh
#   bash run_full_pipeline.sh sleepeeg_epilepsy   # one pair only
# =============================================================================

set -u

mkdir -p logs/pipeline

TS=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="logs/pipeline/pipeline_${TS}.log"
RESULTS_HISTORY="logs/pipeline/results_history.tsv"

log_msg() {
  local msg="$1"
  echo "[$(date '+%F %T')] ${msg}" | tee -a "$PIPELINE_LOG"
}

append_results() {
  local stage="$1"
  local summary_file="$2"
  local metric_name="$3"

  if [ ! -f "$summary_file" ]; then
    log_msg "No summary file found for ${stage}: ${summary_file}"
    return
  fi

  if [ ! -f "$RESULTS_HISTORY" ]; then
    echo -e "timestamp\tstage\tsummary_file\trun_name\tmetric_name\tmetric_value\tepochs_trained" > "$RESULTS_HISTORY"
  fi

  tail -n +2 "$summary_file" | while IFS=$'\t' read -r run_name metric_value epochs_trained; do
    [ -z "$run_name" ] && continue
    echo -e "${TS}\t${stage}\t${summary_file}\t${run_name}\t${metric_name}\t${metric_value}\t${epochs_trained}" >> "$RESULTS_HISTORY"
  done

  log_msg "Appended results from ${summary_file}"
}

REQUESTED="${1:-all}"

log_msg "Pipeline started (pair: ${REQUESTED})"

log_msg "Running sweep: bash run_sweep.sh ${REQUESTED}"
bash run_sweep.sh "${REQUESTED}" 2>&1 | tee -a "$PIPELINE_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log_msg "Sweep failed; aborting."
  exit 1
fi

# Aggregate multi-seed results
log_msg "Aggregating multi-seed results: python aggregate_results.py"
python aggregate_results.py 2>&1 | tee -a "$PIPELINE_LOG"

# Collect summaries from all datasets discovered in output folders.
for pretrain_summary in out_pretrain/*/final_pretrain_summary.tsv; do
  [ -f "$pretrain_summary" ] || continue
  append_results "pretrain" "$pretrain_summary" "best_valid_loss"
done

for finetune_summary in out_finetune/*/final_test_metric_summary.tsv; do
  [ -f "$finetune_summary" ] || continue
  append_results "finetune" "$finetune_summary" "final_test_score"
done

log_msg "Pipeline finished successfully"
log_msg "Pipeline log: ${PIPELINE_LOG}"
log_msg "Persistent results: ${RESULTS_HISTORY}"
