#!/usr/bin/env bash
# Monitors logs/ for OOM errors and appends findings to oom_log.txt.

LOG_DIR="${1:-logs}"
OOM_LOG="oom_log.txt"
INTERVAL=30
SEEN_FILE="/tmp/.oom_watch_seen_$$"

trap "rm -f $SEEN_FILE; exit" INT TERM

touch "$SEEN_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] OOM watcher started (watching $LOG_DIR, interval=${INTERVAL}s)" | tee -a "$OOM_LOG"

while true; do
    for logfile in "$LOG_DIR"/*.log; do
        [ -f "$logfile" ] || continue
        job=$(basename "$logfile")

        # Skip files we've already reported OOM for
        grep -qxF "$job" "$SEEN_FILE" && continue

        if grep -qE "OutOfMemoryError|out of memory|CUDA out of memory" "$logfile" 2>/dev/null; then
            snippet=$(grep -E "OutOfMemoryError|out of memory|CUDA out of memory" "$logfile" | tail -1)
            msg="[$(date '+%Y-%m-%d %H:%M:%S')] OOM in $job: $snippet"
            echo "$msg" | tee -a "$OOM_LOG"
            echo "$job" >> "$SEEN_FILE"
        fi
    done
    sleep "$INTERVAL"
done
