#!/usr/bin/env bash
# watchdog.sh — autonomous training monitor and recovery agent
# Monitors Qwen3-8B → Qwen3-14B pipeline
# Heartbeats: 60s pulse, 5m metrics, 30m summary
#
# Staleness fix: uses log FILE mtime (not step-line recency) to detect
# genuine hangs. Allows 15min before flagging — covers XLA compilation
# which can take 10-12min with no [step N] output.

set -uo pipefail

WATCHDOG_LOG="/path/to/workspace/logs/watchdog_v2_$(date +%Y%m%d_%H%M%S).log"
REPO_ROOT="/path/to/specjax"
LOG_DIR="/path/to/workspace/logs"

PIPELINE_PID=""
RESTART_COUNT=0
MAX_RESTARTS=3
START_TIME=$(date +%s)
BEST_LOSS=99999
CHECKPOINTS_SAVED=0
ISSUES=""

# Stale threshold: 15 min (covers XLA compile during model transition)
STALE_THRESHOLD_SECS=900

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

# ── Helpers ──────────────────────────────────────────────────────────────────

get_active_log() {
    # Most recently MODIFIED w0 log across all pending models
    ls -t "$LOG_DIR"/qwen3_8b_eagle3_w0_*.log \
           "$LOG_DIR"/qwen3_14b_eagle3_w0_*.log 2>/dev/null | head -1
}

get_latest_step_line() {
    local log="$1"
    [[ -z "$log" || ! -f "$log" ]] && echo "" && return
    grep "^\[step" "$log" | tail -1
}

log_stale() {
    # True if the most recently modified w0 log hasn't changed in STALE_THRESHOLD_SECS
    local log
    log=$(get_active_log)
    [[ -z "$log" ]] && return 0   # no log at all = stale
    local age
    age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
    [[ $age -gt $STALE_THRESHOLD_SECS ]]
}

training_alive() {
    pgrep -f "specjax.train" > /dev/null 2>&1
}

check_loss_health() {
    local log="$1"
    [[ -z "$log" || ! -f "$log" ]] && echo "UNKNOWN" && return
    local last_10
    last_10=$(grep "^\[step" "$log" 2>/dev/null | tail -10)
    [[ -z "$last_10" ]] && echo "COMPILING" && return

    if echo "$last_10" | grep -qi "nan\|inf"; then
        echo "NAN"; return
    fi

    local losses first last delta
    losses=$(echo "$last_10" | grep -oP 'loss=\K[0-9.]+')
    first=$(echo "$losses" | head -1)
    last=$(echo "$losses" | tail -1)
    delta=$(awk "BEGIN{print ($last - $first)}" 2>/dev/null || echo "0")
    if awk "BEGIN{exit !($delta > 3.0)}" 2>/dev/null; then
        echo "DIVERGING"; return
    fi
    echo "OK"
}

get_last_checkpoint() {
    local ckpt_dir="$1"
    ls -dt "$ckpt_dir"/epoch_* 2>/dev/null | head -1 || echo ""
}

uptime_human() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "%dh%02dm" $((secs/3600)) $(( (secs%3600)/60 ))
}

# ── Pipeline launcher ─────────────────────────────────────────────────────────

launch_pipeline() {
    local attempt="${1:-1}"
    local script="/tmp/pipeline_attempt_${attempt}.sh"

    local qwen3_8b_done=false
    [[ -d "/path/to/workspace/checkpoints/qwen3-8b-eagle3/final" ]] && qwen3_8b_done=true

    cat > "$script" << 'HEREDOC'
#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="/path/to/specjax"
LOG_DIR="/path/to/workspace/logs"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_model() {
    local name="$1" launch_script="$2" checkpoint_dir="$3"
    log "========================================================"
    log "Starting: $name"
    log "========================================================"
    local LATEST
    LATEST=$(ls -dt "$checkpoint_dir"/epoch_* 2>/dev/null | head -1 || true)
    if [[ -d "$checkpoint_dir/final" ]]; then
        log "[$name] Already has final checkpoint — skipping"
        return 0
    elif [[ -n "$LATEST" ]]; then
        log "[$name] Resuming from: $LATEST"
        bash "$launch_script" --resume="$LATEST"
    else
        log "[$name] No checkpoint — dry-run then full training"
        bash "$launch_script" --dry-run
        rm -rf "$checkpoint_dir/final" "$checkpoint_dir/epoch_"* 2>/dev/null || true
        bash "$launch_script"
    fi
    local ec=$?
    if [[ $ec -ne 0 ]]; then
        local LATEST2
        LATEST2=$(ls -dt "$checkpoint_dir"/epoch_* 2>/dev/null | head -1 || true)
        if [[ -n "$LATEST2" ]]; then
            log "[$name] Retry from $LATEST2"
            bash "$launch_script" --resume="$LATEST2"
        else
            log "[$name] BLOCKED — no checkpoint to resume from"; return 1
        fi
    fi
    log "[$name] Complete."
}
HEREDOC

    # Append model calls based on what's done
    if ! $qwen3_8b_done; then
        cat >> "$script" << HEREDOC
run_model "Qwen3-8B" "$REPO_ROOT/scripts/run/launch_all_workers_qwen3_8b.sh" "/path/to/workspace/checkpoints/qwen3-8b-eagle3"
HEREDOC
    fi
    cat >> "$script" << HEREDOC
run_model "Qwen3-14B" "$REPO_ROOT/scripts/run/launch_all_workers_qwen3_14b.sh" "/path/to/workspace/checkpoints/qwen3-14b-eagle3"
log "ALL MODELS COMPLETE"
HEREDOC

    chmod +x "$script"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local plog="$LOG_DIR/pipeline_a${attempt}_${ts}.log"
    nohup bash "$script" > "$plog" 2>&1 &
    PIPELINE_PID=$!
    log "[PIPELINE] Launched attempt=$attempt PID=$PIPELINE_PID log=$plog"
}

# ── Heartbeat loops ───────────────────────────────────────────────────────────

heartbeat_60s() {
    while true; do
        sleep 60
        local active_log step_line alive health stale_info
        active_log=$(get_active_log)
        step_line=$(get_latest_step_line "$active_log")
        alive=$(training_alive && echo "yes" || echo "no")
        local age=0
        if [[ -n "$active_log" ]]; then
            age=$(( $(date +%s) - $(stat -c %Y "$active_log" 2>/dev/null || echo 0) ))
        fi
        stale_info="${age}s ago"
        health="OK"
        [[ "$alive" == "no" ]] && health="FAILED"
        [[ $age -gt $STALE_THRESHOLD_SECS ]] && health="DEGRADED(stale)"
        log "[HEARTBEAT:60s] job alive: $alive | log updated: $stale_info | health: $health | last step: ${step_line:-compiling/loading...}"
    done
}

heartbeat_5m() {
    while true; do
        sleep 300
        local active_log step_line loss acc0 acc1 acc2 lr wandb_ok ckpt_info
        active_log=$(get_active_log)
        step_line=$(get_latest_step_line "$active_log")
        if [[ -n "$step_line" ]]; then
            loss=$(echo "$step_line" | grep -oP 'loss=\K[0-9.]+' || echo "?")
            acc0=$(echo "$step_line" | grep -oP 'acc_0=\K[0-9.]+%' || echo "?")
            acc1=$(echo "$step_line" | grep -oP 'acc_1=\K[0-9.]+%' || echo "?")
            acc2=$(echo "$step_line" | grep -oP 'acc_2=\K[0-9.]+%' || echo "?")
            lr=$(echo  "$step_line" | grep -oP 'lr=\K[0-9e.+-]+'   || echo "?")
            if [[ "$loss" != "?" ]] && awk "BEGIN{exit !($loss < $BEST_LOSS)}" 2>/dev/null; then
                BEST_LOSS=$loss
            fi
        else
            loss="?"; acc0="?"; acc1="?"; acc2="?"; lr="?"
        fi
        wandb_ok=$(pgrep -f "wandb" > /dev/null 2>&1 && echo "yes" || echo "no")
        # Checkpoint summary
        local q8_ckpts q14_ckpts
        q8_ckpts=$(ls -d /path/to/workspace/checkpoints/qwen3-8b-eagle3/epoch_* \
                          /path/to/workspace/checkpoints/qwen3-8b-eagle3/final 2>/dev/null | wc -l)
        q14_ckpts=$(ls -d /path/to/workspace/checkpoints/qwen3-14b-eagle3/epoch_* \
                           /path/to/workspace/checkpoints/qwen3-14b-eagle3/final 2>/dev/null | wc -l)
        log "[HEARTBEAT:5m]  loss: $loss | acc[0..2]: $acc0 $acc1 $acc2 | lr: $lr | W&B: $wandb_ok | ckpts(qwen3-8b:$q8_ckpts qwen3-14b:$q14_ckpts)"
    done
}

heartbeat_30m() {
    while true; do
        sleep 1800
        local active_log health status
        active_log=$(get_active_log)
        health=$(check_loss_health "$active_log")
        status="ON TRACK"
        [[ "$health" == "DIVERGING" || "$health" == "NAN" ]] && status="AT RISK ($health)"
        [[ -z "$PIPELINE_PID" ]] || ! kill -0 "$PIPELINE_PID" 2>/dev/null && \
            [[ ! -d "/path/to/workspace/checkpoints/qwen3-14b-eagle3/final" ]] && \
            status="BLOCKED (pipeline dead)"

        local q8_ckpts q14_ckpts
        q8_ckpts=$(ls -d /path/to/workspace/checkpoints/qwen3-8b-eagle3/epoch_* \
                          /path/to/workspace/checkpoints/qwen3-8b-eagle3/final 2>/dev/null | wc -l)
        q14_ckpts=$(ls -d /path/to/workspace/checkpoints/qwen3-14b-eagle3/epoch_* \
                           /path/to/workspace/checkpoints/qwen3-14b-eagle3/final 2>/dev/null | wc -l)
        CHECKPOINTS_SAVED=$((q8_ckpts + q14_ckpts))

        log "[HEARTBEAT:30m] uptime: $(uptime_human) | restarts: $RESTART_COUNT | best loss: $BEST_LOSS"
        log "[HEARTBEAT:30m] checkpoints saved: $CHECKPOINTS_SAVED (qwen3-8b:$q8_ckpts qwen3-14b:$q14_ckpts)"
        log "[HEARTBEAT:30m] issues: ${ISSUES:-none} | status: $status"
    done
}

# ── Main watchdog loop ────────────────────────────────────────────────────────

main() {
    log "========================================================"
    log "[WATCHDOG v2] Starting. Staleness threshold: ${STALE_THRESHOLD_SECS}s"
    log "[WATCHDOG v2] Monitoring: Qwen3-8B → Qwen3-14B"
    log "[WATCHDOG v2] Log: $WATCHDOG_LOG"
    log "========================================================"

    # Launch pipeline
    launch_pipeline 1

    # Launch heartbeat subshells
    heartbeat_60s & HB60_PID=$!
    heartbeat_5m  & HB5_PID=$!
    heartbeat_30m & HB30_PID=$!
    log "[BACKGROUND] created: heartbeat_60s PID=$HB60_PID | purpose: job pulse  | cleanup: pending"
    log "[BACKGROUND] created: heartbeat_5m  PID=$HB5_PID  | purpose: metrics    | cleanup: pending"
    log "[BACKGROUND] created: heartbeat_30m PID=$HB30_PID | purpose: summary    | cleanup: pending"

    local stale_count=0
    local diverge_count=0

    while true; do
        sleep 60

        # ── All done? ──────────────────────────────────────────────────────
        if [[ -d "/path/to/workspace/checkpoints/qwen3-14b-eagle3/final" ]]; then
            log "[WATCHDOG v2] Qwen3-14B final checkpoint found — all models complete."
            kill $HB60_PID $HB5_PID $HB30_PID 2>/dev/null || true
            log "[BACKGROUND] removed: heartbeat_60s | cleanup: done"
            log "[BACKGROUND] removed: heartbeat_5m  | cleanup: done"
            log "[BACKGROUND] removed: heartbeat_30m | cleanup: done"
            log "[WATCHDOG v2] Exiting cleanly."
            exit 0
        fi

        # ── Pipeline dead without completion? ─────────────────────────────
        if [[ -n "$PIPELINE_PID" ]] && ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
            if [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
                log "[WATCHDOG v2] [BLOCKED] Max restarts ($MAX_RESTARTS) reached."
                log "[WATCHDOG v2] Issues: $ISSUES"
                kill $HB60_PID $HB5_PID $HB30_PID 2>/dev/null || true
                exit 1
            fi
            RESTART_COUNT=$((RESTART_COUNT + 1))
            ISSUES="${ISSUES}; pipeline died at $(date '+%H:%M') restart #$RESTART_COUNT"
            log "[WATCHDOG v2] Pipeline dead — restart #$RESTART_COUNT"
            pkill -f "specjax.train" 2>/dev/null || true
            pkill -f "launch_all_workers" 2>/dev/null || true
            sleep 5
            launch_pipeline $((RESTART_COUNT + 1))
            stale_count=0
            continue
        fi

        # ── Log staleness (genuine hang, not just compiling) ──────────────
        if log_stale; then
            stale_count=$((stale_count + 1))
            log "[WATCHDOG v2] WARNING: log stale >${STALE_THRESHOLD_SECS}s (count=$stale_count)"
            if [[ $stale_count -ge 2 ]]; then
                log "[WATCHDOG v2] Genuine hang confirmed — restarting"
                ISSUES="${ISSUES}; hang at $(date '+%H:%M')"
                kill "$PIPELINE_PID" 2>/dev/null || true
                pkill -f "specjax.train" 2>/dev/null || true
                sleep 5
                RESTART_COUNT=$((RESTART_COUNT + 1))
                launch_pipeline $((RESTART_COUNT + 1))
                stale_count=0
            fi
        else
            stale_count=0
        fi

        # ── Loss health ────────────────────────────────────────────────────
        local active_log health
        active_log=$(get_active_log)
        health=$(check_loss_health "$active_log")
        if [[ "$health" == "DIVERGING" || "$health" == "NAN" ]]; then
            diverge_count=$((diverge_count + 1))
            log "[WATCHDOG v2] WARNING: loss $health (count=$diverge_count)"
            if [[ $diverge_count -ge 2 ]]; then
                log "[WATCHDOG v2] Loss $health confirmed — restarting from checkpoint"
                ISSUES="${ISSUES}; loss $health at $(date '+%H:%M')"
                kill "$PIPELINE_PID" 2>/dev/null || true
                pkill -f "specjax.train" 2>/dev/null || true
                sleep 5
                RESTART_COUNT=$((RESTART_COUNT + 1))
                launch_pipeline $((RESTART_COUNT + 1))
                diverge_count=0
            fi
        else
            diverge_count=0
        fi

    done
}

main
