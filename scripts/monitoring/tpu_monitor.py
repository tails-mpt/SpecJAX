#!/usr/bin/env python3
"""
TPU Monitor — standalone chip health and activity monitor.

Reads TPU state via sysfs (/sys/class/accel/accelN/) and the owning process's
/proc/<pid>/status. Safe to run alongside any active job — zero TPU device ops,
no torch_xla initialization, no gRPC connection.

What this covers (from outside the TPU process):
  - Chip state and health (ALIVE / ERROR) — all 4 chips
  - Chip ownership (which PID holds each chip)
  - TPU activity proxy: interrupt count delta per interval
  - Owner process CPU and RSS (host RAM) usage via /proc
  - Wall-clock elapsed time and polling timestamps

What requires in-process instrumentation (cannot be done standalone):
  - HBM (on-chip memory) usage — use xm.get_memory_info() inside your job
  - XLA compile/execute/transfer times — use torch_xla.debug.metrics inside your job
  - CPU fallback ops — use met.executed_fallback_ops() inside your job

See docs/benchmarks/ and docs/references/ for in-process instrumentation patterns.

Usage:
  # Run alongside a job, poll every 10s until Ctrl-C
  python3 scripts/monitoring/tpu_monitor.py

  # Poll every 5s for 10 minutes, quiet mode
  python3 scripts/monitoring/tpu_monitor.py --interval 5 --duration 600 --quiet

  # Custom output directory
  python3 scripts/monitoring/tpu_monitor.py \\
      --output-dir ./logs/tpu_monitor
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("tpu_monitor")

SYSFS_ACCEL = Path("/sys/class/accel")
NUM_CHIPS = 4


# ── sysfs readers ─────────────────────────────────────────────────────────────
def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except Exception:
        return None


def sample_chip(chip_idx: int, prev_interrupts: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """
    Read sysfs state for one TPU chip.

    Returns:
      state            : "available" | "used" | "unavailable" | None
      status           : "ALIVE" | "ERROR" | None
      device_owner_pid : int or None (PID that opened the device)
      interrupt_counts : dict of {hex_irq: count}
      interrupt_delta  : total interrupt count change since last sample
    """
    base = SYSFS_ACCEL / f"accel{chip_idx}"
    state  = _read(base / "state")
    status = _read(base / "status")
    owner_str = _read(base / "device_owner")
    owner_pid = int(owner_str) if owner_str and owner_str.isdigit() else None

    # Interrupt counts — proxy for TPU activity
    irq_raw = _read(base / "interrupt_counts") or ""
    irq: Dict[str, int] = {}
    for line in irq_raw.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            k, v = parts[0].strip(), parts[1].strip()
            try:
                irq[k] = int(v)
            except ValueError:
                pass

    delta = 0
    if prev_interrupts:
        for k, v in irq.items():
            delta += v - prev_interrupts.get(k, 0)

    return {
        "chip_idx": chip_idx,
        "state": state,
        "status": status,
        "device_owner_pid": owner_pid,
        "interrupt_counts": irq,
        "interrupt_delta": delta,
    }


def sample_all_chips(prev_chip_data: Optional[List[Dict]]) -> List[Dict]:
    prev_irqs = None
    if prev_chip_data:
        prev_irqs = [c["interrupt_counts"] for c in prev_chip_data]

    chips = []
    for i in range(NUM_CHIPS):
        prev = prev_irqs[i] if prev_irqs else None
        chips.append(sample_chip(i, prev))
    return chips


# ── process-level stats ───────────────────────────────────────────────────────
def _read_proc_status(pid: int) -> Dict[str, str]:
    """Parse /proc/<pid>/status into a flat dict."""
    path = Path(f"/proc/{pid}/status")
    result: Dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


def sample_owner_process(owner_pid: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    Read /proc/<pid>/status for the TPU owner process.

    Returns CPU and host-RAM stats (not HBM — that requires in-process API).
    """
    if not owner_pid:
        return None
    status = _read_proc_status(owner_pid)
    if not status:
        return None

    def _kb(key: str) -> Optional[int]:
        v = status.get(key, "")
        try:
            return int(v.split()[0]) * 1024  # kB → bytes
        except (ValueError, IndexError):
            return None

    return {
        "pid": owner_pid,
        "name": status.get("Name"),
        "state": status.get("State"),
        "threads": int(status.get("Threads", 0) or 0),
        "vm_rss_bytes": _kb("VmRSS"),    # resident host RAM (not HBM)
        "vm_peak_bytes": _kb("VmPeak"),  # peak virtual memory
        "vm_size_bytes": _kb("VmSize"),  # current virtual memory
    }


# ── session state ─────────────────────────────────────────────────────────────
class MonitorSession:
    def __init__(self):
        self.samples: List[Dict] = []
        self.start_ts = time.time()
        self.peak_irq_delta = 0

    def record(self, ts: float, chips: List[Dict], proc: Optional[Dict]):
        total_delta = sum(c["interrupt_delta"] for c in chips)
        if total_delta > self.peak_irq_delta:
            self.peak_irq_delta = total_delta

        self.samples.append({
            "ts": ts,
            "ts_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "chips": chips,
            "owner_process": proc,
            "total_interrupt_delta": total_delta,
        })

    def summary(self) -> Dict:
        return {
            "start_ts_iso": datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat(),
            "end_ts_iso": datetime.now(tz=timezone.utc).isoformat(),
            "duration_s": time.time() - self.start_ts,
            "num_samples": len(self.samples),
            "peak_interrupt_delta_per_poll": self.peak_irq_delta,
            "note": (
                "HBM usage and XLA metrics require in-process instrumentation. "
                "See benchmark/docs/tpu/tpu_metrics_inference.md."
            ),
            "samples": self.samples,
        }


# ── printer ───────────────────────────────────────────────────────────────────
def print_sample(ts: float, chips: List[Dict], proc: Optional[Dict], poll_num: int):
    ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    print(f"[{ts_str}] poll #{poll_num}", flush=True)

    for c in chips:
        status = c["status"] or "?"
        state  = c["state"] or "?"
        owner  = c["device_owner_pid"] or "—"
        delta  = c["interrupt_delta"]
        marker = "  " if status == "ALIVE" else "! "
        print(f"  {marker}accel{c['chip_idx']}: {status:<6} state={state:<12} "
              f"owner={str(owner):<8} irq_delta={delta}", flush=True)

    if proc:
        rss_gib = (proc.get("vm_rss_bytes") or 0) / (1 << 30)
        print(f"  proc pid={proc['pid']} ({proc.get('name','?')}) "
              f"RSS={rss_gib:.2f} GiB  threads={proc.get('threads','?')}  "
              f"state={proc.get('state','?')}", flush=True)
    elif chips and chips[0]["device_owner_pid"]:
        print(f"  proc: pid {chips[0]['device_owner_pid']} (could not read /proc)", flush=True)

    total_delta = sum(c["interrupt_delta"] for c in chips)
    print(f"  total IRQ delta this poll: {total_delta}", flush=True)


def print_summary(summary: Dict):
    print("\n" + "=" * 60, flush=True)
    print("TPU Monitor Session Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  Duration     : {summary['duration_s']:.1f} s  ({summary['num_samples']} samples)", flush=True)
    print(f"  Peak IRQ δ   : {summary['peak_interrupt_delta_per_poll']} interrupts/poll", flush=True)
    print("  Note         : HBM usage requires in-process instrumentation.", flush=True)
    print("                 See tpu_metrics_inference.md / tpu_metrics_training.md", flush=True)
    print("=" * 60, flush=True)


# ── main loop ─────────────────────────────────────────────────────────────────
def run(args):
    session = MonitorSession()
    stop = False

    def _handle_sig(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"tpu_monitor_{session_tag}.json"

    print(f"TPU Monitor — interval={args.interval}s  "
          f"duration={'∞' if args.duration <= 0 else f'{args.duration}s'}  "
          f"output={output_path}", flush=True)
    print("Monitoring chip health via sysfs. "
          "For HBM usage, add xm.get_memory_info() to your job.", flush=True)

    deadline = time.time() + args.duration if args.duration > 0 else float("inf")
    poll_num = 0
    next_flush = args.flush_interval
    prev_chips: Optional[List[Dict]] = None

    while not stop and time.time() < deadline:
        poll_num += 1
        ts = time.time()

        chips = sample_all_chips(prev_chips)
        # All chips share one owner; read process stats from chip 0's owner
        owner_pid = chips[0]["device_owner_pid"] if chips else None
        proc = sample_owner_process(owner_pid)

        session.record(ts, chips, proc)
        prev_chips = chips

        if not args.quiet:
            print_sample(ts, chips, proc, poll_num)

        if poll_num >= next_flush:
            _write_json(output_path, session.summary())
            next_flush += args.flush_interval

        remaining = (ts + args.interval) - time.time()
        if remaining > 0 and not stop:
            time.sleep(remaining)

    summary = session.summary()
    _write_json(output_path, summary)
    print_summary(summary)
    print(f"\nFull data written to: {output_path}", flush=True)


def _write_json(path: Path, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "TPU chip health monitor (sysfs-based). "
            "Safe to run alongside any active job. "
            "For HBM/XLA metrics, see in-process patterns in the docs."
        )
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Polling interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--duration", type=float, default=0,
        help="Total monitoring duration in seconds. 0 = run until Ctrl-C (default: 0)"
    )
    parser.add_argument(
        "--output-dir", default="./logs/tpu_monitor",
        help="Directory for JSON output"
    )
    parser.add_argument(
        "--flush-interval", type=int, default=6,
        help="Write JSON to disk every N polls (default: 6)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-poll stdout output (summary still printed)"
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
