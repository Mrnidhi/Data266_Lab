"""Bound the training processes. This is NOT a cloud billing cutoff or pod controller."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["smoke", "rehearsal"], default="rehearsal")
    parser.add_argument("--max-minutes", type=float, default=90)
    parser.add_argument("--hourly-rate", type=float, default=0.74)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=["gpt", "sentiment", "cyclegan"],
                        default=["gpt", "sentiment", "cyclegan"])
    args = parser.parse_args()
    if args.max_minutes <= 0 or args.hourly_rate < 0:
        parser.error("Use a positive time limit and non-negative hourly rate.")
    output = args.output or ROOT / "runs" / ("rehearsal-" + utc_now().replace(":", "-"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    print("This script limits training time only. It does not stop the pod or stop billing.", flush=True)
    for task in args.tasks:
        remaining = args.max_minutes * 60 - (time.monotonic() - started)
        if remaining <= 0:
            records.append({"task": task, "status": "not_run_time_limit"})
            continue
        command = [sys.executable, "-m", "lab1.run", "--task", task, "--mode", args.mode,
                   "--device", args.device, "--output", str(output / task)]
        if args.mode == "rehearsal":
            if task == "gpt" and (ROOT / "task1_llm/srinidhi/data_processed/rehearsal").is_dir():
                command += ["--set", 'data_cache="task1_llm/srinidhi/data_processed/rehearsal"', "--set", "offline=true"]
            if task == "sentiment" and (ROOT / "task2_sentiment/srinidhi/data_processed/rehearsal/manifest.json").is_file():
                command += ["--set", 'data_dir="task2_sentiment/srinidhi/data_processed/rehearsal"']
        task_started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, timeout=remaining)
            status = "passed" if completed.returncode == 0 else "failed"
            records.append({"task": task, "status": status, "returncode": completed.returncode,
                            "elapsed_seconds": time.monotonic() - task_started})
        except subprocess.TimeoutExpired:
            records.append({"task": task, "status": "terminated_at_time_limit",
                            "elapsed_seconds": time.monotonic() - task_started})
            break
    elapsed = time.monotonic() - started
    result = {"mode": args.mode, "tasks": records, "elapsed_seconds": elapsed,
              "gpu_time_cost_estimate_usd": elapsed / 3600 * args.hourly_rate,
              "cost_excludes": ["pod startup", "idle time", "storage", "taxes"],
              "pod_stopped": False, "billing_cutoff": False,
              "all_requested_tasks_passed": len(records) == len(args.tasks) and all(
                  r["status"] == "passed" for r in records)}
    write_json(output / "rehearsal_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print("DOWNLOAD RESULTS AND STOP THE POD. This script has NOT stopped billing.", flush=True)
    return 0 if result["all_requested_tasks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
