"""Show Part 2 validation progress and current GPU load without reading test scores."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
runs = [ROOT / "runs/part-b-full", *sorted((ROOT / "runs/part-b-tuning").glob("*"))]
print(f"{'Run / model':65} {'Epochs':>6} {'Best val F1':>12} {'Status':>12}")
for run in runs:
    if not run.is_dir():
        continue
    provenance = next((p for p in (run / "provenance.json", run / "run_summary.json") if p.is_file()), None)
    try:
        status = json.loads(provenance.read_text()).get("status", "unknown") if provenance else "in progress"
        for path in sorted(run.glob("*/history.json")):
            history = json.loads(path.read_text())
            best = max((r["validation_macro_f1"] for r in history), default=0)
            label = f"{run.name} / {path.parent.name}"
            print(f"{label:65} {len(history):6} {best:12.5f} {status:>12}")
    except (OSError, json.JSONDecodeError):
        print(f"{run.name}: metadata being updated; refresh shortly")
try:
    subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv"], check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    print("Live GPU telemetry is available on the training host.")
