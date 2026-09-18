"""Course-layout entry point for held-out CycleGAN evaluation and the full metrics CSV."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import task_config_path
from lab1.cyclegan import evaluate_checkpoint

COLUMNS = ["direction", "metric", "value", "status", "details", "checkpoint_sha256"]


def write_metrics_report(result, destination):
    """Unavailable metrics stay blank, with their reason and status preserved."""
    destination = Path(destination)
    if destination.exists() and len(destination.read_text().splitlines()) > 1:
        raise FileExistsError("Metrics report already contains results; choose a fresh path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for direction, entry in result["directions"].items():
            for name, metric in entry["metrics"].items():
                writer.writerow({"direction": direction, "metric": name,
                    "value": "" if metric["value"] is None else metric["value"],
                    "status": metric["status"],
                    "details": json.dumps({k: v for k, v in metric.items() if k not in {"value", "status"}}, sort_keys=True),
                    "checkpoint_sha256": result.get("checkpoint_sha256", "")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=task_config_path("cyclegan"))
    parser.add_argument("--mode", choices=["smoke", "rehearsal", "full"], default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--report", type=Path, help="Defaults to full_metrics_report.csv within the fresh output directory")
    args = parser.parse_args()
    report = args.report or args.output_dir / "full_metrics_report.csv"
    if report.exists() and len(report.read_text().splitlines()) > 1:
        parser.error("The report already contains results; use a fresh --report path.")
    profiles = json.loads(args.config.read_text())
    result = evaluate_checkpoint(args.checkpoint, profiles[args.mode], args.output_dir, args.device, args.split)
    write_metrics_report(result, report)
    print(json.dumps({"metrics_report": str(report), "class_results": result["class_results"]}, indent=2))


if __name__ == "__main__":
    main()
