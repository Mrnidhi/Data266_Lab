"""Inventory finished cloud artifacts, or verify their complete local copy."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--training-only", action="store_true", help="Back up training before local final evaluation")
    args = parser.parse_args()
    if args.mode == "inventory":
        recipes = ("mlp_wider", "maxpool_mlp_long_schedule_control", "cnn_long_context",
                   "dilated_cnn_long_schedule_control", "bilstm_wider", "bilstm_long_schedule_control")
        runs = [(ROOT / "runs/part-b-full", "run_summary.json")]
        if not args.training_only:
            runs.append((ROOT / "runs/part-b-selected", "run_summary.json"))
        runs += [(ROOT / "runs/part-b-tuning" / name, "provenance.json") for name in recipes]
        paths = []
        for run, provenance in runs:
            if json.loads((run / provenance).read_text())["status"] != "completed":
                raise ValueError(f"Cannot inventory an unfinished run: {run.name}")
            paths.extend(p for p in run.rglob("*") if p.is_file())
        if not args.training_only:
            paths.append(ROOT / "runs/part-b-selection.json")
        files = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(paths)}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        scope = "completed_training_runs" if args.training_only else "completed_training_and_final_evaluation"
        args.manifest.write_text(json.dumps({"scope": scope, "created_utc": datetime.now(timezone.utc).isoformat(),
                                             "files": files}, indent=2) + "\n")
        print(json.dumps({"inventoried_files": len(files), "bytes": sum(p["bytes"] for p in files.values())}))
    else:
        manifest = json.loads(args.manifest.read_text())
        for relative, expected in manifest["files"].items():
            path = (ROOT / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(ROOT):
                raise ValueError("Artifact path escapes the repository")
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError(f"Backup mismatch: {relative}")
        receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "verified": True,
                   "scope": manifest.get("scope", "unspecified"),
                   "files": len(manifest["files"]), "manifest_sha256": digest(args.manifest),
                   "bytes": sum(p["bytes"] for p in manifest["files"].values())}
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
