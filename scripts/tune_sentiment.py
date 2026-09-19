"""Run a separately logged validation-only candidate with portable checkpoints."""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import sentiment as s
from lab1.common import Tee, code_manifest, environment, utc_now, write_json
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    name = candidate["model"]
    if name not in s.MODEL_NAMES:
        parser.error("Unknown model family")
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise FileExistsError("Use an empty output directory or explicit resume")
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "task2_sentiment/srinidhi/config.json").read_text())["full"]
    base.update(data_dir="task2_sentiment/srinidhi/data_processed/full",
                encoded_cache="task2_sentiment/srinidhi/data_processed/full_encoded")
    cfg = dict(base)
    cfg.update(candidate["overrides"], validation_only=True)
    feature_keys = ("seed", "dataset", "dataset_revision", "mode", "train_limit", "validation_limit", "test_limit", "vocabulary_size", "min_frequency", "max_length")
    if any(cfg[key] != base[key] for key in feature_keys):
        raise ValueError("This candidate runner requires the frozen preprocessing and splits")
    manifest = {"candidate": candidate, "config": cfg, "started_utc": utc_now(),
                "status": "running", "source_sha256": code_manifest(), "environment": environment(),
                "test_policy": "No test predictions or test metrics; selection uses validation macro-F1 only"}
    write_json(args.output / "candidate.json", candidate)
    write_json(args.output / "config.json", cfg)
    write_json(args.output / "provenance.json", manifest)
    started = time.perf_counter()
    with (args.output / "RUN_LOG.txt").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                torch.set_num_threads(cfg.get("cpu_threads", 2))
                records = s._load_records(base)
                vocabulary, datasets = s._load_features(base, records)
                fingerprint = s._fingerprint(cfg, vocabulary, records)
                # Remove test data from the training call entirely.
                datasets = {key: value for key, value in datasets.items() if key != "test"}
                result, _, _ = s._train_one(name, cfg, datasets, vocabulary, args.output,
                                          args.device, fingerprint, args.resume)
                manifest.update(status="completed", result=result)
                print(json.dumps(result), flush=True)
            except BaseException as error:
                manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
                raise
            finally:
                manifest.update(ended_utc=utc_now(), elapsed_seconds=time.perf_counter()-started)
                write_json(args.output / "provenance.json", manifest)


if __name__ == "__main__":
    main()
