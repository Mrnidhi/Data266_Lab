"""Compare verified GPT checkpoints on identical frozen validation windows.

Native-context losses are retained separately: a 512-character model and a
256-character model otherwise see different prefixes at window boundaries.
This script only evaluates saved checkpoints and never modifies training logs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lab1 import gpt
from lab1.common import utc_now
from publish_gpt_results import digest, verify_training_evidence


def validate_comparison_contract(bundles: list[dict], context_length: int) -> None:
    """Reject different target populations, encodings, or unavailable contexts."""
    if not bundles or context_length < 1:
        raise ValueError("At least one run and a positive comparison context are required")
    first = bundles[0]
    for bundle in bundles:
        if bundle["summary"]["manifest_sha256"] != first["summary"]["manifest_sha256"]:
            raise ValueError("Comparison requires exactly the same frozen data manifest")
        if bundle["vocabulary"] != first["vocabulary"]:
            raise ValueError("Comparison requires identical training-derived character encodings")
        if context_length > bundle["config"]["context_length"]:
            raise ValueError("Comparison context exceeds a checkpoint's learned position capacity")


def compare(run_dirs: list[Path], output: Path, *, context_length: int = 256,
            batch_size: int = 128, device: str = "cuda", include_train: bool = False,
            minimum_improvement: float = 0.005, root: Path = ROOT) -> dict:
    root, output = root.resolve(), output.resolve()
    runs = [path.resolve() for path in run_dirs]
    if output.exists():
        raise FileExistsError("Comparison receipt exists; choose a fresh output path")
    if len(set(runs)) != len(runs):
        raise ValueError("Duplicate run directory")
    if any(output.is_relative_to(run) for run in runs):
        raise ValueError("Comparison output must be outside the immutable training runs")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("Minimum improvement must be finite and nonnegative")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    bundles = []
    for run in runs:
        # Includes exact source hashes, complete epoch/target coverage, checkpoint
        # contents, selected metrics, manifest and every cached story text hash.
        bundle = verify_training_evidence(root, run)
        bundle["vocabulary"] = json.loads((run / "vocabulary.json").read_text(encoding="utf-8"))
        bundle["checkpoint_sha256"] = digest(run / "checkpoints/best.pt")
        bundles.append(bundle)
    validate_comparison_contract(bundles, context_length)

    paths = bundles[0]["cache_paths"]
    split_names = ["train", "validation"] if include_train else ["validation"]
    data = {}
    for split in split_names:
        with paths[split].open(encoding="utf-8") as handle:
            stories = [json.loads(line)["text"] for line in handle]
        data[split] = gpt.StoryWindows(stories, bundles[0]["vocabulary"], context_length)
    torch.set_num_threads(2)
    gpt.seed_everything(2342, deterministic=True)
    amp = target.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    candidates = []
    for run, bundle in zip(runs, bundles):
        checkpoint_path = run / "checkpoints/best.pt"
        checkpoint = gpt.load_checkpoint(checkpoint_path)
        if digest(checkpoint_path) != bundle["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after verification")
        model = gpt.CharacterGPT(len(bundle["vocabulary"]["tokens"]), bundle["config"]).to(target)
        model.load_state_dict(checkpoint["model"], strict=True)
        started = time.perf_counter()
        metrics = {}
        for split in split_names:
            loader = gpt._loader(data[split], batch_size, seed=2342, shuffle=False)
            metrics[split] = gpt.evaluate(model, loader, target, amp=amp, dtype=dtype)
            if metrics[split]["scored_targets_including_eos"] != data[split].target_count:
                raise ValueError("Evaluation omitted or duplicated character/EOS targets")
        row = {"run": run.relative_to(root).as_posix(),
               "checkpoint_sha256": bundle["checkpoint_sha256"],
               "selected_epoch": bundle["summary"]["best_checkpoint_epoch"],
               "parameter_count": bundle["summary"]["parameter_count"],
               "native_context_length": bundle["config"]["context_length"],
               "native_validation_from_training": bundle["summary"]["best_validation"],
               "matched_context_metrics": metrics,
               "evaluation_seconds": time.perf_counter() - started}
        if include_train:
            row["matched_eval_generalization_gap"] = (
                metrics["validation"]["cross_entropy"] - metrics["train"]["cross_entropy"])
        candidates.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        del model, checkpoint
        gc.collect()
        if amp:
            torch.cuda.empty_cache()
    ranked = sorted(candidates, key=lambda row: (
        row["matched_context_metrics"]["validation"]["cross_entropy"], row["run"]))
    reference = candidates[0]
    improvement = (reference["matched_context_metrics"]["validation"]["cross_entropy"] -
                   ranked[0]["matched_context_metrics"]["validation"]["cross_entropy"])
    qualifies = ranked[0]["run"] != reference["run"] and improvement >= minimum_improvement
    receipt = {
        "created_utc": utc_now(), "script_sha256": digest(Path(__file__)),
        "evaluation_only": True, "training_evidence_modified": False,
        "selection_metric": "lowest matched-context validation cross_entropy",
        "best_validation_candidate": ranked[0]["run"],
        "promotion_rule": {"reference_run": reference["run"],
                           "minimum_cross_entropy_improvement_nats": minimum_improvement,
                           "observed_improvement_nats": improvement,
                           "candidate_qualifies": qualifies,
                           "preferred_run": ranked[0]["run"] if qualifies else reference["run"],
                           "automatic_publication_performed": False},
        "interpretation": "Validation comparison for model development; not an untouched test estimate or a guarantee of generation quality.",
        "protocol": {"context_length": context_length, "batch_size": batch_size,
                     "window_rule": "story-bounded nonoverlapping targets; pad tails; score all characters and EOS exactly once",
                     "model_mode": "eval; dropout disabled", "device": str(target),
                     "device_name": torch.cuda.get_device_name(target) if amp else "CPU",
                     "torch_version": torch.__version__,
                     "precision": str(dtype) if amp else "torch.float32",
                     "manifest_sha256": bundles[0]["summary"]["manifest_sha256"],
                     "data": {split: {"stories": len(dataset.encoded), "windows": len(dataset),
                                      "scored_targets_including_eos": dataset.target_count,
                                      "unknown_characters": dataset.unknown_characters}
                              for split, dataset in data.items()}},
        "candidates": candidates,
        "ranking": [row["run"] for row in ranked],
    }
    gpt._json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-improvement", type=float, default=0.005,
                        help="Required validation CE reduction from the first --run-dir (default .005 nats)")
    parser.add_argument("--include-train", action="store_true",
                        help="Also evaluate all training stories without dropout for a matched generalization gap")
    args = parser.parse_args()
    compare(args.run_dir, args.output, context_length=args.context_length,
            batch_size=args.batch_size, device=args.device, include_train=args.include_train,
            minimum_improvement=args.minimum_improvement)


if __name__ == "__main__":
    main()
