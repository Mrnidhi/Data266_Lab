"""Run a predeclared, validation-only CycleGAN continuation experiment.

Run from any working directory. ``prepare`` freezes two recipes and the source
checkpoint identity. ``train-arm`` performs only the named arm, with an explicit
resume required after a capped rehearsal or interruption. ``select`` compares
both completed arms and the unchanged source using validation KID; it never
evaluates the test split, exports images, publishes results, or rents hardware.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import Tee, code_manifest, environment, resolve_device, utc_now, write_json

ARMS = {"identity5": 5.0, "identity2p5": 2.5}
DIRECTIONS = ("photo_to_monet", "monet_to_photo")
METRIC = "mean_validation_KID_both_directions"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def portable_path(path):
    path = Path(path).expanduser().resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


@contextlib.contextmanager
def project_directory():
    old = Path.cwd()
    try:
        os.chdir(ROOT)
        yield
    finally:
        os.chdir(old)


def prepare(output, checkpoint, config_path, expected_sha256=None):
    output, checkpoint = Path(output).resolve(), Path(checkpoint).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepare requires a new, empty experiment directory")
    source_hash = digest(checkpoint)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("Source checkpoint differs from the expected SHA-256")
    saved = load_checkpoint(checkpoint)
    base = read_json(config_path)
    base = copy.deepcopy(base.get("full", base))
    if saved.get("format_version") != 1 or saved["config"].get("mode") != "full":
        raise ValueError("The source must be a trusted completed full CycleGAN checkpoint")
    for key in ("image_size", "base_channels", "residual_blocks", "batch_size"):
        if base.get(key) != saved["config"].get(key):
            raise ValueError(f"Source/config architecture mismatch: {key}")
    if base.get("seed") != 2342 or base.get("batch_size") != 1:
        raise ValueError("This protocol fixes seed=2342 and batch_size=1")
    if not all(base.get("data", {}).get(key) for key in ("monet_dir", "photo_dir", "manifest_dir")):
        raise ValueError("All original class data and frozen manifest paths are required")
    fingerprint = saved["state"].get("manifest_fingerprint", "")
    if not fingerprint or fingerprint.startswith("synthetic:"):
        raise ValueError("The source must use real class data")
    base.update(mode="full", seed=2342, learning_rate=5e-5, epochs=10, constant_epochs=5,
                precision="bf16", replay_device="device", max_steps=None, allow_synthetic=False,
                select_best=True, validation_every_epochs=2, evaluate_after_run=False)
    base.setdefault("evaluation", {}).update(allow_metric_downloads=True, max_images=None, seed=2342)
    if digest(checkpoint) != source_hash:
        raise ValueError("Source checkpoint changed during preparation")
    configs = {arm: dict(copy.deepcopy(base), identity_weight=weight) for arm, weight in ARMS.items()}
    protocol = {
        "format_version": 1, "created_utc": utc_now(), "status": "prepared_not_trained",
        "source_checkpoint": portable_path(checkpoint), "source_checkpoint_sha256": source_hash,
        "source_global_step": saved["state"]["global_step"],
        "source_manifest_fingerprint": fingerprint, "source_config": saved["config"],
        "arms": {}, "selection": {"split": "val", "metric": METRIC, "lower_is_better": True,
            "include_unchanged_source": True, "ties_favor_unchanged_source": True,
            "require_all_arms_complete": True, "visual_review_required_before_publication": True},
        "training": {"additional_epochs_per_arm": 10, "constant_lr_epochs": 5,
            "linear_decay_epochs": 5, "learning_rate": 5e-5, "seed": 2342,
            "validation_epochs": [0, 2, 4, 6, 8, 10],
            "restore": "all four source network weights",
            "reset": ["optimizers", "schedulers", "replay_pools", "rng", "global_step", "selection"]},
        "test_policy": "No test evaluation or publication during prepare/train-arm/select.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for arm, config in configs.items():
        path = output / "configs" / f"{arm}.json"
        write_json(path, config)
        protocol["arms"][arm] = {"config": str(path.relative_to(output)), "config_sha256": digest(path),
                                 "run_dir": f"arms/{arm}", "identity_weight": ARMS[arm]}
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n")
    return protocol


def load_protocol(output, checkpoint=None):
    output = Path(output).resolve()
    if digest(output / "protocol.json") != (output / "protocol.sha256").read_text().strip():
        raise ValueError("The frozen protocol was modified")
    protocol = read_json(output / "protocol.json")
    if protocol.get("format_version") != 1 or set(protocol.get("arms", {})) != set(ARMS):
        raise ValueError("Unsupported tuning protocol")
    for arm, entry in protocol["arms"].items():
        if entry["config"] != f"configs/{arm}.json" or entry["run_dir"] != f"arms/{arm}":
            raise ValueError("Invalid arm paths in protocol")
        if digest(output / entry["config"]) != entry["config_sha256"]:
            raise ValueError(f"The frozen {arm} recipe was modified")
    source = Path(checkpoint).expanduser().resolve() if checkpoint else resolve_path(protocol["source_checkpoint"])
    if digest(source) != protocol["source_checkpoint_sha256"]:
        raise ValueError("The frozen source checkpoint was modified or a different file was supplied")
    return output, protocol, source


def train_arm(output, arm, device="cuda", checkpoint=None, resume=None, max_steps=None):
    from lab1 import cyclegan as cg
    output, protocol, source = load_protocol(output, checkpoint)
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    config = read_json(output / protocol["arms"][arm]["config"])
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive cumulative update count")
    config["max_steps"] = max_steps
    run = output / protocol["arms"][arm]["run_dir"]
    if (run / "run_summary.json").exists() and read_json(run / "run_summary.json").get("status") == "completed":
        raise FileExistsError("This arm is already complete; it will not restart")
    if resume is None and run.exists() and any(run.iterdir()):
        raise FileExistsError("Existing arm requires explicit --resume to preserve its evidence")
    if resume is not None:
        resume = Path(resume).expanduser().resolve()
        if resume != (run / "last.pt").resolve():
            raise ValueError("Resume requires this arm's own last.pt checkpoint")
        restored = load_checkpoint(resume)
        if restored["state"].get("initialization", {}).get("checkpoint_sha256") != protocol["source_checkpoint_sha256"]:
            raise ValueError("Resume checkpoint is not initialized from the frozen source")
        prior = read_json(run / "resolved_config.json")
        if any(prior.get(key) != value for key, value in config.items() if key != "max_steps"):
            raise ValueError("Resume must preserve the frozen arm recipe")
        del restored
    device = resolve_device(device)
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This arm already has a live runner") from error
        manifest = {"task": "cyclegan", "mode": "full", "device": device,
            "started_utc": utc_now(), "status": "running", "is_final_result": False,
            "config": config, "source_sha256": code_manifest(), "environment": environment(),
            "resumed": resume is not None, "tuning_arm": arm,
            "protocol_sha256": digest(output / "protocol.json"),
            "warm_start_source_sha256": protocol["source_checkpoint_sha256"],
            "invocation": shlex.join([sys.executable, *sys.argv])}
        manifest["source_sha256"]["scripts/tune_cyclegan.py"] = digest(__file__)
        provenance = run / "provenance" / f"{manifest['started_utc'].replace(':', '-')}.json"
        write_json(provenance, manifest)
        write_json(run / "run_summary.json", manifest)
        started = time.perf_counter()
        with (run / "RUN_LOG.txt").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(json.dumps({"event": "start", "arm": arm, "device": device, "resumed": resume is not None,
                                  "source_checkpoint_sha256": protocol["source_checkpoint_sha256"]}), flush=True)
                try:
                    with project_directory():
                        summary = cg.run(config, run, device, resume=resume,
                                         warm_start=None if resume else source)
                    resolved = read_json(run / "resolved_config.json")
                    manifest["config"] = resolved
                    full_updates = config["epochs"] * summary["steps_per_epoch"]
                    completed = (summary["completed_updates"] == full_updates
                                 and summary["completed_epoch_fraction"] == config["epochs"])
                    manifest.update(status="completed" if completed else "partial", summary=summary)
                    if digest(source) != protocol["source_checkpoint_sha256"]:
                        raise ValueError("Source checkpoint changed during training")
                    print(json.dumps({"event": manifest["status"], "arm": arm,
                                      "completed_updates": summary["completed_updates"], "full_updates": full_updates}), flush=True)
                except BaseException as error:
                    manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
                    traceback.print_exc()
                    raise
                finally:
                    manifest.update(ended_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
                    write_json(run / "run_summary.json", manifest)
                    write_json(provenance, manifest)
    return manifest


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite computed {label}")
    return value


def validation_kid(path, manifest):
    result = read_json(path)
    if (result.get("split") != "val" or result.get("class_results") is not True
            or result.get("data_kind") != "explicit_class_manifests"
            or result.get("manifest_fingerprint") != manifest["manifest_fingerprint"]
            or set(result.get("directions", {})) != set(DIRECTIONS)):
        raise ValueError("Selection requires validation-only metrics on the frozen class manifests")
    scores = {}
    for direction, entry in result["directions"].items():
        source, target = direction.split("_to_")
        if (entry.get("source_count") != manifest["counts"][f"val_{source}"]
                or entry.get("generated_count") != manifest["counts"][f"val_{source}"]
                or entry.get("real_reference_count") != manifest["counts"][f"val_{target}"]):
            raise ValueError("Selection requires every validation image in both directions")
        kid = entry.get("metrics", {}).get("kid", {})
        if kid.get("status") != "computed":
            raise ValueError("Both validation KID values must be computed")
        scores[direction] = finite(kid.get("value"), f"{direction} KID")
    return {"value": sum(scores.values()) / 2, "directions": scores,
            "metrics_path": str(path), "metrics_sha256": digest(path)}


def select(output, checkpoint=None):
    output, protocol, source = load_protocol(output, checkpoint)
    candidates, baselines = [], []
    for arm, spec in protocol["arms"].items():
        run = output / spec["run_dir"]
        wrapper = read_json(run / "run_summary.json")
        summary, config = wrapper.get("summary", {}), read_json(run / "resolved_config.json")
        planned = read_json(output / spec["config"])
        manifest = read_json(run / "data_manifest.json")
        expected = config["epochs"] * max(manifest["counts"]["train_photo"], manifest["counts"]["train_monet"])
        if (wrapper.get("status") != "completed" or wrapper.get("task") != "cyclegan"
                or wrapper.get("mode") != "full" or wrapper.get("config") != config
                or not wrapper.get("source_sha256") or not wrapper.get("environment")
                or summary.get("completed_updates") != expected or summary.get("completed_epoch_fraction") != 10
                or summary.get("nan_events") != 0
                or any(config.get(key) != value for key, value in planned.items() if key != "max_steps")):
            raise ValueError(f"Selection requires the complete predeclared 10-epoch {arm} arm without NaNs")
        if (manifest.get("manifest_fingerprint") != protocol["source_manifest_fingerprint"]
                or summary.get("initialization", {}).get("checkpoint_sha256") != protocol["source_checkpoint_sha256"]):
            raise ValueError("Arm/source provenance differs from the frozen protocol")
        baseline = validation_kid(run / "validation/step_0000000/metrics.json", manifest)
        baseline["arm"] = arm
        baselines.append(baseline)
        selection = summary.get("best_selection", {})
        step = selection.get("step")
        if (selection.get("metric") != METRIC or selection.get("validation_only") is not True
                or isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= expected):
            raise ValueError("Missing validation-only best checkpoint selection")
        result = validation_kid(run / "validation" / f"step_{step:07d}" / "metrics.json", manifest)
        if not math.isclose(result["value"], finite(selection.get("value"), "selection KID"), abs_tol=1e-12):
            raise ValueError("Best selection score differs from validation metrics")
        best = run / "best.pt"
        saved = load_checkpoint(best)
        state = saved.get("state", {})
        if (state.get("global_step") != step or state.get("manifest_fingerprint") != manifest["manifest_fingerprint"]
                or state.get("best_selection") != selection or state.get("best_validation_score") != selection["value"]
                or state.get("initialization", {}).get("checkpoint_sha256") != protocol["source_checkpoint_sha256"]
                or any(saved.get("config", {}).get(key) != value for key, value in config.items() if key != "max_steps")):
            raise ValueError("Selected checkpoint differs from saved validation evidence")
        del saved
        candidates.append(dict(result, kind="arm", arm=arm, step=step, checkpoint=str(best), checkpoint_sha256=digest(best)))
    baseline = baselines[0]
    if any(not math.isclose(item["directions"][d], baseline["directions"][d], abs_tol=1e-6, rel_tol=1e-6)
           for item in baselines[1:] for d in DIRECTIONS):
        raise ValueError("Unchanged-source validation scores disagree between arms; investigate before selection")
    unchanged = dict(baseline, kind="unchanged_source", arm=None, step=protocol["source_global_step"],
                     checkpoint=str(source), checkpoint_sha256=protocol["source_checkpoint_sha256"])
    # Step-zero arm files wrap unchanged weights with a new schedule: retain the
    # original source file instead of publishing it as a fine-tuned improvement.
    ranking = sorted([unchanged, *(row for row in candidates if row["step"] > 0)],
                     key=lambda row: (row["value"], row["kind"] != "unchanged_source", row.get("arm") or ""))
    result = {"created_utc": utc_now(), "status": "validation_selected_pending_visual_review",
              "split": "val", "metric": METRIC, "protocol_sha256": digest(output / "protocol.json"),
              "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
              "baseline_rechecks": baselines, "completed_arm_candidates": candidates,
              "ranking": ranking, "winner": ranking[0],
              "mean_kid_improvement": unchanged["value"] - ranking[0]["value"],
              "test_evaluated": False, "published": False}
    write_json(output / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("prepare", "train-arm", "select"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Source; can relocate it only if its frozen SHA-256 matches")
    parser.add_argument("--expected-sha256", help="Additional source identity check for prepare")
    parser.add_argument("--config", type=Path, default=ROOT / "task3_gan/srinidhi/config.json")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="Cumulative update cap within the unchanged 10-epoch schedule")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.checkpoint is None:
            parser.error("prepare requires --checkpoint")
        result = prepare(args.output, args.checkpoint, args.config, args.expected_sha256)
    elif args.action == "train-arm":
        if args.arm is None:
            parser.error("train-arm requires --arm")
        result = train_arm(args.output, args.arm, args.device, args.checkpoint, args.resume, args.max_steps)
    else:
        result = select(args.output, args.checkpoint)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
