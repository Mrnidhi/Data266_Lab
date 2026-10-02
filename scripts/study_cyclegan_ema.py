"""Frozen two-arm LR/EMA study; validation-only selection, never class/test scoring."""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import Tee, environment, resolve_device, utc_now, write_json

ARMS = {"lr5e5": 5e-5, "lr2e5": 2e-5}
DIRECTIONS = ("photo_to_monet", "monet_to_photo")
VARIANTS = ("raw", "ema")
METRIC = "mean_validation_KID_both_directions"
SOURCES = ("scripts/study_cyclegan_ema.py", "task3_gan/srinidhi/src/cyclegan_ema.py",
           "task3_gan/srinidhi/src/cyclegan.py", "src/lab1/common.py", "src/lab1/__init__.py")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite {label}")
    return value


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def portable_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def relative_path(path, output):
    return Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix()


def same_recipe(actual, planned):
    return {k: v for k, v in actual.items() if k != "max_steps"} == {k: v for k, v in planned.items() if k != "max_steps"}


@contextlib.contextmanager
def runner_lock(path):
    with Path(path).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if not lock.tell():
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise RuntimeError("Study/arm already has a live runner") from error
                raise
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Study/arm already has a live runner") from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


@contextlib.contextmanager
def project_directory():
    old = Path.cwd()
    try:
        os.chdir(ROOT)
        yield
    finally:
        os.chdir(old)


def verify_models(saved):
    import torch
    expected = {"G_photo_to_monet", "G_monet_to_photo", "D_photo", "D_monet"}
    if set(saved.get("models", {})) != expected:
        raise ValueError("Checkpoint must contain the paired generators and both discriminators")
    for name, state in saved["models"].items():
        for key, tensor in state.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite checkpoint tensor: {name}/{key}")


def prepare(output, checkpoint, config_path, expected_sha256):
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepare requires a new, empty study directory")
    if digest(checkpoint) != expected_sha256:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    saved = load_checkpoint(checkpoint)
    if saved.get("format_version") != 1 or saved.get("config", {}).get("mode") != "full":
        raise ValueError("A trusted full CycleGAN checkpoint is required")
    verify_models(saved)
    base = read_json(config_path)
    base = copy.deepcopy(base.get("full", base))
    for key in ("image_size", "base_channels", "residual_blocks", "batch_size"):
        if base.get(key) != saved["config"].get(key):
            raise ValueError(f"Source/config architecture mismatch: {key}")
    fingerprint = saved["state"].get("manifest_fingerprint", "")
    if len(fingerprint) != 64 or fingerprint.startswith("synthetic:"):
        raise ValueError("Original real-data manifest fingerprint is required")
    base.update(mode="full", seed=2342, epochs=8, constant_epochs=4, batch_size=1,
                cycle_weight=10.0, identity_weight=2.5, betas=[0.5, 0.999], replay_size=50,
                precision="bf16", replay_device="device", max_steps=None, allow_synthetic=False,
                monet_translation_ratio=0.0, ema_beta=0.999, select_best=False,
                validation_every_epochs=4, validation_epochs=[4, 8], evaluate_after_run=False,
                expected_source_sha256=expected_sha256, source_checkpoint=portable_path(checkpoint))
    base.setdefault("evaluation", {}).update(allow_metric_downloads=True, max_images=None,
        seed=2342, kid_subsets=50, kid_subset_size=100, prdc_k=5, batch_size=16)
    if not all(base.get("data", {}).get(k) for k in ("monet_dir", "photo_dir", "manifest_dir")):
        raise ValueError("All original data paths and frozen manifests are required")
    source_hashes = {path: digest(ROOT / path) for path in SOURCES}
    protocol = {"format_version": 1, "experiment": "lr_ema", "created_utc": utc_now(),
        "source_checkpoint": portable_path(checkpoint), "source_checkpoint_sha256": expected_sha256,
        "source_global_step": saved["state"]["global_step"], "source_manifest_fingerprint": fingerprint,
        "source_code_hashes": source_hashes, "arms": {},
        "training": {"epochs": 8, "constant_epochs": 4, "ema_beta": .999, "seed": 2342,
                     "validation_epochs": [0, 4, 8], "candidate_epochs": [4, 8], "variants": list(VARIANTS)},
        "selection": {"split": "val", "metric": METRIC, "lower_is_better": True,
            "max_candidates": 9, "include_unchanged_source": True, "require_both_arms_complete": True,
            "directional_kid_guardrail": "each_direction <= unchanged_source_direction + tolerance",
            "guardrail_tolerance": 1e-8, "exact_ties_favor_unchanged_source": True,
            "baseline_agreement_abs_tolerance": 1e-6, "baseline_agreement_rel_tolerance": 1e-6,
            "secondary_tie_order": "arm name, earlier step, raw before ema",
            "mix_generators_across_candidates": False, "visual_review_required_before_publication": True},
        "budget": {"total_seconds": 21600, "final_evaluation_reserve_seconds": 900,
            "start": "first train-arm invocation", "automatic_additional_trials": False,
            "deadline_semantics": "Cooperative: check before updates/validation; an in-progress metric call may finish. Reserve final 15 minutes for evaluation/package; external finalizer must honor total deadline."},
        "test_policy": "No test, class-evaluator or leaderboard feedback during prepare/train-arm/select."}
    output.mkdir(parents=True, exist_ok=True)
    for path, expected in source_hashes.items():
        target = output / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
        if digest(target) != expected or digest(ROOT / path) != expected:
            raise ValueError("Source code changed during preparation")
    for arm, lr in ARMS.items():
        config = dict(copy.deepcopy(base), learning_rate=lr)
        path = output / "configs" / f"{arm}.json"
        write_json(path, config)
        protocol["arms"][arm] = {"config": f"configs/{arm}.json", "config_sha256": digest(path),
                                 "run_dir": f"arms/{arm}", "learning_rate": lr}
    if digest(checkpoint) != expected_sha256:
        raise ValueError("Source checkpoint changed during preparation")
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n")
    return protocol


def load_protocol(output, checkpoint=None):
    output = Path(output).resolve()
    if digest(output / "protocol.json") != (output / "protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol modified")
    protocol = read_json(output / "protocol.json")
    if protocol.get("format_version") != 1 or protocol.get("experiment") != "lr_ema" or set(protocol.get("arms", {})) != set(ARMS):
        raise ValueError("Unsupported study protocol")
    for path, expected in protocol["source_code_hashes"].items():
        if digest(ROOT / path) != expected or digest(output / "source_snapshot" / path) != expected:
            raise ValueError(f"Frozen source code modified: {path}")
    if set(protocol["source_code_hashes"]) != set(SOURCES):
        raise ValueError("Incomplete source-code manifest")
    for arm, entry in protocol["arms"].items():
        if entry["config"] != f"configs/{arm}.json" or entry["run_dir"] != f"arms/{arm}":
            raise ValueError("Invalid arm paths")
        if digest(output / entry["config"]) != entry["config_sha256"]:
            raise ValueError(f"Frozen recipe modified: {arm}")
    source = Path(checkpoint).resolve() if checkpoint else resolve_path(protocol["source_checkpoint"])
    if digest(source) != protocol["source_checkpoint_sha256"]:
        raise ValueError("Frozen source checkpoint modified")
    return output, protocol, source


def budget_receipt(output, protocol):
    path = output / "budget.json"
    if not path.exists():
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(seconds=protocol["budget"]["total_seconds"])
        write_json(path, {"started_utc": now.isoformat(), "deadline_utc": deadline.isoformat(),
            "training_deadline_utc": (deadline - timedelta(seconds=protocol["budget"]["final_evaluation_reserve_seconds"])).isoformat(),
            "protocol_sha256": digest(output / "protocol.json")})
        (output / "budget.sha256").write_text(digest(path) + "\n")
    budget = read_budget(output, protocol)
    if datetime.now(timezone.utc) >= datetime.fromisoformat(budget["training_deadline_utc"]):
        raise TimeoutError("Study training budget exhausted; no further automatic trial")
    return budget


def read_budget(output, protocol=None):
    """Verify the frozen six-hour clock without imposing the training-only cutoff."""
    output = Path(output).resolve()
    if protocol is None:
        output, protocol, _ = load_protocol(output)
    path = output / "budget.json"
    if digest(path) != (output / "budget.sha256").read_text().strip():
        raise ValueError("Frozen budget receipt modified")
    budget = read_json(path)
    if budget.get("protocol_sha256") != digest(output / "protocol.json"):
        raise ValueError("Budget/protocol mismatch")
    try:
        started = datetime.fromisoformat(budget["started_utc"])
        deadline = datetime.fromisoformat(budget["deadline_utc"])
        training_deadline = datetime.fromisoformat(budget["training_deadline_utc"])
        valid = (all(value.tzinfo is not None for value in (started, deadline, training_deadline))
                 and (deadline - started).total_seconds() == protocol["budget"]["total_seconds"]
                 and (deadline - training_deadline).total_seconds() == protocol["budget"]["final_evaluation_reserve_seconds"])
    except (KeyError, ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Budget timestamps differ from frozen six-hour protocol")
    return budget


def train_arm(output, arm, device="cuda", checkpoint=None, resume=None, max_steps=None, wall_budget_seconds=None):
    from lab1 import cyclegan_ema as trainer
    output, protocol, source = load_protocol(output, checkpoint)
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive cumulative update count")
    if wall_budget_seconds is not None and finite(wall_budget_seconds, "wall budget seconds") <= 0:
        raise ValueError("Wall budget seconds must be positive")
    run = output / protocol["arms"][arm]["run_dir"]
    config = read_json(output / protocol["arms"][arm]["config"])
    config["max_steps"] = max_steps
    with runner_lock(output / ".study_runner.lock"):
        if (run / "training_failure.json").exists():
            raise ValueError("Recorded training failure must be investigated; evidence preserved")
        prior = read_json(run / "run_summary.json") if (run / "run_summary.json").exists() else None
        if prior and prior.get("status") in ("completed", "failed"):
            raise FileExistsError(f"Arm already {prior['status']}; no restart")
        if resume is None and run.exists() and any(run.iterdir()):
            raise FileExistsError("Existing arm requires explicit --resume")
        if resume is not None:
            resume = Path(resume).resolve()
            if resume != (run / "last.pt").resolve():
                raise ValueError("Resume requires this arm's own last.pt")
            restored = load_checkpoint(resume)
            if (restored.get("trainer") != "cyclegan_ema_v1" or restored.get("inference_only") is not False
                    or restored.get("state", {}).get("initialization", {}).get("checkpoint_sha256") != protocol["source_checkpoint_sha256"]
                    or not same_recipe(restored.get("config", {}), config)
                    or not same_recipe(read_json(run / "resolved_config.json"), config)):
                raise ValueError("Resume checkpoint/config differs from frozen arm")
            del restored
        budget = budget_receipt(output, protocol)
        deadline = datetime.fromisoformat(budget["training_deadline_utc"])
        if wall_budget_seconds is not None:
            deadline = min(deadline, datetime.now(timezone.utc) + timedelta(seconds=wall_budget_seconds))
        device = resolve_device(device)
        run.mkdir(parents=True, exist_ok=True)
        with runner_lock(run / ".runner.lock"):
            manifest = {"task": "cyclegan_ema", "mode": "full", "arm": arm, "status": "running",
                "started_utc": utc_now(), "config": config, "device": device, "resumed": resume is not None,
                "protocol_sha256": digest(output / "protocol.json"), "budget_sha256": digest(output / "budget.json"),
                "source_sha256": protocol["source_code_hashes"], "environment": environment(),
                "warm_start_source_sha256": protocol["source_checkpoint_sha256"],
                "operational_deadline_utc": deadline.isoformat(),
                "invocation": shlex.join([sys.executable, *sys.argv]), "is_final_result": False}
            provenance_path = run / "provenance" / f"{manifest['started_utc'].replace(':', '-')}.json"
            write_json(provenance_path, manifest)
            write_json(run / "run_summary.json", manifest)
            started = time.perf_counter()
            with (run / "RUN_LOG.txt").open("a", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    try:
                        with project_directory():
                            summary = trainer.run(config, run, device, resume=resume,
                                warm_start=None if resume else source, deadline_utc=deadline.isoformat())
                        full_steps = 8 * summary["steps_per_epoch"]
                        completed = (summary.get("status") == "completed" and not summary.get("pending_validation")
                                     and summary["completed_updates"] == full_steps and summary["completed_epoch_fraction"] == 8)
                        if summary.get("nan_events") != 0:
                            raise ValueError("Trainer reports nonfinite events")
                        manifest.update(status="completed" if completed else "partial", summary=summary)
                        manifest["config"] = read_json(run / "resolved_config.json")
                        if not same_recipe(manifest["config"], config):
                            raise ValueError("Trainer changed the frozen recipe")
                        load_protocol(output, source)
                    except BaseException as error:
                        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
                        traceback.print_exc()
                        raise
                    finally:
                        manifest.update(ended_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
                        write_json(run / "run_summary.json", manifest)
                        write_json(provenance_path, manifest)
    return manifest


def validation_kid(path, manifest, output):
    result = read_json(path)
    if (result.get("split") != "val" or result.get("class_results") is not True
            or result.get("data_kind") != "explicit_class_manifests"
            or result.get("manifest_fingerprint") != manifest["manifest_fingerprint"]
            or set(result.get("directions", {})) != set(DIRECTIONS)):
        raise ValueError("Validation-only evidence on the frozen manifests is required")
    scores = {}
    for direction, entry in result["directions"].items():
        source, target = direction.split("_to_")
        if (entry.get("source_count") != manifest["counts"][f"val_{source}"]
                or entry.get("generated_count") != manifest["counts"][f"val_{source}"]
                or entry.get("real_reference_count") != manifest["counts"][f"val_{target}"]):
            raise ValueError("Every validation image in both directions is required")
        kid = entry.get("metrics", {}).get("kid", {})
        if kid.get("status") != "computed":
            raise ValueError("Both directional KID values must be computed")
        scores[direction] = finite(kid.get("value"), "validation KID")
    return {"value": sum(scores.values()) / 2, "directions": scores,
            "metrics_path": relative_path(path, output), "metrics_sha256": digest(path)}


def validation_receipt(directory, checkpoint, step, variant, row, protocol, output):
    receipt = read_json(directory / "receipt.json")
    metrics = read_json(directory / "metrics.json")
    checkpoint_hash = digest(checkpoint)
    if (receipt.get("status") != "complete" or receipt.get("trainer") != "cyclegan_ema_v1"
            or receipt.get("checkpoint_sha256") != checkpoint_hash or receipt.get("metrics_sha256") != row["metrics_sha256"]
            or receipt.get("candidate_variant") != variant or receipt.get("step") != step
            or receipt.get("source_checkpoint_sha256") != protocol["source_checkpoint_sha256"]
            or receipt.get("manifest_fingerprint") != protocol["source_manifest_fingerprint"]
            or receipt.get("directional_kid") != row["directions"]
            or finite(receipt.get("mean_validation_kid"), "receipt mean KID") != row["value"]
            or Path(receipt.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or Path(receipt.get("metrics_path", "")).resolve() != (directory / "metrics.json").resolve()
            or metrics.get("checkpoint_sha256") != checkpoint_hash or metrics.get("checkpoint_step") != step
            or metrics.get("candidate_variant") != variant or metrics.get("ema_beta") != .999
            or metrics.get("ema_updates") != step):
        raise ValueError("Validation receipt differs from checkpoint/validation evidence")
    row.update(receipt_path=relative_path(directory / "receipt.json", output),
               receipt_sha256=digest(directory / "receipt.json"))
    return receipt


def verify_ended(wrapper, final):
    try:
        started = datetime.fromisoformat(wrapper["started_utc"])
        ended = datetime.fromisoformat(wrapper["ended_utc"])
        valid = started.tzinfo is not None and ended.tzinfo is not None and ended >= started
    except (KeyError, TypeError, ValueError):
        valid = False
    if wrapper != final or wrapper.get("status") != "completed" or not valid or finite(wrapper.get("elapsed_seconds"), "elapsed seconds") < 0:
        raise ValueError("Matching ended completed wrapper/final provenance receipts are required")


def verify_training_log(path, expected):
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise ValueError("Incomplete final training log record")
            row = json.loads(line)
            count += 1
            if row.get("step") != count or row.get("nan_events") != 0:
                raise ValueError("Training log contains gaps/duplicate steps/nonfinite events")
            for group in ("losses", "gradient_norms"):
                if not isinstance(row.get(group), dict) or not row[group]:
                    raise ValueError("Training log lacks losses/gradient norms")
                for value in row[group].values():
                    finite(value, f"logged {group}")
    if count != expected:
        raise ValueError("Training log update count does not match completed run")
    return digest(path)


def select(output, checkpoint=None):
    output, protocol, source = load_protocol(output, checkpoint)
    with runner_lock(output / ".study_runner.lock"):
        read_budget(output, protocol)
        rows, baselines, arm_receipts = [], [], {}
        for arm, spec in protocol["arms"].items():
            run = output / spec["run_dir"]
            if (run / "training_failure.json").exists():
                raise ValueError("Recorded training failure blocks selection")
            wrapper = read_json(run / "run_summary.json")
            provenance_paths = sorted((run / "provenance").glob("*.json"))
            if not provenance_paths:
                raise ValueError("Completed provenance is missing")
            verify_ended(wrapper, read_json(provenance_paths[-1]))
            config, planned = read_json(run / "resolved_config.json"), read_json(output / spec["config"])
            manifest = read_json(run / "data_manifest.json")
            summary = wrapper.get("summary", {})
            steps_per_epoch = max(manifest["counts"]["train_photo"], manifest["counts"]["train_monet"])
            if (wrapper.get("task") != "cyclegan_ema" or wrapper.get("arm") != arm
                    or wrapper.get("config") != config or not same_recipe(config, planned)
                    or wrapper.get("source_sha256") != protocol["source_code_hashes"]
                    or wrapper.get("protocol_sha256") != digest(output / "protocol.json")
                    or wrapper.get("budget_sha256") != digest(output / "budget.json")
                    or wrapper.get("warm_start_source_sha256") != protocol["source_checkpoint_sha256"]
                    or not wrapper.get("environment") or summary.get("completed_updates") != 8 * steps_per_epoch
                    or summary.get("completed_epoch_fraction") != 8 or summary.get("nan_events") != 0
                    or summary.get("status") != "completed" or summary.get("pending_validation")
                    or summary.get("trainer") != "cyclegan_ema_v1" or summary.get("steps_per_epoch") != steps_per_epoch
                    or summary.get("ema_beta") != .999 or summary.get("ema_updates") != 8 * steps_per_epoch
                    or manifest.get("manifest_fingerprint") != protocol["source_manifest_fingerprint"]
                    or summary.get("initialization", {}).get("checkpoint_sha256") != protocol["source_checkpoint_sha256"]):
                raise ValueError("Selection requires both complete frozen eight-epoch arms without nonfinite failures")
            if read_json(run / "trainer_summary.json") != summary:
                raise ValueError("Trainer/wrapper summaries differ")
            baseline = validation_kid(run / "validation/step_0000000/raw/metrics.json", manifest, output)
            receipts = [validation_receipt(run / "validation/step_0000000/raw", source, 0, "raw", baseline, protocol, output)]
            baselines.append(dict(baseline, arm=arm))
            arm_receipts[arm] = {"run_summary_sha256": digest(run / "run_summary.json"),
                                 "final_provenance_sha256": digest(provenance_paths[-1]), "completed_updates": 8 * steps_per_epoch,
                                 "trainer_summary_sha256": digest(run / "trainer_summary.json"),
                                 "provenance": [{"path": relative_path(path, output), "sha256": digest(path)} for path in provenance_paths],
                                 "training_log_sha256": verify_training_log(run / "training_log.jsonl", 8 * steps_per_epoch)}
            for epoch in (4, 8):
                step = epoch * steps_per_epoch
                for variant in VARIANTS:
                    directory = run / "validation" / f"step_{step:07d}" / variant
                    row = validation_kid(directory / "metrics.json", manifest, output)
                    candidate = run / "candidates" / f"step_{step:07d}" / f"{variant}.pt"
                    candidate_hash = digest(candidate)
                    receipts.append(validation_receipt(directory, candidate, step, variant, row, protocol, output))
                    saved = load_checkpoint(candidate)
                    state = saved.get("state", {})
                    if (saved.get("format_version") != 1 or saved.get("inference_only") is not True
                            or saved.get("trainer") != "cyclegan_ema_v1"
                            or any(key in saved for key in ("optimizers", "schedulers", "replay_pools", "rng"))
                            or not same_recipe(saved.get("config", {}), config)
                            or state.get("global_step") != step or state.get("candidate_variant") != variant
                            or state.get("ema_beta") != .999 or state.get("ema_updates") != step
                            or state.get("manifest_fingerprint") != protocol["source_manifest_fingerprint"]
                            or state.get("initialization", {}).get("checkpoint_sha256") != protocol["source_checkpoint_sha256"]):
                        raise ValueError("Candidate checkpoint differs from frozen recipe/provenance")
                    verify_models(saved)
                    del saved
                    if digest(candidate) != candidate_hash:
                        raise ValueError("Candidate changed during selection")
                    rows.append(dict(row, kind="arm", arm=arm, step=step, epoch=epoch, variant=variant,
                                     checkpoint=relative_path(candidate, output), checkpoint_sha256=candidate_hash))
            if summary.get("validation_history") != receipts:
                raise ValueError("Trainer validation history differs from the five required receipts")
        baseline = baselines[0]
        if any(not math.isclose(b["directions"][d], baseline["directions"][d], abs_tol=1e-6, rel_tol=1e-6) for b in baselines[1:] for d in DIRECTIONS):
            raise ValueError("Unchanged-source validation disagrees between arms")
        source_row = dict(baseline, kind="unchanged_source", arm=None, variant="source", epoch=None,
                          step=protocol["source_global_step"], checkpoint=relative_path(source, output),
                          checkpoint_sha256=protocol["source_checkpoint_sha256"])
        candidates = [source_row, *rows]
        for row in candidates:
            row["eligible"] = row["kind"] == "unchanged_source" or all(row["directions"][d] <= baseline["directions"][d] + 1e-8 for d in DIRECTIONS)
            row["guardrail_deltas"] = {d: row["directions"][d] - baseline["directions"][d] for d in DIRECTIONS}
            row["exclusion_reason"] = None if row["eligible"] else "At least one directional validation KID exceeds incumbent + 1e-8"
        ranking = sorted([row for row in candidates if row["eligible"]],
            key=lambda r: (r["value"], r["kind"] != "unchanged_source", r["arm"] or "", r["step"], r["variant"] == "ema"))
        if len(candidates) != 9:
            raise ValueError("Exactly the nine predeclared candidate slots are required")
        load_protocol(output, source)
        result = {"format_version": 1, "created_utc": utc_now(), "status": "validation_selected_pending_visual_review",
            "split": "val", "metric": METRIC, "protocol_sha256": digest(output / "protocol.json"),
            "source_checkpoint_sha256": protocol["source_checkpoint_sha256"], "source_code_hashes": protocol["source_code_hashes"],
            "budget_sha256": digest(output / "budget.json"),
            "path_base": "selection_json_directory", "baseline_rechecks": baselines,
            "arm_completion_receipts": arm_receipts, "all_candidates": candidates, "ranking": ranking, "winner": ranking[0],
            "guardrail": protocol["selection"], "mean_kid_improvement": source_row["value"] - ranking[0]["value"],
            "test_evaluated": False, "class_evaluated": False, "published": False}
        write_json(output / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "train-arm", "select"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--config", type=Path, default=ROOT / "task3_gan/srinidhi/config.json")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--wall-budget-seconds", type=float, help="Optional stricter operational time allowance; leaves frozen epochs/config unchanged")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.checkpoint is None or args.expected_checkpoint_sha256 is None:
            parser.error("prepare requires --checkpoint and --expected-checkpoint-sha256")
        result = prepare(args.output, args.checkpoint, args.config, args.expected_checkpoint_sha256)
    elif args.action == "train-arm":
        if args.arm is None:
            parser.error("train-arm requires --arm")
        result = train_arm(args.output, args.arm, args.device, args.checkpoint, args.resume, args.max_steps, args.wall_budget_seconds)
    else:
        result = select(args.output, args.checkpoint)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
