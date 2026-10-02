"""Run a bounded, preregistered sentiment search using validation scores only.

Existing desktop checkpoints remain eligible controls. This command never calls
test prediction, evaluation or publication code.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lab1 import sentiment as s
from lab1.common import utc_now, write_json
import finalize_sentiment_selection as finalizer
from run_sentiment_desktop import base_config, runtime_manifest

RECIPES = ("mlp_regularized", "mlp_wide_regularized", "cnn_regularized", "cnn_wide_regularized")
RECIPE_DIR = "task2_sentiment/srinidhi/experiments/desktop-quality-20261001"
BASELINE = "reproducibility/raw_logs/srinidhi/desktop-20261001/part2-full/selection_manifest.json"
DEFAULT_OUTPUT = "reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search"
MARGIN = Decimal("0.001")
RULE = (
    "For each family, consider its existing desktop control and every completed new "
    "candidate. Find the highest selected-checkpoint validation macro-F1. Among "
    "candidates within 0.001 inclusive of that maximum, prefer fewer parameters, "
    "then higher validation macro-F1, then source path and candidate ID. A wider "
    "candidate must therefore exceed a smaller control by more than 0.001 to "
    "displace it. Checkpoints follow configured early_stopping_min_delta=0.0001. "
    "No test score or test prediction enters selection. This practical tolerance "
    "does not establish statistical significance."
)


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect_source(row):
    verified = finalizer.verify_source(ROOT, row["model_name"], row, base_config(ROOT))
    result = verified["metadata_sha256"]
    del verified
    gc.collect()
    return result


def freeze_plan(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Plan requires a fresh output folder")
    baseline = finalizer.read_json(ROOT / BASELINE)
    controls = []
    for name in s.MODEL_NAMES:
        row = dict(baseline["selected"][name])
        row["source_metadata_sha256"] = inspect_source(row)
        row["role"] = "existing_desktop_control"
        controls.append(row)
    recipes = {}
    for identity in RECIPES:
        relative = f"{RECIPE_DIR}/{identity}.json"
        recipe = finalizer.read_json(ROOT / relative)
        if recipe["name"] != identity or recipe["model"] not in s.MODEL_NAMES:
            raise ValueError("Unexpected candidate identity")
        recipes[identity] = {"path": relative, "sha256": finalizer.digest(ROOT / relative), "recipe": recipe}
    plan = {
        "format_version": 1, "created_utc": utc_now(), "selection_rule": RULE,
        "practical_margin": float(MARGIN), "base_config": base_config(ROOT),
        "baseline_selection_path": BASELINE,
        "baseline_selection_sha256": finalizer.digest(ROOT / BASELINE),
        "training_source_sha256": runtime_manifest(ROOT),
        "orchestration_sha256": {"scripts/run_sentiment_quality.py": finalizer.digest(Path(__file__))},
        "recipes": recipes, "controls": controls,
        "candidate_count": 7, "new_training_invocations": 4,
        "historical_test_metrics_already_observed": True,
        "test_policy": "Training receives only train/validation tensors. Existing tuner verifies frozen test cache identity but computes no test predictions or metrics. No fresh test evaluation until the final selection is frozen.",
        "research": [
            {"url": "https://aclanthology.org/P17-1052.pdf", "use": "Residual word CNN and stronger head-dropout motivation; architecture differs and its 2.64 percent Yelp error includes unsupervised embeddings, so that number is not a target or directly comparable scratch result."},
            {"url": "https://arxiv.org/abs/1706.04599", "use": "Calibration is separate from classification accuracy; no temperature fitting is included in these four training experiments."},
        ],
        "runtime_policy": "Candidates run serially. Other explicitly coordinated GPU training may overlap; actual invocation notes and GPU snapshots record that throughput is not an isolated benchmark.",
    }
    exclusive_json(output / "quality_plan.json", plan)
    exclusive_json(output / "baseline_validation_receipt.json", {
        "created_utc": utc_now(), "plan_sha256": finalizer.digest(output / "quality_plan.json"),
        "test_inference_performed": False,
        "controls": [{key: row[key] for key in ("candidate_id", "model_name", "validation_macro_f1", "selected_epoch", "parameter_count", "sha256", "source_metadata_sha256")} for row in controls],
    })
    return plan


def verify_plan(output):
    plan = finalizer.read_json(output / "quality_plan.json")
    if plan.get("selection_rule") != RULE or set(plan.get("recipes", {})) != set(RECIPES):
        raise ValueError("Frozen plan identity/rule mismatch")
    if plan["base_config"] != base_config(ROOT) or plan["training_source_sha256"] != runtime_manifest(ROOT):
        raise ValueError("Frozen runtime or base configuration changed")
    if plan["baseline_selection_sha256"] != finalizer.digest(ROOT / plan["baseline_selection_path"]):
        raise ValueError("Frozen control selection changed")
    for relative, checksum in plan["orchestration_sha256"].items():
        if finalizer.digest(ROOT / relative) != checksum:
            raise ValueError("Frozen orchestration changed")
    for recipe in plan["recipes"].values():
        if finalizer.digest(ROOT / recipe["path"]) != recipe["sha256"] or finalizer.read_json(ROOT / recipe["path"]) != recipe["recipe"]:
            raise ValueError("Frozen recipe changed")
    for control in plan["controls"]:
        for relative, checksum in control["source_metadata_sha256"].items():
            if finalizer.digest(ROOT / relative) != checksum:
                raise ValueError("Frozen control evidence changed")
    return plan


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv"], capture_output=True, text=True, check=False)
    return {"utc": utc_now(), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def candidate_row(output, plan, identity):
    recipe = plan["recipes"][identity]
    name = recipe["recipe"]["model"]
    run = output / "training" / identity
    provenance = finalizer.read_json(run / "provenance.json")
    cfg = finalizer.read_json(run / "config.json")
    metadata = finalizer.read_json(run / name / "validation_selection.json")
    expected = {**plan["base_config"], **recipe["recipe"]["overrides"], "validation_only": True}
    if (provenance.get("status") != "completed" or metadata.get("test_evaluated") is not False
            or s._resume_contract(cfg) != s._resume_contract(expected)
            or provenance.get("candidate") != recipe["recipe"]
            or provenance.get("training_source_sha256") != plan["training_source_sha256"]):
        raise ValueError(f"Candidate is unfinished or differs from plan: {identity}")
    checkpoint = run / name / "checkpoints/best.pt"
    row = {"candidate_id": identity, "model_name": name,
           "source_run": run.relative_to(ROOT).as_posix(), "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
           "sha256": finalizer.digest(checkpoint), "config": cfg,
           "validation_macro_f1": metadata["selected_validation_macro_f1"],
           "selected_epoch": metadata["selected_epoch"], "parameter_count": metadata["parameter_count"],
           "train_seconds": metadata["train_seconds"], "provenance": provenance,
           "recipe_path": recipe["path"], "recipe_sha256": recipe["sha256"], "role": "new_validation_candidate"}
    row["source_metadata_sha256"] = inspect_source(row)
    return row


def train(output, overlap_note):
    plan = verify_plan(output)
    for identity in RECIPES:
        run = output / "training" / identity
        recipe = plan["recipes"][identity]
        name = recipe["recipe"]["model"]
        provenance_path = run / "provenance.json"
        if provenance_path.exists() and finalizer.read_json(provenance_path).get("status") == "completed":
            candidate_row(output, plan, identity)
            print(f"Already completed and verified: {identity}", flush=True)
            continue
        command = [sys.executable, "scripts/tune_sentiment.py", "--candidate", recipe["path"], "--output", run.relative_to(ROOT).as_posix(), "--device", "cuda"]
        last = run / name / "checkpoints/last.pt"
        if last.exists():
            command += ["--resume", last.relative_to(ROOT).as_posix()]
        invocation = {"candidate_id": identity, "started_utc": utc_now(), "command": command,
                      "overlap_note": overlap_note, "timing_is_isolated_benchmark": False, "gpu_start": gpu_snapshot()}
        stamp = invocation["started_utc"].replace(":", "-")
        receipt = output / "invocations" / f"{identity}_{stamp}.json"
        write_json(receipt, invocation)
        print(f"Starting validation-only candidate {identity}", flush=True)
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        invocation.update(ended_utc=utc_now(), returncode=result.returncode, gpu_end=gpu_snapshot())
        write_json(receipt, invocation)
        if result.returncode:
            raise RuntimeError(f"Candidate failed: {identity}; inspect its preserved logs")
        row = candidate_row(output, plan, identity)
        print(json.dumps({key: row[key] for key in ("candidate_id", "validation_macro_f1", "selected_epoch", "train_seconds")}), flush=True)


def select(output):
    plan = verify_plan(output)
    path = output / "selection_manifest.json"
    if path.exists():
        raise FileExistsError("Selection is immutable")
    candidates = [dict(row) for row in plan["controls"]]
    candidates.extend(candidate_row(output, plan, identity) for identity in RECIPES)
    selected, rationale = {}, {}
    for name in s.MODEL_NAMES:
        family = [row for row in candidates if row["model_name"] == name]
        score = lambda row: Decimal(str(row["validation_macro_f1"]))
        highest = max(score(row) for row in family)
        eligible = [row for row in family if highest - score(row) <= MARGIN]
        chosen = min(eligible, key=lambda row: (row["parameter_count"], -score(row), row["source_run"], row["candidate_id"]))
        for row in family:
            row["decision"] = {"selected": row is chosen, "within_practical_band": highest - score(row) <= MARGIN,
                               "gap_to_highest_validation_macro_f1": float(highest - score(row))}
        selected[name] = chosen
        rationale[name] = {"highest_validation_macro_f1": float(highest), "eligible_candidate_ids": [row["candidate_id"] for row in eligible], "selected_candidate_id": chosen["candidate_id"]}
    for row in candidates:
        inspect_source(row)
    result = {"format_version": 1, "created_utc": utc_now(), "selection_rule": RULE,
              "quality_plan_sha256": finalizer.digest(output / "quality_plan.json"), "practical_margin": float(MARGIN),
              "candidate_count": len(candidates), "required_completed_invocations": 4,
              "candidates": candidates, "selected": selected, "rationale": rationale,
              "test_metrics_used_for_selection": False, "historical_test_metrics_already_observed": True,
              "test_evaluation_started": False}
    exclusive_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--stage", required=True, choices=("plan", "verify", "train", "select"))
    parser.add_argument("--overlap-note", help="Required training disclosure of coordinated concurrent GPU workloads")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Evidence output must stay inside the repository")
    if args.stage == "plan":
        result = freeze_plan(output)
    elif args.stage == "verify":
        result = verify_plan(output)
    elif args.stage == "train":
        if not args.overlap_note:
            parser.error("--overlap-note is required for training")
        train(output, args.overlap_note)
        result = {"training_completed": True, "test_inference_performed": False}
    else:
        result = select(output)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "stage": args.stage,
                      "candidate_count": result.get("candidate_count"), "test_inference_performed": False}, indent=2))


if __name__ == "__main__":
    main()
