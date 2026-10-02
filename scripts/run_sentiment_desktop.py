"""Rebuild the three fixed Part 2 recipes locally, then freeze/evaluate their weights.

This is a fresh replication of the previously chosen architectures, not a new
hyperparameter search. Training processes receive train/validation tensors only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lab1 import sentiment as s
from lab1.common import utc_now, write_json
import finalize_sentiment_selection as finalizer

DATA = "task2_sentiment/srinidhi/data_processed/full"
FEATURES = "task2_sentiment/srinidhi/data_processed/full_encoded"
RUNTIME_FILES = ("src/lab1/run.py", "src/lab1/common.py", "src/lab1/__init__.py",
                 "task2_sentiment/srinidhi/src/sentiment.py", "task2_sentiment/srinidhi/config.json",
                 "scripts/tune_sentiment.py")
RECIPES = {
    "maxpool_mlp": {"epochs": 6, "early_stopping_patience": 2},
    "bilstm": {"epochs": 12, "early_stopping_patience": 4},
    "dilated_cnn": {"epochs": 6, "early_stopping_patience": 2},
}
RULE = ("Architectures and schedules were fixed before desktop training from the "
        "historical Part 2 selection: original MLP and CNN, original BiLSTM with "
        "12 epochs/patience 4. For each family use its best validation macro-F1 "
        "checkpoint, with the configured minimum improvement and early stopping. "
        "No fresh test predictions or test metrics are used to select checkpoints.")


def base_config(root=ROOT):
    cfg = finalizer.read_json(Path(root) / "task2_sentiment/srinidhi/config.json")["full"]
    return dict(cfg, data_dir=DATA, encoded_cache=FEATURES)


def fixed_recipe(name):
    return {"name": f"desktop_{name}", "model": name,
            "hypothesis": "Replicate the already selected family recipe on the desktop.",
            "overrides": {**RECIPES[name], "num_workers": 0,
                          "checkpoint_every_steps": 1000, "log_every_steps": 200}}


def runtime_manifest(root):
    return {relative: finalizer.digest(Path(root) / relative) for relative in RUNTIME_FILES}


def freeze_plan(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Desktop evidence must remain inside the repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh desktop training output directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"format_version": 1, "created_utc": utc_now(), "selection_rule": RULE,
            "base_config": base_config(root),
            "base_config_sha256": finalizer.digest(root / "task2_sentiment/srinidhi/config.json"),
            "training_source_sha256": runtime_manifest(root),
            "historical_test_metrics_already_observed": True,
            "test_policy": "No new test access during training; official test rows stay fixed.",
            "recipes": {}}
    for name in RECIPES:
        recipe = fixed_recipe(name)
        path = output / "recipes" / f"{name}.json"
        write_json(path, recipe)
        plan["recipes"][name] = {"path": path.relative_to(root).as_posix(),
                                 "sha256": finalizer.digest(path), "recipe": recipe}
    write_json(output / "desktop_plan.json", plan)
    return plan


def verify_plan(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Desktop evidence must remain inside the repository")
    plan = finalizer.read_json(output / "desktop_plan.json")
    if (plan.get("format_version") != 1 or plan.get("selection_rule") != RULE
            or set(plan.get("recipes", {})) != set(s.MODEL_NAMES)):
        raise ValueError("Desktop plan changed or is incomplete")
    if (plan.get("base_config") != base_config(root)
            or plan.get("base_config_sha256") != finalizer.digest(root / "task2_sentiment/srinidhi/config.json")):
        raise ValueError("Frozen desktop base configuration changed")
    if plan.get("training_source_sha256") != runtime_manifest(root):
        raise ValueError("Frozen desktop training source changed")
    for name, row in plan["recipes"].items():
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(output) or finalizer.digest(path) != row["sha256"]:
            raise ValueError("Frozen desktop recipe changed")
        recipe = finalizer.read_json(path)
        if recipe != row["recipe"] or recipe != fixed_recipe(name):
            raise ValueError("Frozen desktop recipe identity mismatch")
    return plan


def freeze_selection(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = verify_plan(output, root)
    path = output / "selection_manifest.json"
    if path.exists():
        raise FileExistsError("Selection is immutable; use the existing file or a fresh run")
    selected = {}
    for name in s.MODEL_NAMES:
        run = output / "training" / name
        provenance = finalizer.read_json(run / "provenance.json")
        metadata = finalizer.read_json(run / name / "validation_selection.json")
        cfg = finalizer.read_json(run / "config.json")
        recipe = plan["recipes"][name]
        expected = {**base_config(root), **recipe["recipe"]["overrides"], "validation_only": True}
        if (provenance.get("status") != "completed" or metadata.get("test_evaluated") is not False
                or s._resume_contract(cfg) != s._resume_contract(expected)
                or provenance.get("candidate") != recipe["recipe"]
                or provenance.get("training_source_sha256") != plan["training_source_sha256"]):
            raise ValueError(f"Unfinished or changed desktop training: {name}")
        checkpoint = run / name / "checkpoints/best.pt"
        row = {"candidate_id": f"desktop_{name}", "model_name": name,
               "source_run": run.relative_to(root).as_posix(),
               "checkpoint": checkpoint.relative_to(root).as_posix(),
               "sha256": finalizer.digest(checkpoint), "config": cfg,
               "validation_macro_f1": metadata["selected_validation_macro_f1"],
               "selected_epoch": metadata["selected_epoch"],
               "parameter_count": metadata["parameter_count"],
               "train_seconds": metadata["train_seconds"], "provenance": provenance,
               "recipe_path": recipe["path"], "recipe_sha256": recipe["sha256"]}
        row["source_metadata_sha256"] = finalizer.verify_source(
            root, name, row, base_config(root))["metadata_sha256"]
        selected[name] = row
    result = {"format_version": 1, "created_utc": utc_now(), "selection_rule": RULE,
              "desktop_plan_sha256": finalizer.digest(output / "desktop_plan.json"),
              "candidate_count": 3, "required_completed_invocations": 3,
              "candidates": list(selected.values()), "selected": selected,
              "test_metrics_used_for_selection": False,
              "historical_test_metrics_already_observed": True,
              "hyperparameter_search_performed": False}
    write_json(path, result)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in {"plan", "all"}:
        if args.stage == "all" and (output / "desktop_plan.json").exists():
            verify_plan(output)
        else:
            freeze_plan(output)
    if args.stage in {"train", "all"}:
        plan = verify_plan(output)
        for name in s.MODEL_NAMES:
            plan = verify_plan(output)  # Reject source/config edits between invocations.
            run = output / "training" / name
            provenance = run / "provenance.json"
            if provenance.exists() and finalizer.read_json(provenance).get("status") == "completed":
                print(f"Keeping completed desktop invocation: {name}", flush=True)
                continue
            command = [sys.executable, str(ROOT / "scripts/tune_sentiment.py"),
                       "--candidate", str(ROOT / plan["recipes"][name]["path"]),
                       "--output", str(run), "--device", args.device]
            last = run / name / "checkpoints/last.pt"
            if last.exists():
                command += ["--resume", str(last)]
            subprocess.run(command, cwd=ROOT, check=True)
    if args.stage in {"evaluate", "all"}:
        selection = output / "selection_manifest.json"
        if not selection.exists():
            selection = freeze_selection(output)
        else:
            verify_plan(output)
            frozen = finalizer.read_json(selection)
            if frozen.get("desktop_plan_sha256") != finalizer.digest(output / "desktop_plan.json"):
                raise ValueError("Frozen selection no longer matches the desktop plan")
        result = finalizer.finalize(selection, output / "selected", device=args.device)
        print(json.dumps({"output": str(output / "selected"),
                          "accuracy": {name: row["accuracy"] for name, row in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
