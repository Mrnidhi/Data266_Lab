"""Verify a frozen validation selection, then evaluate its checkpoints without training.

source_run is repository-relative. checkpoint can be repository-relative or relative
to source_run. Each selected provenance object must match the source provenance JSON.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import sentiment as s
from lab1.common import Tee, code_manifest, environment, resolve_device, utc_now, write_json
import numpy as np
import torch


TUNABLE_KEYS = {
    "embedding_dim", "mlp_hidden", "lstm_hidden", "cnn_channels", "dilations",
    "dropout", "cnn_dropout", "learning_rates", "weight_decay", "gradient_clip",
    "batch_size", "epochs", "early_stopping_patience", "early_stopping_min_delta",
    "amp", "model_seed", "validation_only", "data_dir", "encoded_cache",
    "raw_data_cache", "offline", "num_workers", "cpu_threads",
    "checkpoint_every_steps", "log_every_steps",
}
ORCHESTRATION_SCRIPTS = (
    "select_sentiment_candidates.py", "finalize_sentiment_selection.py", "tune_sentiment.py",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluation_code_manifest(root):
    """Record current evaluation tooling without rewriting training provenance."""
    hashes = code_manifest(root)
    script_dir = Path(__file__).resolve().parent
    hashes.update({f"scripts/{name}": digest(script_dir / name) for name in ORCHESTRATION_SCRIPTS})
    return hashes


def fixed_contract(cfg):
    return {key: value for key, value in cfg.items() if key not in TUNABLE_KEYS}


def source_paths(root, selected):
    run_part, checkpoint_part = Path(selected["source_run"]), Path(selected["checkpoint"])
    if run_part.is_absolute() or checkpoint_part.is_absolute():
        raise ValueError("Selection source_run and checkpoint must be relative paths")
    run = (root / run_part).resolve()
    if not run.is_relative_to(root):
        raise ValueError("Source run escapes the repository")
    choices = {(root / checkpoint_part).resolve(), (run / checkpoint_part).resolve()}
    matches = [path for path in choices if path.is_file() and path.is_relative_to(run)]
    if len(matches) != 1:
        raise ValueError("Checkpoint must resolve unambiguously inside its source run")
    checkpoint = matches[0]
    return run, checkpoint


def verify_source(root, name, selected, base):
    """Validate source metadata and load both states, without any prediction calls."""
    required = {"source_run", "checkpoint", "sha256", "config", "validation_macro_f1",
                "selected_epoch", "provenance"}
    if not required <= selected.keys():
        raise ValueError(f"Incomplete selected metadata for {name}")
    run, checkpoint = source_paths(root, selected)
    if checkpoint != (run / name / "checkpoints/best.pt").resolve():
        raise ValueError(f"Selected checkpoint must be the source best.pt for {name}")
    if digest(checkpoint) != selected["sha256"]:
        raise ValueError(f"Selected checkpoint SHA256 mismatch: {name}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if state["model_name"] != name or cfg != selected["config"]:
        raise ValueError(f"Selected checkpoint model/config mismatch: {name}")
    if fixed_contract(cfg) != fixed_contract(base):
        raise ValueError(f"Frozen preprocessing/split/evaluation configuration changed: {name}")
    source_cfg = read_json(run / "config.json")
    if s._resume_contract(source_cfg) != s._resume_contract(cfg):
        raise ValueError(f"Source config disagrees with checkpoint: {name}")
    provenance_path = run / ("provenance.json" if cfg.get("validation_only") else "run_summary.json")
    provenance = read_json(provenance_path)
    if provenance != selected["provenance"] or provenance.get("status") != "completed":
        raise ValueError(f"Source provenance mismatch or incomplete run: {name}")
    if s._resume_contract(provenance.get("config", {})) != s._resume_contract(cfg):
        raise ValueError(f"Provenance config mismatch: {name}")
    if not isinstance(provenance.get("environment"), dict):
        raise ValueError(f"Missing source hardware environment: {name}")
    history_path = run / name / "history.json"
    history = read_json(history_path)
    last_path = checkpoint.with_name("last.pt")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if (last["model_name"] != name or last["vocabulary"] != state["vocabulary"]
            or last["fingerprint"] != state["fingerprint"]
            or s._resume_contract(last["config"]) != s._resume_contract(cfg)
            or last["history"] != history or last.get("epoch_progress")):
        raise ValueError(f"Source last checkpoint/history mismatch: {name}")
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError(f"Incomplete training history: {name}")
    epoch, f1 = selected["selected_epoch"], selected["validation_macro_f1"]
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= len(history)
            or not isinstance(f1, (int, float)) or not math.isfinite(f1)
            or state["epoch"] != epoch or state["best_f1"] != f1
            or history[epoch - 1]["validation_macro_f1"] != f1
            or state["history"] != history[:epoch] or last["epoch"] != len(history)):
        raise ValueError(f"Selected validation score/epoch disagrees with source: {name}")
    best_f1, best_epoch = -1.0, None
    for row in history:
        if row["validation_macro_f1"] > best_f1 + cfg["early_stopping_min_delta"]:
            best_f1, best_epoch = row["validation_macro_f1"], row["epoch"]
    if best_epoch != epoch or last["best_f1"] != f1:
        raise ValueError(f"Selection is not the source validation-best checkpoint: {name}")
    metadata_path = run / name / ("validation_selection.json" if cfg.get("validation_only") else "metrics.json")
    metadata = read_json(metadata_path)
    if (metadata["selected_epoch"] != epoch or metadata["selected_validation_macro_f1"] != f1
            or metadata["train_seconds"] != sum(row["train_seconds"] for row in history)
            or metadata["completed_training_steps"] != last.get("global_step")):
        raise ValueError(f"Source training metadata mismatch: {name}")
    if cfg.get("validation_only") and (metadata.get("test_evaluated") is not False
                                      or provenance.get("result") != metadata):
        raise ValueError(f"Invalid validation-only source metadata: {name}")
    if not cfg.get("validation_only") and provenance.get("summary", {}).get("models", {}).get(name) != metadata:
        raise ValueError(f"Source suite provenance disagrees with model metadata: {name}")
    model = s.build_model(name, len(state["vocabulary"]), cfg)
    model.load_state_dict(state["model"], strict=True)
    del model
    metadata_sha256 = {
        path.relative_to(root).as_posix(): digest(path)
        for path in (checkpoint, last_path, history_path, metadata_path,
                     run / "config.json", provenance_path)
    }
    # The selector first verifies candidates before this map exists. Once frozen,
    # require the same complete mapping, including both checkpoints and raw metadata.
    if "source_metadata_sha256" in selected and selected["source_metadata_sha256"] != metadata_sha256:
        raise ValueError(f"Frozen source metadata SHA256 mapping mismatch: {name}")
    return {"state": state, "history": history, "metadata": metadata,
            "checkpoint": checkpoint, "last": last_path, "run": run,
            "provenance": provenance, "metadata_sha256": metadata_sha256}


def cpu_evidence(root, sources):
    path = root / "verification/part_b_cpu_hardware.txt"
    if path.is_file():
        raw = path.read_text()
        fields = dict(line.split(":", 1) for line in raw.splitlines() if ":" in line)
        model = fields.get("Model name", "").strip()
        if model:
            return {"model": model, "source": path.relative_to(root).as_posix(),
                    "sha256": digest(path), "lscpu": raw,
                    "scope": "Recorded Part B training host"}
    for source in sources.values():
        env = source["provenance"]["environment"]
        value = env.get("cpu") or env.get("cpu_model") or env.get("cpu_hardware")
        if value:
            return {"details": value, "source": "selected source provenance",
                    "scope": "Recorded source training host"}
    raise ValueError("Exact CPU hardware evidence is required")


def finalize(selection_path, output, *, device="cpu", root=ROOT, base_config=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final evaluation requires a fresh empty output directory")
    selection_bytes = Path(selection_path).read_bytes()
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    selection = json.loads(selection_bytes)
    if (not isinstance(selection.get("selection_rule"), str) or not selection["selection_rule"].strip()
            or not isinstance(selection.get("candidates"), list)
            or set(selection.get("selected", {})) != set(s.MODEL_NAMES)):
        raise ValueError("Expected a frozen selection rule, candidate list, and all three models")
    base = dict(base_config) if base_config is not None else read_json(root / "task2_sentiment/srinidhi/config.json")["full"]
    if base_config is None:
        base.update(data_dir=str(root / "task2_sentiment/srinidhi/data_processed/full"),
                    encoded_cache=str(root / "task2_sentiment/srinidhi/data_processed/full_encoded"))
    torch.set_num_threads(base.get("cpu_threads", 2))
    sources = {name: verify_source(root, name, selection["selected"][name], base) for name in s.MODEL_NAMES}
    hardware = cpu_evidence(root, sources)
    records = s._load_records(base)
    audit = s.audit_splits(records)
    vocabulary, datasets = s._load_features(base, records)
    test_ids = [row["id"] for row in records["test"]]
    expected_y = np.asarray([row["label"] for row in records["test"]])
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate test IDs prevent paired evaluation")
    if base["mode"] == "full" and audit["split_counts"] != {"train": 504000, "validation": 56000, "test": 38000}:
        raise ValueError("Final full evaluation must retain every frozen official row")
    for name, source in sources.items():
        state = source["state"]
        if (state["vocabulary"] != vocabulary or state["fingerprint"] != s._fingerprint(
                state["config"], vocabulary, records, legacy=state.get("format_version", 1) == 1)):
            raise ValueError(f"Frozen checkpoint data/vocabulary fingerprint mismatch: {name}")
        for relative, checksum in source["metadata_sha256"].items():
            if digest(root / relative) != checksum:
                raise ValueError(f"Source changed during verification: {relative}")
    # Every checkpoint, source metadata item, split and cache is verified before this point.
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_manifest.json").write_bytes(selection_bytes)
    model_configs = {name: source["state"]["config"] for name, source in sources.items()}
    assembled_config = {**base, "artifact_type": "assembled_validation_selection", "model_configs": model_configs}
    write_json(output / "config.json", assembled_config)
    write_json(output / "resolved_config.json", assembled_config)
    write_json(output / "vocabulary.json", vocabulary)
    write_json(output / "data_audit.json", audit)
    write_json(output / "preprocessing.json", {"description": "HTML/whitespace normalization, lowercase, contraction expansion, punctuation removal, customized stopword removal preserving negation; vocabulary fitted only on training rows.",
        "stopwords": sorted(s.STOPWORDS), "max_length": base["max_length"], "padding_id": s.PAD, "unknown_id": s.UNK,
        "signature": s._preprocessing_signature()})
    write_json(output / "dataset_statistics.json", {name: {"count": len(data),
        "truncated_fraction": float(np.mean(np.asarray(data.lengths) > base["max_length"])),
        "mean_oov_rate": float(np.mean(data.oov_rates)), "mean_tokens_before_truncation": float(np.mean(data.lengths))}
        for name, data in datasets.items()})
    np.savez_compressed(output / "split_ids.npz", **{name: np.asarray([row["id"] for row in rows]) for name, rows in records.items()})
    for name, source in sources.items():
        folder = output / name
        (folder / "checkpoints").mkdir(parents=True)
        for key, filename in (("checkpoint", "best.pt"), ("last", "last.pt")):
            target = folder / "checkpoints" / filename
            shutil.copy2(source[key], target)
            if digest(target) != source["metadata_sha256"][source[key].relative_to(root).as_posix()]:
                raise ValueError(f"Checkpoint changed while freezing output: {name}/{filename}")
        write_json(folder / "history.json", source["history"])
        write_json(folder / "config.json", model_configs[name])
        write_json(folder / "source_selection.json", {**selection["selected"][name],
            "resolved_checkpoint": source["checkpoint"].relative_to(root).as_posix(),
            "source_metadata_sha256": source["metadata_sha256"]})
    training_environment = dict(next(iter(sources.values()))["provenance"]["environment"])
    training_environment["cpu"] = hardware
    evaluation_started_utc = utc_now()
    provenance = {"task": "sentiment", "mode": base["mode"], "status": "running",
        "artifact_type": "assembled_validation_selection", "assembled_results": True,
        "description": "Derived evaluation of separately selected original checkpoints; no training in this invocation.",
        "selection_rule": selection["selection_rule"], "selection_manifest_sha256": selection_sha,
        "selection_frozen_utc": selection.get("created_utc"),
        "evaluation_started_utc": evaluation_started_utc, "source_runs": selection["selected"],
        "config": assembled_config, "environment": training_environment,
        "environment_scope": "Representative source training environment; per-model originals are in source_runs.",
        "evaluation_environment": environment(), "device": device,
        "source_sha256": evaluation_code_manifest(root),
        "source_sha256_scope": "Current project sources and selector/finalizer/tuning scripts; original training evidence remains in source_runs.",
        "started_utc": evaluation_started_utc}
    write_json(output / "run_summary.json", provenance)
    started = time.perf_counter()
    results, predictions = {}, {}
    with (output / "RUN_LOG.txt").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                for name, source in sources.items():
                    print(f"Evaluating frozen {name} checkpoint on {len(test_ids)} paired test rows", flush=True)
                    training = {key: source["metadata"][key] for key in
                                ("train_seconds", "peak_cuda_memory_bytes", "completed_training_steps")}
                    metrics, y, probability = s.evaluate_checkpoint(source["state"], datasets["test"], output / name,
                        device, history=source["history"], training_metadata=training)
                    with (output / name / "test_predictions.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    if ([row["example_id"] for row in rows] != test_ids
                            or [int(row["true_label"]) for row in rows] != expected_y.tolist()
                            or not np.array_equal(y, expected_y) or metrics["count"] != len(test_ids)):
                        raise ValueError(f"Test IDs/labels are not paired correctly for {name}")
                    results[name], predictions[name] = metrics, probability
                comparisons = {name: s.paired_mcnemar(expected_y, predictions["maxpool_mlp"], predictions[name])
                               for name in s.MODEL_NAMES[1:]}
                write_json(output / "paired_mcnemar.json", comparisons)
                summary = {"mode": base["mode"], "synthetic": base["mode"] == "smoke", "models": results,
                    "paired_mcnemar": comparisons, "manual_error_review_complete": False,
                    "artifact_type": "assembled_validation_selection", "selection_manifest_sha256": selection_sha,
                    "fingerprint": s._fingerprint(base, vocabulary, records)}
                write_json(output / "summary.json", summary)
                provenance.update(status="completed", summary=summary)
            except BaseException as error:
                provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                provenance.update(ended_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
                write_json(output / "run_summary.json", provenance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = finalize(args.selection, args.output, device=resolve_device(args.device))
    print(json.dumps({"output": str(args.output), "models": list(summary["models"]),
                      "artifact_type": summary["artifact_type"]}, indent=2))


if __name__ == "__main__":
    main()
