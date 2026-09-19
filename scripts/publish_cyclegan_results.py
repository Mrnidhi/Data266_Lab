"""Validate and stage a completed CycleGAN run; default to a read-only dry run.

The destination must be new. This utility never installs artifacts over a member's
existing notebook, review, or results. Its executed notebook reads saved evidence;
it does not pretend that publication retrained the model.

After a rollback, pass --history-run-dir OLD --history-cutoff OLD=STEP, where
STEP is the saved update from which training resumed. Derived curves exclude that
old run's unsaved tail; its complete original log is still copied unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


DIRECTIONS = ("photo_to_monet", "monet_to_photo")
NETWORKS = ("G_photo_to_monet", "G_monet_to_photo", "D_photo", "D_monet")
METRICS = ("fid", "kid", "precision", "recall", "density", "coverage", "cycle_l1", "lpips_cycle", "content_cosine")
LOSS_NAMES = ("generator_total", "gan_photo_to_monet", "gan_monet_to_photo", "discriminator_photo",
              "discriminator_monet", "cycle_photo_l1", "cycle_monet_l1", "identity_photo_l1", "identity_monet_l1")
COLUMNS = ("scope", "direction", "metric", "value", "status", "details", "checkpoint_sha256", "data_manifest_sha256")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Non-finite or nonnumeric {label}")
    if positive and value <= 0:
        raise ValueError(f"Expected positive {label}")
    return value


def validate_manifest(manifest):
    if manifest.get("kind") != "explicit_class_manifests" or manifest.get("class_results") is not True:
        raise ValueError("Synthetic or unverified data cannot be published")
    entries = manifest["entries"]
    expected = {f"{split}_{domain}" for split in ("train", "val", "test") for domain in ("photo", "monet")}
    if set(entries) != expected or set(manifest["counts"]) != expected:
        raise ValueError("Six complete split manifests are required")
    seen = set()
    for name, records in entries.items():
        if not records or len(records) != manifest["counts"][name]:
            raise ValueError(f"Invalid split count: {name}")
        for record in records:
            path = Path(record["relative_path"])
            sha = record["sha256"]
            if path.is_absolute() or ".." in path.parts or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Unsafe split path or invalid data hash")
            key = (name.rsplit("_", 1)[1], sha)
            if key in seen:
                raise ValueError("Duplicate image hash within/across domain splits")
            seen.add(key)
    actual = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    if manifest.get("manifest_fingerprint") != actual:
        raise ValueError("Data manifest fingerprint does not match its entries")


def load_evidence(run_dir, evaluation_dir, checkpoint, commands_file, history_run_dirs=(), history_cutoffs=None):
    """Fail closed before writing; checkpoint loading is for this project's trusted files only."""
    run, evaluation_dir, checkpoint, commands_file = map(Path, (run_dir, evaluation_dir, checkpoint, commands_file))
    run, evaluation_dir, checkpoint, commands_file = (p.resolve() for p in (run, evaluation_dir, checkpoint, commands_file))
    provenance = read_json(run / "run_summary.json")
    config = read_json(run / "resolved_config.json")
    manifest = read_json(run / "data_manifest.json")
    evaluation = read_json(evaluation_dir / "metrics.json")
    summary = provenance.get("summary", {})
    if not (provenance.get("status") == "completed" and provenance.get("task") == "cyclegan"
            and provenance.get("mode") == config.get("mode") == summary.get("mode") == "full"
            and summary.get("class_results") is True and summary.get("data_kind") == "explicit_class_manifests"):
        raise ValueError("Only a completed full run on real class data can be published")
    if provenance.get("config") != config or not provenance.get("source_sha256") or not provenance.get("environment"):
        raise ValueError("Missing or inconsistent configuration, code hashes, or training environment")
    validate_manifest(manifest)
    fingerprint = manifest["manifest_fingerprint"]
    steps = max(manifest["counts"]["train_photo"], manifest["counts"]["train_monet"])
    epochs = config.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("A positive full epoch schedule is required")
    expected = steps * epochs
    if (summary.get("steps_per_epoch") != steps or summary.get("completed_updates") != expected
            or summary.get("completed_epoch_fraction") != epochs
            or (config.get("max_steps") is not None and config["max_steps"] < expected)):
        raise ValueError("Incomplete training: the full configured epoch schedule must finish")
    if evaluation.get("class_results") is not True or evaluation.get("data_kind") != manifest["kind"]:
        raise ValueError("Evaluation is synthetic or is not verified class data")
    if evaluation.get("split") != "test" or evaluation.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Held-out test evaluation must use the training run's frozen manifests")
    if set(evaluation.get("directions", {})) != set(DIRECTIONS):
        raise ValueError("Both translation directions must be evaluated")
    for direction, entry in evaluation["directions"].items():
        source, target = direction.split("_to_")
        if (entry.get("source_count") != manifest["counts"][f"test_{source}"]
                or entry.get("generated_count") != entry.get("source_count")
                or entry.get("real_reference_count") != manifest["counts"][f"test_{target}"]):
            raise ValueError("Incomplete evaluation: all held-out source/reference images are required")
        if not set(METRICS).issubset(entry.get("metrics", {})):
            raise ValueError("Evaluation metric fields are missing")
        for name, metric in entry["metrics"].items():
            if metric.get("status") == "computed":
                finite(metric.get("value"), f"{direction}/{name}")
            elif metric.get("status") != "unavailable" or metric.get("value") is not None or not metric.get("reason"):
                raise ValueError("Unavailable metrics require a null value and an explicit reason")
        for folder, count in (("source", entry["source_count"]), ("generated", entry["generated_count"]),
                              ("cycle", entry["source_count"]), ("real_target", entry["real_reference_count"])):
            if len(list((evaluation_dir / direction / folder).glob("*.png"))) != count:
                raise ValueError(f"Missing image evidence: {direction}/{folder}")
    checkpoint_hash = digest(checkpoint)
    if evaluation.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Evaluation checkpoint hash mismatch")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("state", {})
    if (saved.get("format_version") != 1 or set(saved.get("models", {})) != set(NETWORKS)
            or state.get("manifest_fingerprint") != fingerprint
            or state.get("global_step") != evaluation.get("checkpoint_step")
            or not 0 < state.get("global_step", 0) <= expected):
        raise ValueError("Checkpoint identity, data, or evaluation step is inconsistent")
    for key in ("mode", "seed", "image_size", "base_channels", "residual_blocks", "batch_size",
                "learning_rate", "betas", "cycle_weight", "identity_weight", "epochs", "constant_epochs", "replay_size"):
        if key not in config or saved["config"].get(key) != config[key]:
            raise ValueError(f"Checkpoint recipe differs from completed run: {key}")
    # Match the trainer's legacy checkpoint rule: absent precision means FP32.
    if saved["config"].get("precision", "fp32") != config.get("precision", "fp32"):
        raise ValueError("Checkpoint recipe differs from completed run: precision")
    if config.get("select_best", True):
        selection = summary.get("best_selection") or {}
        finite(selection.get("value"), "validation selection value")
        if (selection.get("metric") != "mean_validation_KID_both_directions" or selection.get("validation_only") is not True
                or selection.get("step") != state["global_step"] or digest(run / "best.pt") != checkpoint_hash
                or state.get("best_selection") != selection or state.get("best_validation_score") != selection["value"]):
            raise ValueError("Evaluation must reference the run's validation-selected best checkpoint")
    elif state["global_step"] != expected:
        raise ValueError("Without validation selection, evaluate the completed final checkpoint")
    del saved
    if set(summary.get("parameter_counts", {})) != set(NETWORKS):
        raise ValueError("All four network parameter counts are required")
    for network, count in summary["parameter_counts"].items():
        finite(count, f"{network} parameter count", positive=True)
    finite(summary.get("training_seconds"), "training time", positive=True)
    finite(summary.get("training_source_images_per_second"), "training throughput", positive=True)

    histories = list(dict.fromkeys([*(Path(p).resolve() for p in history_run_dirs), run]))
    cutoffs = {Path(path).resolve(): cutoff for path, cutoff in (history_cutoffs or {}).items()}
    if any(path not in histories or path == run for path in cutoffs):
        raise ValueError("History cutoffs must refer to explicit earlier --history-run-dir paths")
    if any(isinstance(cutoff, bool) or not isinstance(cutoff, int) or not 0 <= cutoff < expected for cutoff in cutoffs.values()):
        raise ValueError("History cutoffs must be saved update integers before the completed final update")
    if cutoffs and provenance.get("resumed") is not True:
        raise ValueError("Rollback cutoffs require a final run explicitly recorded as resumed")
    records = {}
    reconstruction = []
    for index, source_run in enumerate(histories):
        prior_manifest = read_json(source_run / "data_manifest.json")
        if prior_manifest.get("manifest_fingerprint") != fingerprint:
            raise ValueError("Resumed history belongs to different data")
        prior_config = read_json(source_run / "resolved_config.json")
        for key in ("mode", "seed", "image_size", "base_channels", "residual_blocks", "batch_size",
                    "learning_rate", "betas", "cycle_weight", "identity_weight", "epochs", "constant_epochs", "replay_size", "precision"):
            default = "fp32" if key == "precision" else None
            if prior_config.get(key, default) != config.get(key, default):
                raise ValueError(f"Resumed history belongs to a different training recipe: {key}")
        if not (source_run / "RUN_LOG.txt").is_file():
            raise ValueError("The unedited training console log is required")
        discarded = 0
        cutoff = cutoffs.get(source_run)
        for line in (source_run / "training_log.jsonl").read_text().splitlines():
            record = json.loads(line)
            step = record["step"]
            if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= expected:
                raise ValueError("Invalid training log step")
            if cutoff is not None and step > cutoff:
                discarded += 1
                continue
            if step in records and records[step] != record:
                raise ValueError("Conflicting resumed training records at the same step")
            if not set(LOSS_NAMES).issubset(record.get("losses", {})) or set(record.get("gradient_norms", {})) != set(NETWORKS):
                raise ValueError("Loss or gradient evidence is incomplete")
            for group in ("losses", "gradient_norms"):
                for name, value in record[group].items():
                    finite(value, f"step {step} {group}/{name}")
            finite(record.get("step_seconds"), f"step {step} time", positive=True)
            finite(record.get("nan_events"), f"step {step} NaN count")
            records[step] = record
        reconstruction.append({"raw_log": f"outputs/full/raw_logs/run_{index:02d}/training_log.jsonl",
                               "explicit_saved_update_cutoff": cutoff, "excluded_unsaved_tail_records": discarded,
                               "raw_log_sha256": digest(source_run / "training_log.jsonl")})
    if sorted(records) != list(range(1, expected + 1)):
        raise ValueError("Incomplete training history; include all preceding --history-run-dir values")
    logs = [records[step] for step in sorted(records)]
    if logs[-1]["nan_events"] != summary.get("nan_events"):
        raise ValueError("Summary NaN count differs from the final training record")
    measured_seconds = math.fsum(record["step_seconds"] for record in logs)
    if not math.isclose(measured_seconds, summary["training_seconds"], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("Summary training time differs from the reconstructed per-update timing")
    measured_throughput = 2 * expected / measured_seconds
    if not math.isclose(measured_throughput, summary["training_source_images_per_second"], rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Summary throughput differs from reconstructed source-image count and training time")
    commands = read_json(commands_file)
    if (not isinstance(commands.get("training"), list) or not commands["training"]
            or not all(isinstance(item, str) and item.strip() for item in commands["training"])
            or not isinstance(commands.get("evaluation"), str) or not commands["evaluation"].strip()):
        raise ValueError("Commands JSON requires training (list of exact commands) and evaluation (exact command)")
    return {"run": run, "evaluation_dir": evaluation_dir, "checkpoint": checkpoint,
            "commands_file": commands_file, "histories": histories, "provenance": provenance,
            "history_reconstruction": reconstruction,
            "config": config, "data": manifest, "evaluation": evaluation, "summary": summary, "logs": logs,
            "commands": commands, "checkpoint_sha256": checkpoint_hash,
            "data_manifest_sha256": digest(run / "data_manifest.json")}


def metric_rows(evidence):
    rows = []

    def add(scope, direction, metric, value, status="computed", **details):
        rows.append(dict(zip(COLUMNS, (scope, direction, metric, value if value is not None else "", status,
            json.dumps(details, sort_keys=True, allow_nan=False), evidence["checkpoint_sha256"], evidence["data_manifest_sha256"]))))

    for direction, entry in evidence["evaluation"]["directions"].items():
        for name, metric in entry["metrics"].items():
            add("test", direction, name, metric["value"], metric["status"],
                **{k: v for k, v in metric.items() if k not in ("value", "status")})
        for key in ("source_count", "generated_count", "real_reference_count", "inference_seconds", "inference_images_per_second"):
            add("test", direction, key, entry.get(key), "computed" if entry.get(key) is not None else "unavailable")
    summary = evidence["summary"]
    for key in ("completed_updates", "completed_epoch_fraction", "training_seconds", "session_wall_seconds",
                "training_source_images_per_second", "peak_gpu_memory_bytes", "nan_events"):
        detail = {"definition": summary.get("throughput_definition", "")} if "images_per_second" in key else {}
        if key == "session_wall_seconds":
            detail["definition"] = "Final invocation wall time; not total across earlier resumed invocations"
        if key == "peak_gpu_memory_bytes":
            detail["definition"] = summary.get("memory_note", "Peak allocated training memory")
        add("training", "both", key, summary.get(key), "computed" if summary.get(key) is not None else "unavailable", **detail)
    for name, count in summary["parameter_counts"].items():
        add("model", name, "parameter_count", count)
    add("model", "both", "parameter_count_total", sum(summary["parameter_counts"].values()))
    for group in ("losses", "gradient_norms"):
        for name in evidence["logs"][0][group]:
            values = [record[group][name] for record in evidence["logs"]]
            for statistic, value in (("mean", sum(values) / len(values)), ("final_update", values[-1])):
                add("training", name, f"{group}_{statistic}", value, count=len(values),
                    definition="Raw per-update values; final update need not be the validation-selected checkpoint")
    environment = evidence["provenance"]["environment"]
    for name, value in (("gpu", "; ".join(gpu["name"] for gpu in environment.get("gpus", []))),
                        ("cpu", environment.get("cpu_model")), ("platform", environment.get("platform")),
                        ("python", environment.get("python")), ("cuda_build", environment.get("cuda_build"))):
        add("hardware", "both", name, value or None, "recorded" if value else "unavailable",
            source="Training provenance; never the publisher's current machine")
    for direction in DIRECTIONS:
        for criterion in ("style", "content", "artifacts"):
            for statistic in ("mean_score", "percent_agreement", "quadratic_weighted_kappa"):
                add("human_audit", direction, f"{criterion}_{statistic}", None, "pending",
                    reason="Requires two independent human raters on the fixed blinded sample set")
    for name in ("public_score", "private_score", "leaderboard_rank", "class_fid", "class_mifid"):
        add("kaggle", "competition", name, None, "pending", reason="Verified class evaluator/submission evidence has not been supplied")
    return rows


def write_curves(records, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True)
    # Plot bounded point counts while keeping every original record byte unchanged.
    stride = max(1, math.ceil(len(records) / 5000))
    sampled = records[::stride]
    if sampled[-1] is not records[-1]:
        sampled.append(records[-1])
    x = [r["step"] for r in sampled]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    groups = (("Adversarial losses", ["gan_photo_to_monet", "gan_monet_to_photo", "discriminator_photo", "discriminator_monet"]),
              ("Raw cycle and identity L1 losses", ["cycle_photo_l1", "cycle_monet_l1", "identity_photo_l1", "identity_monet_l1"]),
              ("Weighted generator objective", ["generator_total"]))
    for ax, (title, names) in zip(axes, groups):
        for name in names:
            ax.plot(x, [r["losses"][name] for r in sampled], label=name, linewidth=.8)
        ax.set(title=title, ylabel="Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Completed update")
    fig.suptitle(f"Actual training records; every {stride} update(s) shown; final update included")
    fig.tight_layout()
    fig.savefig(destination / "loss_curves.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in NETWORKS:
        ax.plot(x, [r["gradient_norms"][name] for r in sampled], label=name, linewidth=.8)
    ax.set(xlabel="Completed update", ylabel="Gradient L2 norm", title=f"Gradient norms; every {stride} update(s) shown")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / "gradient_curves.png", dpi=140)
    plt.close(fig)


def build_notebook():
    import nbformat
    cells = [nbformat.v4.new_markdown_cell(
        "# CycleGAN — actual saved run evidence\n\nSrinidhi, Lab Pair 49. This notebook **summarizes a saved completed training run "
        "and held-out evaluation**. Its executed cells verify and read actual artifacts; they do not train the models. "
        "The exact recorded training/evaluation commands and resolved configuration are displayed below. "
        "Human ratings and Kaggle results remain pending; this is not a claim that every lab requirement is complete."),
        nbformat.v4.new_code_cell('''from pathlib import Path
import csv
import hashlib
import json
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "publication.json").is_file())
manifest = json.loads((ROOT / "publication.json").read_text())
for relative, info in manifest["files"].items():
    sha = hashlib.sha256()
    with (ROOT / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    assert sha.hexdigest() == info["sha256"], relative
print("Verified", len(manifest["files"]), "saved evidence files.")
OUT = ROOT / "outputs/full"
provenance = json.loads((OUT / "run_summary.json").read_text())
assert provenance["status"] == "completed" and provenance["summary"]["class_results"]
print("Training environment (not this notebook's execution machine)")
display(provenance["environment"])
print("Resolved training configuration")
display(json.loads((OUT / "resolved_config.json").read_text()))
data = json.loads((OUT / "data_manifest.json").read_text())
display({"split_counts": data["counts"], "manifest_fingerprint": data["manifest_fingerprint"],
         "checkpoint_sha256": manifest["checkpoint_sha256"]})
commands = json.loads((OUT / "commands.json").read_text())
print("Exact recorded training commands:")
print("\\n".join(commands["training"]))
print("Exact recorded evaluation command:")
print(commands["evaluation"])'''),
        nbformat.v4.new_code_cell('''metrics = pd.read_csv(ROOT / "metrics_report.csv", keep_default_na=False)
with pd.option_context("display.max_rows", None, "display.max_colwidth", 100):
    display(metrics[["scope", "direction", "metric", "value", "status", "details"]])
print("Pending entries are not measurements and are not zero.")'''),
        nbformat.v4.new_code_cell('''for name in ("loss_curves.png", "gradient_curves.png"):
    display(Image(filename=str(OUT / "figures" / name), width=950))'''),
        nbformat.v4.new_markdown_cell(
            "## Fixed inspection examples\n\nThe first three saved examples in each direction are shown deterministically. "
            "They are inspection examples, not a hand-picked competition submission. Cycle images reconstruct the original source; "
            "unpaired real targets are not image-level ground truth."),
        nbformat.v4.new_code_cell('''for direction in ("photo_to_monet", "monet_to_photo"):
    folder = OUT / "evaluation" / direction
    print(direction)
    for generated in sorted((folder / "generated").glob("*.png"))[:3]:
        print("Saved sample", generated.stem, "— source, translation, reconstruction")
        for label in ("source", "generated", "cycle"):
            display(Image(filename=str(folder / label / generated.name), width=256))'''),
        nbformat.v4.new_markdown_cell(
            "## Preserved raw console logs\n\nThe complete byte-identical logs and source hashes are retained. "
            "Only the final 4,000 characters of each console log are displayed here."),
        nbformat.v4.new_code_cell('''for log in sorted((OUT / "raw_logs").glob("run_*/RUN_LOG.txt")):
    print(log.relative_to(ROOT))
    print(log.read_text()[-4000:])
print("Human audit: two genuine independent raters are still required.")
print("Kaggle: class evaluation and submission evidence are still required.")''')]
    return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def results_text(evidence, rows):
    cfg, summary = evidence["config"], evidence["summary"]
    lines = ["# Srinidhi — CycleGAN results", "",
        "Status: actual full training and held-out evaluation are published. Human ratings, Kaggle results, student interpretation, and team comparison remain pending.", "",
        f"The model uses two {cfg['residual_blocks']}-block generators and two PatchGAN discriminators, {cfg['base_channels']} base channels, "
        f"{cfg['image_size']}px images, batch size {cfg['batch_size']}, seed {cfg['seed']}, Adam learning rate {cfg['learning_rate']} and betas {cfg['betas']}. "
        f"The configured {cfg['epochs']} epochs completed ({summary['completed_updates']} updates).", "",
        f"Cycle weight is {cfg['cycle_weight']} and the absolute identity weight is {cfg['identity_weight']}. "
        "These describe the recorded recipe; they do not by themselves establish that it is optimal.", "",
        f"Selected checkpoint SHA-256: `{evidence['checkpoint_sha256']}`. "
        f"Frozen data fingerprint: `{evidence['data']['manifest_fingerprint']}`.", "",
        "| Direction | Metric | Value | Status |", "|---|---|---:|---|"]
    for row in rows:
        if row["scope"] == "test" and row["metric"] in METRICS:
            value = f"{row['value']:.6g}" if isinstance(row["value"], (int, float)) else "—"
            lines.append(f"| {row['direction']} | {row['metric']} | {value} | {row['status']} |")
    lines += ["", "Both CSV reports contain all recorded image-quality, training, parameter, hardware, and pending human/Kaggle fields. "
        "Unavailable values remain blank with a reason; pending values are not zero.", "",
        "Local FID/KID use the recorded held-out evaluator; they are not verified class FID/MiFID or leaderboard scores. "
        "LPIPS compares the source with its own cycle reconstruction. Content cosine compares the source and its translation. "
        "Small Monet reference sets limit distributional metric precision.", "",
        "The source training environment is in `outputs/full/run_summary.json`; the publisher's hardware is never substituted. "
        "Training throughput counts two source images per paired-domain update and excludes data loading/evaluation. "
        "Final-invocation wall time does not include earlier resumed sessions. Per-network losses and gradient norms retain their logged semantics.", "",
        "If training resumed after a rollback, `outputs/full/history_reconstruction.json` records each explicit saved-update cutoff. "
        "Only derived curves and loss summaries exclude the discarded unsaved tail; original raw logs remain byte-identical.", "",
        "Evidence: `outputs/full/raw_logs/`, `outputs/full/evaluation/metrics.json`, `outputs/full/figures/`, "
        "`outputs/full/resolved_config.json`, `outputs/full/data_manifest.json`, `checkpoints/selected.pt`, and `publication.json`. "
        "The executed `src/cyclegan.ipynb` reads these saved artifacts; its outputs do not imply notebook retraining.", "",
        "## Exact recorded commands", "", "```sh", *evidence["commands"]["training"], evidence["commands"]["evaluation"], "```", "",
        "## Analysis still requiring review", "",
        "Inspect fixed source/translation/cycle examples for style changes, content loss, and artifacts. Explain observed convergence and stability "
        "using the actual curves, justify model choices in your own words, and compare with the teammate on the same evaluation protocol. "
        "Have two independent people complete the blinded audit; do not replace these ratings with automatically generated opinions.", ""]
    return "\n".join(lines)


def publish(evidence, output_dir, *, write=False):
    """Build only a fresh staging directory. No --write means no filesystem writes."""
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Publication destination must not exist; member artifacts are never overwritten")
    rows = metric_rows(evidence)
    result = {"write": write, "destination": str(output), "metric_rows": len(rows),
              "checkpoint_sha256": evidence["checkpoint_sha256"], "data_manifest_sha256": evidence["data_manifest_sha256"],
              "completed_updates": evidence["summary"]["completed_updates"], "human_audit_status": "pending",
              "kaggle_status": "pending", "all_lab_requirements_complete": False}
    if not write:
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cyclegan-publication-", dir=output.parent))
    try:
        out = stage / "outputs/full"
        out.mkdir(parents=True)
        sources = {}

        def copy(source, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            original = digest(source)
            shutil.copy2(source, target)
            if digest(target) != original:
                raise ValueError("Evidence changed while copying")
            sources[target.relative_to(stage).as_posix()] = {"sha256": original, "bytes": target.stat().st_size}

        for name in ("run_summary.json", "resolved_config.json", "data_manifest.json"):
            copy(evidence["run"] / name, out / name)
        copy(evidence["commands_file"], out / "commands.json")
        write_json(out / "history_reconstruction.json", evidence["history_reconstruction"])
        copy(evidence["checkpoint"], stage / "checkpoints/selected.pt")
        if digest(stage / "checkpoints/selected.pt") != evidence["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after validation")
        for index, run in enumerate(evidence["histories"]):
            for name in ("RUN_LOG.txt", "training_log.jsonl", "run_summary.json", "resolved_config.json", "data_manifest.json"):
                if (run / name).is_file():
                    copy(run / name, out / "raw_logs" / f"run_{index:02d}" / name)
            for path in sorted((run / "provenance").glob("*.json")):
                copy(path, out / "raw_logs" / f"run_{index:02d}" / "provenance" / path.name)
        for path in sorted(evidence["evaluation_dir"].rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".png", ".txt", ".log"}:
                copy(path, out / "evaluation" / path.relative_to(evidence["evaluation_dir"]))
        for name in ("metrics_report.csv", "full_metrics_report.csv"):
            with (stage / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        write_curves(evidence["logs"], out / "figures")
        (stage / "results.md").write_text(results_text(evidence, rows))
        write_json(stage / "checkpoints/manifest.json", {"selected.pt": {
            "sha256": evidence["checkpoint_sha256"], "step": evidence["evaluation"]["checkpoint_step"],
            "manifest_fingerprint": evidence["data"]["manifest_fingerprint"], "selection": evidence["summary"].get("best_selection")}})
        files = {path.relative_to(stage).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
                 for path in sorted(stage.rglob("*")) if path.is_file()}
        write_json(stage / "publication.json", {"format_version": 1, "member": "srinidhi", "task": "cyclegan",
            "artifact_type": "saved_run_publication", "complete_lab_submission": False,
            "checkpoint_sha256": evidence["checkpoint_sha256"], "data_manifest_sha256": evidence["data_manifest_sha256"],
            "manifest_fingerprint": evidence["data"]["manifest_fingerprint"], "files": files,
            "byte_identical_source_copies": sources,
            "notebook_note": "Executed by reading saved evidence; no training or evaluation is performed by the presentation cells"})
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import AsyncKernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        notebook = build_notebook()
        # Use this environment, not a stale user-level "python3" kernelspec.
        with tempfile.TemporaryDirectory(prefix="cyclegan-publisher-kernel-") as kernel_root:
            write_json(Path(kernel_root) / "python3/kernel.json", {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "CycleGAN evidence verification", "language": "python"})
            manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[kernel_root], ensure_native_kernel=False))
            NotebookClient(notebook, km=manager, timeout=180, kernel_name="python3",
                           resources={"metadata": {"path": str(stage)}}).execute()
        for cell in notebook.cells:
            if cell.cell_type == "code" and (cell.execution_count is None or any(o.output_type == "error" for o in cell.outputs)):
                raise ValueError("Evidence notebook did not execute completely")
        (stage / "src").mkdir()
        nbformat.write(notebook, stage / "src/cyclegan.ipynb")
        write_json(stage / "notebook_execution.json", {"status": "completed", "execution_kind": "saved_artifact_presentation",
            "code_cells": sum(c.cell_type == "code" for c in notebook.cells), "errors": 0,
            "notebook_sha256": digest(stage / "src/cyclegan.ipynb")})
        if output.exists() or output.is_symlink():
            raise FileExistsError("Destination appeared during publication; refusing to overwrite it")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="This project's trusted evaluated checkpoint")
    parser.add_argument("--commands-file", type=Path, required=True, help='JSON: {"training": ["exact command"], "evaluation": "exact command"}')
    parser.add_argument("--history-run-dir", type=Path, action="append", default=[], help="Earlier resumed run; repeat to preserve a complete history")
    parser.add_argument("--history-cutoff", action="append", default=[], metavar="OLD_RUN_DIR=SAVED_STEP",
                        help="Explicitly exclude an earlier run's unsaved tail from derived curves only; original logs remain unchanged")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh staging directory; existing member artifacts are never replaced")
    parser.add_argument("--write", action="store_true", help="Write and execute the evidence notebook; otherwise validate only")
    args = parser.parse_args()
    cutoffs = {}
    for value in args.history_cutoff:
        path, separator, step = value.rpartition("=")
        try:
            if not separator or not path or Path(path).resolve() in cutoffs:
                raise ValueError
            cutoffs[Path(path).resolve()] = int(step)
        except ValueError:
            parser.error("Each --history-cutoff must be a unique OLD_RUN_DIR=SAVED_STEP")
    evidence = load_evidence(args.run_dir, args.evaluation_dir, args.checkpoint, args.commands_file, args.history_run_dir, cutoffs)
    print(json.dumps(publish(evidence, args.output_dir, write=args.write), indent=2))


if __name__ == "__main__":
    main()
