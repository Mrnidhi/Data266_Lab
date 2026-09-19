"""Publish completed Yelp results while preserving the original run evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("maxpool_mlp", "bilstm", "dilated_cnn")
TIMING_CONTEXT = (
    "These source runs overlapped other Part B jobs on one RTX 5090. Their measured training times and throughput "
    "describe that shared session and are not controlled, isolated architecture-speed comparisons. "
    "The initial benchmark is separate evidence in verification/sentiment_5090_benchmark.json."
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def training_hardware(provenance, cpu_evidence):
    """Read training hardware, never substitute the publisher's local hardware."""
    environment = provenance.get("environment", {})
    cpu = environment.get("cpu_model") or provenance.get("cpu_model")
    cpu_source = "training provenance" if cpu else "not recorded"
    if not cpu and cpu_evidence.is_file():
        for line in cpu_evidence.read_text().splitlines():
            if line.strip().startswith("Model name:"):
                cpu = line.split(":", 1)[1].strip()
                cpu_source = cpu_evidence.relative_to(ROOT).as_posix()
                break
    gpus = environment.get("gpus", [])
    gpu = "; ".join(item["name"] for item in gpus) or "No GPU recorded"
    return {"cpu": cpu or "not recorded", "cpu_source": cpu_source, "gpu": gpu}


def model_sources(run, cfg, provenance, selection):
    """Keep training provenance distinct from a derived suite's evaluation run."""
    if selection is not None and set(selection["selected"]) != set(NAMES):
        raise ValueError("Selection manifest must contain exactly the three model families")
    sources = {}
    for name in NAMES:
        source = dict(selection["selected"][name]) if selection else {
            "source_run": run.relative_to(ROOT).as_posix(), "config": cfg,
            "provenance": provenance,
            "checkpoint": (run / name / "checkpoints/best.pt").relative_to(ROOT).as_posix(),
        }
        source_run = (ROOT / source["source_run"]).resolve()
        if not source_run.is_relative_to(ROOT.resolve()):
            raise ValueError("Training source must remain within the repository")
        checkpoint = (ROOT / source["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(source_run):
            checkpoint = (source_run / source["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(source_run):
            raise ValueError("Checkpoint must remain within its source run")
        source["checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
        source["hardware"] = training_hardware(source["provenance"], ROOT / "verification/part_b_cpu_hardware.txt")
        sources[name] = source
    return sources


def copy_raw_logs(run, outputs, sources):
    """Copy exact log bytes with a portable manifest; leave source runs untouched."""
    folder = outputs / "raw_logs"
    folder.mkdir(exist_ok=True)
    training_runs = {}
    for name, source in sources.items():
        training_runs.setdefault(source["source_run"], []).append(name)
    entries = [("training", source_run, sorted(names)) for source_run, names in sorted(training_runs.items())]
    entries.append(("evaluation", run.relative_to(ROOT).as_posix(), []))
    logs = []
    for role, source_run, models in entries:
        source = ROOT / source_run / "RUN_LOG.txt"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_id = hashlib.sha256(source_run.encode()).hexdigest()[:12]
        label = "_".join(models) if models else "suite"
        target = folder / f"{role}_{label}_{source_id}_{digest[:12]}.txt"
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Raw log copy mismatch: {source_run}")
        logs.append({"role": role, "models": models, "source_run": source_run,
                     "source_path": source.relative_to(ROOT).as_posix(),
                     "published_path": target.relative_to(outputs).as_posix(),
                     "sha256": digest, "bytes": target.stat().st_size})
    manifest = {"format_version": 1, "path_base": "outputs/full", "logs": logs}
    write_json(folder / "manifest.json", manifest)
    return manifest


def write_error_review(path, selected, checkpoint_sha256):
    """Retain annotations for unchanged predictions; archive annotated prior sets."""
    selected = selected.copy()
    selected["checkpoint_sha256"] = checkpoint_sha256
    if path.is_file():
        prior = pd.read_csv(path, keep_default_na=False)
        annotation_columns = [column for column in prior if column not in selected or
                              column in {"error_type", "testable_fix", "student_reviewed"}]
        annotated = any(str(value).strip().lower() not in {"", "false", "0", "nan"}
                        for column in annotation_columns for value in prior[column])
        if annotated:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archive = path.parent / "review_history" / f"{path.stem}.{digest[:16]}.csv"
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                shutil.copy2(path, archive)
        if prior.example_id.duplicated().any():
            raise ValueError("Existing error-review file has duplicate example IDs")
        prior = prior.set_index("example_id")
        for index, row in selected.iterrows():
            if row.example_id not in prior.index:
                continue
            old = prior.loc[row.example_id]
            same_prediction = (int(old.true_label) == int(row.true_label)
                               and int(old.predicted_label) == int(row.predicted_label)
                               and abs(float(old.positive_probability) - float(row.positive_probability)) < 1e-12)
            same_checkpoint = not old.get("checkpoint_sha256") or old["checkpoint_sha256"] == checkpoint_sha256
            if same_prediction and same_checkpoint:
                for column in annotation_columns:
                    if column not in selected:
                        selected[column] = ""
                    if column == "student_reviewed":
                        selected.at[index, column] = str(old[column]).strip().lower() in {"true", "1"}
                    elif str(old[column]).strip():
                        selected.at[index, column] = old[column]
    selected.to_csv(path, index=False)


def reproduction_text(run, outputs, sources, selection):
    """Publish commands against preserved candidate recipes and frozen configs."""
    folder = outputs / "reproduction"
    folder.mkdir(exist_ok=True)
    commands, seen = [], set()
    for name, source in sources.items():
        if source["source_run"] in seen:
            continue
        seen.add(source["source_run"])
        candidate = ROOT / source["source_run"] / "candidate.json"
        if candidate.is_file():
            target = folder / f"{name}_candidate.json"
            shutil.copy2(candidate, target)
            commands.append(shlex.join([".venv/bin/python", "scripts/tune_sentiment.py", "--candidate",
                target.relative_to(ROOT).as_posix(), "--output", f"runs/reproduce-part2-{name}", "--device", "cuda"]))
        else:
            target = folder / f"{name}_full_config.json"
            write_json(target, {"full": source["config"]})
            commands.append(shlex.join([".venv/bin/python", "-m", "lab1.run", "--task", "sentiment", "--mode", "full",
                "--config", target.relative_to(ROOT).as_posix(), "--device", "cuda", "--output", f"runs/reproduce-part2-{name}"]))
    text = ("## Reproduction commands\n\nRun from the repository root after preparing the documented environment and frozen full-data cache. "
            "Each destination below must be new. These commands retrain the selected recipes; preserved source manifests record the original code and environment.\n\n"
            "```bash\n" + "\n".join(commands) + "\n```\n\n")
    if selection:
        command = shlex.join([".venv/bin/python", "scripts/finalize_sentiment_selection.py", "--selection",
            (run / "selection_manifest.json").relative_to(ROOT).as_posix(), "--output", "runs/reproduce-part2-final-evaluation"])
        text += ("To repeat final evaluation of the **preserved selected checkpoints**, use:\n\n```bash\n" + command + "\n```\n\n"
                 "After retraining, create a new selection manifest from the new validation results, paths and checkpoint hashes before finalizing those new runs. "
                 "The command above intentionally references the original frozen selection; it does not select the newly trained runs.\n\n")
    return text + "See results.md for metric definitions and limitations. Student error review and teammate comparisons are separate deliverables."


def select_errors(predictions, records):
    data = predictions.copy()
    data["text"] = data.example_id.map({r["id"]: r["text"] for r in records})
    errors = data[data.true_label != data.predicted_label].copy()
    selected, used = [], set()

    def take(group, candidates):
        chosen = candidates[~candidates.example_id.isin(used)].head(5).copy()
        if len(chosen) != 5:
            raise ValueError(f"Fewer than five distinct errors for {group}")
        chosen["review_group"] = group
        selected.append(chosen)
        used.update(chosen.example_id)

    take("confident_false_positive", errors[errors.true_label == 0].sort_values(["positive_probability", "example_id"], ascending=[False, True]))
    take("confident_false_negative", errors[errors.true_label == 1].sort_values(["positive_probability", "example_id"]))
    errors["distance_from_threshold"] = abs(errors.positive_probability - 0.5)
    take("near_threshold", errors.sort_values(["distance_from_threshold", "example_id"]))
    # A prespecified long-review slice; select the longest incorrectly predicted reviews.
    take("long_review_slice", errors[errors.original_token_length >= 200].sort_values(["original_token_length", "example_id"], ascending=[False, True]))
    result = pd.concat(selected, ignore_index=True)
    result["error_type"] = ""
    result["testable_fix"] = ""
    result["student_reviewed"] = False
    return result.drop(columns=["distance_from_threshold"], errors="ignore")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    relative = run.relative_to(ROOT).as_posix()
    summary = json.loads((run / "summary.json").read_text())
    provenance = json.loads((run / "run_summary.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    audit = json.loads((run / "data_audit.json").read_text())
    selection_path = run / "selection_manifest.json"
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else None
    sources = model_sources(run, cfg, provenance, selection)
    if not (provenance["status"] == "completed" and summary["mode"] == "full"
            and not summary["synthetic"] and set(summary["models"]) == set(NAMES)
            and audit["split_counts"] == {"train": 504000, "validation": 56000, "test": 38000}):
        raise ValueError("Only a completed full official-data run can be published")
    member = ROOT / "task2_sentiment/srinidhi"
    outputs = member / "outputs/full"
    if not selection and (outputs / "selection_manifest.json").exists():
        raise ValueError("Published selection metadata exists; use the matching selected suite")
    outputs.mkdir(parents=True, exist_ok=True)
    for file in ("summary.json", "config.json", "vocabulary.json", "preprocessing.json", "data_audit.json", "dataset_statistics.json", "split_ids.npz", "run_summary.json", "paired_mcnemar.json"):
        shutil.copy2(run / file, outputs / file)
    if selection:
        shutil.copy2(selection_path, outputs / "selection_manifest.json")
    write_json(outputs / "training_sources.json", sources)
    copy_raw_logs(run, outputs, sources)
    test_records = [json.loads(line) for line in (args.data_dir / "test.jsonl").read_text().splitlines()]
    rows, checkpoints, comparison = [], {}, []
    for name in NAMES:
        source_info = sources[name]
        model_config = source_info["config"]
        hardware = source_info["hardware"]
        model_dir = outputs / name
        model_dir.mkdir(exist_ok=True)
        for file in (run / name).iterdir():
            if file.is_file() and file.name != "required_20_errors_for_review.csv":
                shutil.copy2(file, model_dir / file.name)
        write_json(model_dir / "training_config.json", model_config)
        write_json(model_dir / "training_provenance.json", source_info["provenance"])
        metrics = summary["models"][name]
        history = json.loads((run / name / "history.json").read_text())
        if metrics["count"] != 38000:
            raise ValueError("Incomplete test evaluation")
        checkpoints[name] = {}
        for filename in ("best.pt", "last.pt"):
            source = run / name / "checkpoints" / filename
            target = member / "checkpoints" / name / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError("Checkpoint copy mismatch")
            if filename == "best.pt" and selection and digest != source_info["sha256"]:
                raise ValueError(f"Selected checkpoint hash mismatch for {name}")
            checkpoints[name][filename] = {"path": target.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": target.stat().st_size}
        if selection and (metrics["selected_epoch"] != source_info["selected_epoch"] or
                          abs(metrics["selected_validation_macro_f1"] - source_info["validation_macro_f1"]) > 1e-12):
            raise ValueError(f"Selected validation metadata mismatch for {name}")

        def add(metric, value, split="test"):
            rows.append({"model": name, "split": split, "metric": metric,
                         "value": json.dumps(value) if isinstance(value, (dict, list)) else value,
                         "status": "measured_full_test_selected_model" if selection else "measured_full_run",
                         "run_id": source_info["source_run"], "evaluation_run_id": relative,
                         "checkpoint": checkpoints[name]["best.pt"]["path"],
                         "source_checkpoint": source_info["checkpoint"],
                         "training_config": (model_dir / "training_config.json").relative_to(ROOT).as_posix(),
                         "training_cpu": hardware["cpu"], "training_cpu_source": hardware["cpu_source"],
                         "training_gpu": hardware["gpu"]})

        for key in ("accuracy", "confusion_matrix", "roc_auc", "pr_auc_trapezoidal", "mcc", "brier", "ece_15", "test_loss"):
            add(key, metrics[key])
        for average in ("macro", "micro", "weighted"):
            for metric, value in metrics[average].items():
                add(f"{average}_{metric}", value)
        for metric, bounds in metrics["bootstrap_95_percentile"].items():
            add(f"{metric}_95ci_low", bounds["low"])
            add(f"{metric}_95ci_high", bounds["high"])
        for key in ("parameter_count", "train_seconds", "peak_cuda_memory_bytes", "selected_epoch", "selected_validation_macro_f1", "completed_training_steps"):
            add(key, metrics[key], "training")
        add("examples_per_second", 504000 * len(history) / metrics["train_seconds"], "training")
        for slice_name, values in json.loads((run / name / "slices.json").read_text()).items():
            add("support", values["count"], slice_name)
            if values["count"]:
                add("macro_f1", values["macro"]["f1"], slice_name)
                add("error_rate", 1 - values["accuracy"], slice_name)
        if name != "maxpool_mlp":
            for key in ("table", "discordant_pairs", "pvalue_exact_two_sided"):
                add(f"paired_mcnemar_vs_baseline_{key}", summary["paired_mcnemar"][name][key])
        predictions = pd.read_csv(run / name / "test_predictions.csv")
        selected = select_errors(predictions, test_records)
        write_error_review(model_dir / "required_20_errors_for_review.csv", selected, checkpoints[name]["best.pt"]["sha256"])
        comparison.append({"model": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro"]["f1"],
                           "roc_auc": metrics["roc_auc"], "mcc": metrics["mcc"], "selected_epoch": metrics["selected_epoch"],
                           "epochs_run": len(history), "training_minutes": metrics["train_seconds"] / 60,
                           "source_run": source_info["source_run"], "embedding_dim": model_config["embedding_dim"],
                           "max_length": model_config["max_length"], "batch_size": model_config["batch_size"],
                           "initial_learning_rate": model_config["learning_rates"][name], "dropout": model_config["dropout"],
                           "training_cpu": hardware["cpu"], "training_gpu": hardware["gpu"]})
    pd.DataFrame(rows).to_csv(member / "metrics_report.csv", index=False)
    pd.DataFrame(comparison).to_csv(outputs / "model_comparison.csv", index=False)
    write_json(member / "checkpoints/manifest.json", checkpoints)
    reproduction = reproduction_text(run, outputs, sources, selection)
    source_table = "| Model | Actual training run | CPU | GPU | Configuration |\n| --- | --- | --- | --- | --- |\n"
    for name, source in sources.items():
        hardware = source["hardware"]
        source_table += (f"| {name} | `{source['source_run']}` | {hardware['cpu']} | {hardware['gpu']} | "
                         f"[Exact settings](outputs/full/{name}/training_config.json) |\n")
    selection_text = ("This is a derived evaluation suite assembled from separately preserved training runs. "
                      f"Selection rule: {selection['selection_rule']} "
                      "All candidate records and selected checkpoint hashes are preserved in outputs/full/selection_manifest.json. "
                      "Original run evaluations and logs remain unchanged.\n\n" if selection else
                      "This publishes one preserved full training suite. Within each model, best.pt updates when validation macro-F1 "
                      "improves by more than the configured early_stopping_min_delta; this need not be the strict maximum over all epochs.\n\n")
    (member / "results.md").write_text(
        "# Srinidhi - Yelp sentiment results\n\n"
        "All three models trained independently using 504,000 training reviews and 56,000 validation reviews; "
        "final evaluation covers all 38,000 official test reviews. Training hardware and configurations below come from each actual source run. "
        "CPU evidence sources are recorded in outputs/full/training_sources.json.\n\n"
        + source_table + "\n" + selection_text +
        "The baseline max-pool MLP tests what unordered lexical evidence can achieve. The BiLSTM adds bidirectional sequence context; "
        "the residual dilated CNN learns local patterns over a wider receptive field. Each model learns its own embeddings from scratch. "
        "See each model's training_config.json for its actual widths, learning rates, dropout, batch size and stopping settings. "
        "The source history records the validation trajectory and selected epoch.\n\n"
        "Confidence intervals use the bootstrap sample count in each evaluation metric file and IID test-row resampling; "
        "they do not measure variation across training seeds. "
        "PR-AUC uses trapezoidal integration; ECE uses 15 equal-width top-label confidence bins. McNemar tests compare paired predictions against the baseline, with two unadjusted p-values.\n\n"
        "All required numeric results, including slice support, macro-F1 and error rate, are in metrics_report.csv. "
        "Peak memory is PyTorch's maximum allocated CUDA memory, not the entire device reservation. Training time sums the source run's measured training epochs; "
        "source provenance records total invocation elapsed time, without a separate setup/validation/export breakdown. "
        "Training throughput is processed training examples divided by those measured epoch times. "
        + TIMING_CONTEXT + "\n\n" +
        "Data length and class distributions, class balance, blanks and truncation/OOV statistics are in outputs/full/data_distributions.json and .png.\n\n"
        f"Original evaluation evidence: ../../{relative}. The notebook reads the published copies in outputs/full; "
        "byte-identical training and evaluation logs are mapped to their original paths in outputs/full/raw_logs/manifest.json. "
        "Original training evidence is listed per model above. Checkpoint mapping: checkpoints/manifest.json. "
        "outputs/full/model_comparison.csv summarizes the three models. Each model's required_20_errors_for_review.csv contains five errors from each required category. "
        "Student review is recorded only by explicit student_reviewed values; newly generated interpretation fields remain blank and unreviewed. "
        "Existing annotations are retained for unchanged checkpoint predictions; annotated prior sets are archived under review_history. "
        "Teammate comparisons and the combined final report require the teammate's actual results.\n\n" + reproduction + "\n")

    notebook = build_notebook(reproduction)
    nbformat.write(notebook, member / "src/sentiment.ipynb")
    print(json.dumps({"run": relative, "metric_rows": len(rows), "models": comparison, "manual_review_complete": False}, indent=2))


def build_notebook(reproduction):
    cells = [nbformat.v4.new_markdown_cell("# Yelp sentiment - preserved full-run results\n\nSrinidhi, Lab Pair 49. "
        "This notebook displays preserved training and evaluation results. Its inference cell reloads the selected checkpoints on CPU when executed. "
        "Generating this notebook does not execute its cells. The accompanying sentiment.py contains the complete model and training implementation; "
        "these presentation and inference cells do not retrain the models."),
        nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import torch
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
MEMBER = ROOT / "task2_sentiment/srinidhi"
OUTPUTS = MEMBER / "outputs/full"
summary = json.loads((OUTPUTS / "summary.json").read_text())
provenance = json.loads((OUTPUTS / "run_summary.json").read_text())
assert provenance["status"] == "completed" and summary["mode"] == "full"
print("Evaluation provenance")
display(provenance.get("evaluation_environment", provenance.get("environment", {})))
sources = json.loads((OUTPUTS / "training_sources.json").read_text())
for name, source in sources.items():
    print(name, "training source:", source["source_run"])
    display(source["hardware"])
    display(json.loads((OUTPUTS / name / "training_config.json").read_text()))
selection_path = OUTPUTS / "selection_manifest.json"
if selection_path.is_file():
    selection = json.loads(selection_path.read_text())
    print("Selection rule:", selection["selection_rule"])
    selected_models = [{"model": name, "source_run": source["source_run"],
                        "selected_epoch": source["selected_epoch"],
                        "validation_macro_f1": source["validation_macro_f1"],
                        "checkpoint_sha256": source["sha256"]}
                       for name, source in selection["selected"].items()]
    display(pd.DataFrame(selected_models))
display(json.loads((OUTPUTS / "data_audit.json").read_text()))'''),
        nbformat.v4.new_markdown_cell("## Data analysis\n\nReview lengths are measured after the documented preprocessing and before truncation. "
            "The table records label counts, class balance, blank or empty reviews, length quantiles, truncation and unknown-token rates for each split."),
        nbformat.v4.new_code_cell('''distributions = json.loads((OUTPUTS / "data_distributions.json").read_text())
print(distributions["length_definition"])
display(pd.DataFrame.from_dict(distributions["splits"], orient="index"))
display(Image(filename=str(OUTPUTS / "data_distributions.png"), width=1000))'''),
        nbformat.v4.new_code_cell('display(pd.read_csv(OUTPUTS / "model_comparison.csv"))\ndisplay(pd.read_csv(MEMBER / "metrics_report.csv"))'),
        nbformat.v4.new_markdown_cell("## Timing interpretation\n\n" + TIMING_CONTEXT),
        nbformat.v4.new_code_cell('''for name in ("maxpool_mlp", "bilstm", "dilated_cnn"):
    print(name)
    for file in sorted((OUTPUTS / name).glob("*.png")):
        print(file.name)
        display(Image(filename=str(file), width=800))'''),
        nbformat.v4.new_markdown_cell("## Required error-review examples\n\nFive confident false positives, five confident false negatives, "
            "five near-threshold errors, and five long-review slice failures per model. Error types and testable fixes require student review."),
        nbformat.v4.new_code_cell('''for name in ("maxpool_mlp", "bilstm", "dilated_cnn"):
    print(name)
    display(pd.read_csv(OUTPUTS / name / "required_20_errors_for_review.csv"))'''),
        nbformat.v4.new_markdown_cell("## Checkpoints and fresh CPU inference"),
        nbformat.v4.new_code_cell('''import hashlib
from lab1.sentiment import Reviews, build_model, collate_reviews
torch.set_num_threads(2)
manifest = json.loads((MEMBER / "checkpoints/manifest.json").read_text())
texts = ["The meal was delicious and the staff were friendly.", "The food was cold and the service was terrible."]
for name, files in manifest.items():
    for filename, info in files.items():
        assert hashlib.sha256((ROOT / info["path"]).read_bytes()).hexdigest() == info["sha256"]
    saved = torch.load(ROOT / files["best.pt"]["path"], map_location="cpu", weights_only=False)
    model = build_model(name, len(saved["vocabulary"]), saved["config"]).eval()
    model.load_state_dict(saved["model"], strict=True)
    data = Reviews([{"id": str(i), "text": t, "label": 0} for i,t in enumerate(texts)], saved["vocabulary"], saved["config"]["max_length"])
    ids, _, _ = collate_reviews([data[i] for i in range(len(data))])
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(ids))
    assert torch.isfinite(probabilities).all()
    print(name, "selected epoch", saved["epoch"])
    print(model)
    display(pd.DataFrame({"demo_text": texts, "positive_probability": probabilities.numpy()}))
print("All six checkpoint hashes verified. Demo predictions are separate from test metrics.")'''),
        nbformat.v4.new_markdown_cell("## Raw log excerpts\n\nThe complete unedited log copies are included in outputs/full/raw_logs. "
            "Their manifest preserves each original source path and SHA-256. These cells verify the copied bytes and display only the final 12,000 characters."),
        nbformat.v4.new_code_cell('''import hashlib
log_manifest = json.loads((OUTPUTS / "raw_logs/manifest.json").read_text())
for info in log_manifest["logs"]:
    log = OUTPUTS / info["published_path"]
    assert hashlib.sha256(log.read_bytes()).hexdigest() == info["sha256"]
    print(info["role"], info["models"], "original:", info["source_path"])
    print("Published copy:", log.relative_to(ROOT))
    print(log.read_text()[-12000:])'''),
        nbformat.v4.new_markdown_cell(reproduction)]
    return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


if __name__ == "__main__":
    main()
