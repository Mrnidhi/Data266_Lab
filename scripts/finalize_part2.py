"""Verify and package completed Part 2 results without training or selecting models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import sentiment as s
import finalize_sentiment_selection as selector
import publish_sentiment_results as publisher
import verify_part2_notebook as notebook_runner

ARCHIVE_ROOT = "Part 2"
DEPENDENCIES = ("torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "datasets", "nbformat", "nbclient", "ipykernel")
RUNTIME_FILES = ("src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                 "task2_sentiment/srinidhi/src/sentiment.py", "task2_sentiment/srinidhi/config.json",
                 "scripts/tune_sentiment.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def metadata(path):
    return {"bytes": Path(path).stat().st_size, "sha256": digest(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_values(actual, expected, label):
    """Compare metric structures without weakening discrete counts or nulls."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{label} keys differ")
        for key, value in expected.items():
            compare_values(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label} shape differs")
        for index, value in enumerate(expected):
            compare_values(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0.0), f"{label} value differs")
    else:
        require(actual == expected, f"{label} value differs")


def verify_prediction_metrics(predictions, metrics, *, seed, bootstrap_samples):
    y = predictions.true_label.to_numpy(dtype=np.int64)
    probability = predictions.positive_probability.to_numpy(dtype=np.float64)
    require(np.array_equal(predictions.predicted_label.to_numpy(dtype=np.int64), (probability >= .5).astype(np.int64)),
            "Prediction labels disagree with the 0.5 threshold")
    measured, _ = s.compute_metrics(y, probability, bootstrap_samples=bootstrap_samples, seed=seed)
    for key, value in measured.items():
        require(key in metrics, f"Required metric missing: {key}")
        compare_values(metrics[key], value, f"prediction metric {key}")
    return probability


def verify_ai_error_draft(packet_path, draft_path, name, checkpoint_sha256):
    """Optional AI interpretations must refer exactly to real, unchanged review cases."""
    packet_path, draft_path = Path(packet_path), Path(draft_path)
    if not draft_path.is_file():
        return None
    packet = pd.read_csv(packet_path, keep_default_na=False, encoding="utf-8").set_index("example_id")
    draft = pd.read_csv(draft_path, keep_default_na=False, encoding="utf-8")
    required = {"example_id", "model", "checkpoint_sha256", "source_text_sha256", "human_packet_sha256",
                "review_group", "true_label", "predicted_label", "positive_probability", "original_token_length",
                "oov_rate", "has_negation", "ai_draft_error_type", "ai_draft_evidence_quotes", "ai_draft_reason",
                "ai_draft_testable_fix", "ai_draft_status"}
    require(required <= set(draft.columns) and len(draft) == draft.example_id.nunique() == len(packet) == 20
            and set(draft.example_id) == set(packet.index), f"AI draft is not paired with all 20 real cases: {name}")
    packet_sha256 = digest(packet_path)
    for _, row in draft.iterrows():
        actual = packet.loc[row.example_id]
        require(row.model == name and row.checkpoint_sha256 == actual.checkpoint_sha256 == checkpoint_sha256
                and row.human_packet_sha256 == packet_sha256
                and row.source_text_sha256 == hashlib.sha256(actual.text.encode("utf-8")).hexdigest(),
                f"AI draft source/checkpoint/text identity changed: {name}/{row.example_id}")
        for column in ("review_group", "true_label", "predicted_label", "original_token_length", "has_negation"):
            require(row[column] == actual[column], f"AI draft case field changed: {name}/{row.example_id}/{column}")
        for column in ("positive_probability", "oov_rate"):
            require(math.isclose(float(row[column]), float(actual[column]), rel_tol=1e-12, abs_tol=0),
                    f"AI draft case field changed: {name}/{row.example_id}/{column}")
        quotes = json.loads(row.ai_draft_evidence_quotes)
        require(isinstance(quotes, list) and bool(quotes)
                and all(isinstance(quote, str) and bool(quote.strip()) and quote in actual.text for quote in quotes),
                f"AI draft quote is not an exact source excerpt: {name}/{row.example_id}")
        require(row.ai_draft_status == "AI_ASSISTED_NOT_STUDENT_REVIEW"
                and all(isinstance(row[column], str) and bool(row[column].strip())
                        for column in ("ai_draft_error_type", "ai_draft_reason", "ai_draft_testable_fix")),
                f"AI draft interpretation/status missing or misrepresented: {name}/{row.example_id}")
    return {"cases": len(draft), "human_packet_sha256": packet_sha256, "all_case_text_hashes_and_quotes_verified": True,
            "scope": "AI draft linkage verified; this does not mark any student review complete."}


def verify_evaluation(root, run, data_dir, *, verify_sources=True):
    root, run, data_dir = (Path(path).resolve() for path in (root, run, data_dir))
    require(run.is_relative_to(root) and data_dir.is_relative_to(root), "Run/data must remain within repository")
    summary, cfg, provenance = (read_json(run / name) for name in ("summary.json", "config.json", "run_summary.json"))
    require(provenance["status"] == "completed" and provenance["mode"] == summary["mode"] == "full"
            and summary["synthetic"] is False and set(summary["models"]) == set(s.MODEL_NAMES)
            and provenance["summary"] == summary, "Part 2 requires completed official-data full evaluation")
    audit = read_json(run / "data_audit.json")
    counts = {"train": 504000, "validation": 56000, "test": 38000}
    require(audit["split_counts"] == counts, "Official frozen split counts are incomplete")
    data_manifest = read_json(data_dir / "manifest.json")
    require(data_manifest["split_counts"] == counts and data_manifest["seed"] == cfg["seed"],
            "Processed data manifest count/seed differs from evaluation")
    for split in counts:
        require(digest(data_dir / f"{split}.jsonl") == data_manifest["sha256"][split],
                f"Frozen {split} data checksum mismatch")
    feature_cache = Path(cfg["encoded_cache"])
    feature_cache = feature_cache.resolve() if feature_cache.is_absolute() else (root / feature_cache).resolve()
    require(feature_cache.is_relative_to(root), "Encoded cache must remain within repository")
    features = read_json(feature_cache / "manifest.json")
    require(features["audit"]["split_counts"] == counts and features["vocabulary"] == read_json(run / "vocabulary.json"),
            "Encoded feature vocabulary/counts differ from evaluation")
    for split in counts:
        require(digest(feature_cache / f"{split}.npz") == features["sha256"][split], f"Encoded {split} checksum mismatch")
    with (data_dir / "test.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    test_ids = [row["id"] for row in records]
    labels = [row["label"] for row in records]
    require(len(records) == 38000 and len(set(test_ids)) == 38000, "Test IDs are incomplete or duplicated")
    selection_path = run / "selection_manifest.json"
    require(selection_path.is_file(), "Fresh Part 2 publication requires an explicit frozen validation selection")
    selection = read_json(selection_path)
    require(set(selection["selected"]) == set(s.MODEL_NAMES)
            and digest(selection_path) == provenance["selection_manifest_sha256"] == summary["selection_manifest_sha256"],
            "Frozen selection hash/families differ from evaluation")
    sources = {}
    probabilities = {}
    runtime_hashes = {relative: digest(root / relative) for relative in RUNTIME_FILES}
    for name in s.MODEL_NAMES:
        chosen = selection["selected"][name]
        source_cfg = chosen["config"]
        require(chosen["provenance"].get("training_source_sha256") == runtime_hashes,
                f"Current runtime differs from frozen desktop training sources: {name}")
        if verify_sources:
            sources[name] = selector.verify_source(root, name, chosen, read_json(root / "task2_sentiment/srinidhi/config.json")["full"])
        metrics = read_json(run / name / "metrics.json")
        require(metrics == summary["models"][name] and metrics["count"] == 38000
                and metrics["selected_epoch"] == chosen["selected_epoch"]
                and metrics["selected_validation_macro_f1"] == chosen["validation_macro_f1"],
                f"Selected checkpoint/evaluation metadata mismatch: {name}")
        require(digest(run / name / "checkpoints/best.pt") == chosen["sha256"], f"Selected weight hash mismatch: {name}")
        history = read_json(run / name / "history.json")
        require([row["epoch"] for row in history] == list(range(1, len(history) + 1)), f"Epoch history incomplete: {name}")
        compare_values(metrics["train_seconds"], sum(row["train_seconds"] for row in history), f"{name} train time")
        predictions = pd.read_csv(run / name / "test_predictions.csv", encoding="utf-8")
        require(predictions.example_id.tolist() == test_ids and predictions.true_label.tolist() == labels,
                f"Test prediction IDs/labels are not paired with frozen records: {name}")
        probabilities[name] = verify_prediction_metrics(predictions, metrics, seed=source_cfg["seed"],
                                                        bootstrap_samples=source_cfg["bootstrap_samples"])
        slice_values = read_json(run / name / "slices.json")
        masks = {"length_lt50": predictions.original_token_length < 50,
                 "length_50_199": (predictions.original_token_length >= 50) & (predictions.original_token_length < 200),
                 "length_ge200": predictions.original_token_length >= 200,
                 "negation_present": predictions.has_negation.astype(bool),
                 "negation_absent": ~predictions.has_negation.astype(bool),
                 "oov_lt10pct": predictions.oov_rate < .1, "oov_ge10pct": predictions.oov_rate >= .1}
        for slice_name, mask in masks.items():
            computed = (s.compute_metrics(np.asarray(labels)[mask], probabilities[name][mask])[0]
                        if mask.any() else {"count": 0})
            compare_values(slice_values[slice_name], computed, f"{name} slice {slice_name}")
    expected_paired = {name: s.paired_mcnemar(labels, probabilities["maxpool_mlp"], probabilities[name])
                       for name in s.MODEL_NAMES[1:]}
    compare_values(summary["paired_mcnemar"], expected_paired, "paired McNemar summary")
    compare_values(read_json(run / "paired_mcnemar.json"), expected_paired, "paired McNemar file")
    return {"run": run.relative_to(root).as_posix(), "summary": summary, "config": cfg, "selection": selection,
            "data_dir": data_dir, "feature_cache": feature_cache, "sources": sources, "test_rows": len(records),
            "metric_recomputation_verified": True}


def verify_publication(root, evidence):
    root = Path(root)
    member = root / "task2_sentiment/srinidhi"
    manifest = read_json(member / "checkpoints/manifest.json")
    inference = read_json(root / "verification/part2_checkpoint_inference.json")
    execution = read_json(root / "verification/part2_notebook.json")
    require(execution["sha256"] == digest(member / "src/sentiment.ipynb"), "Executed notebook bytes changed after verification")
    require(set(manifest) == set(inference["models"]) == set(s.MODEL_NAMES), "All three published models are required")
    review_status = {}
    ai_drafts = {}
    published = member / "outputs/full"
    raw = root / evidence["run"]
    with (evidence["data_dir"] / "test.jsonl").open(encoding="utf-8") as handle:
        test_texts = {record["id"]: record["text"] for line in handle if line.strip()
                      for record in [json.loads(line)]}
    for filename in ("summary.json", "vocabulary.json", "preprocessing.json", "data_audit.json",
                     "dataset_statistics.json", "split_ids.npz", "paired_mcnemar.json", "selection_manifest.json"):
        require(digest(published / filename) == digest(raw / filename), f"Published evidence differs from source: {filename}")
    for filename in ("config.json", "run_summary.json"):
        require(read_json(published / filename) == publisher.normalize_repository_paths(read_json(raw / filename), root),
                f"Published derived metadata changed more than repository paths: {filename}")
    table_metrics = pd.read_csv(member / "metrics_report.csv", keep_default_na=False)
    require(set(table_metrics.model) == set(s.MODEL_NAMES)
            and not table_metrics.duplicated(["model", "split", "metric"]).any(), "Published metric models/rows are incomplete or duplicated")
    indexed = table_metrics.set_index(["model", "split", "metric"])
    def csv_metric(name, split, metric, expected):
        key = (name, split, metric)
        require(key in indexed.index, f"Required metric CSV row missing: {key}")
        value = indexed.loc[key, "value"]
        if isinstance(expected, (dict, list)):
            actual = json.loads(value)
        elif expected is None:
            actual = None if value in ("", "null") else value
        else:
            actual = float(value)
        compare_values(actual, expected, f"published metric {key}")
    for name, files in manifest.items():
        metrics = evidence["summary"]["models"][name]
        for key in ("accuracy", "confusion_matrix", "roc_auc", "pr_auc_trapezoidal", "mcc", "brier", "ece_15", "test_loss"):
            csv_metric(name, "test", key, metrics[key])
        for average in ("macro", "micro", "weighted"):
            for key, value in metrics[average].items():
                csv_metric(name, "test", f"{average}_{key}", value)
        for key, bounds in metrics["bootstrap_95_percentile"].items():
            for suffix in ("low", "high"):
                csv_metric(name, "test", f"{key}_95ci_{suffix}", bounds[suffix])
        for key in ("parameter_count", "train_seconds", "peak_cuda_memory_bytes", "selected_epoch",
                    "selected_validation_macro_f1", "completed_training_steps"):
            csv_metric(name, "training", key, metrics[key])
        history = read_json(raw / name / "history.json")
        csv_metric(name, "training", "examples_per_second", 504000 * len(history) / metrics["train_seconds"])
        if "peak_host_rss_bytes" in metrics:
            csv_metric(name, "training", "peak_host_rss_bytes", metrics["peak_host_rss_bytes"])
        for slice_name, values in read_json(raw / name / "slices.json").items():
            csv_metric(name, slice_name, "support", values["count"])
            if values["count"]:
                csv_metric(name, slice_name, "macro_f1", values["macro"]["f1"])
                csv_metric(name, slice_name, "error_rate", 1 - values["accuracy"])
        if name != "maxpool_mlp":
            for key in ("table", "discordant_pairs", "pvalue_exact_two_sided"):
                csv_metric(name, "test", f"paired_mcnemar_vs_baseline_{key}", evidence["summary"]["paired_mcnemar"][name][key])
        for path in (raw / name).iterdir():
            if path.is_file() and path.name != "required_20_errors_for_review.csv":
                require(digest(published / name / path.name) == digest(path), f"Published model artifact differs: {name}/{path.name}")
        for filename in ("best.pt", "last.pt"):
            require(metadata(root / files[filename]["path"]) == {key: files[filename][key] for key in ("bytes", "sha256")},
                    f"Published weight hash/size mismatch: {name}/{filename}")
            require(digest(root / files[filename]["path"]) == digest(root / evidence["run"] / name / "checkpoints" / filename),
                    f"Published weights differ from completed evaluation: {name}/{filename}")
        record = inference["models"][name]
        require(record["checkpoint_sha256"] == files["best.pt"]["sha256"]
                and any(row["device"] == "cpu" for row in record["devices"])
                and all(row["finite_logits"] and row["independent_reload_identical"] and row["right_padding_invariant"]
                        for row in record["devices"]), f"Fresh checkpoint inference verification failed: {name}")
        require(all(row["parameter_count"] == metrics["parameter_count"] for row in record["devices"]),
                f"Fresh model parameter count differs from reported count: {name}")
        if execution["inference_device"] != "cpu" and torch.cuda.is_available():
            require(any(row["device"] == "cuda" for row in record["devices"]), f"Available CUDA was not checked: {name}")
        table = pd.read_csv(member / "outputs/full" / name / "required_20_errors_for_review.csv", keep_default_na=False)
        require(len(table) == table.example_id.nunique() == 20
                and table.groupby("review_group").size().to_dict() == {
                    "confident_false_positive": 5, "confident_false_negative": 5,
                    "near_threshold": 5, "long_review_slice": 5}, f"Required 20-error selection incomplete: {name}")
        require(table.checkpoint_sha256.eq(files["best.pt"]["sha256"]).all(), f"Review rows refer to another checkpoint: {name}")
        predictions = pd.read_csv(raw / name / "test_predictions.csv", keep_default_na=False).set_index("example_id")
        for _, row in table.iterrows():
            require(row.example_id in predictions.index, f"Review ID absent from actual predictions: {name}")
            measured = predictions.loc[row.example_id]
            require(row.true_label == measured.true_label and row.predicted_label == measured.predicted_label
                    and row.true_label != row.predicted_label and row.text == test_texts[row.example_id]
                    and math.isclose(row.positive_probability, measured.positive_probability, rel_tol=1e-12, abs_tol=0),
                    f"Review prediction fields were changed: {name}/{row.example_id}")
        review_status[name] = int(table.student_reviewed.astype(str).str.lower().isin(["true", "1"]).sum())
        draft_path = published / name / "ai_error_review_draft.csv"
        ai_check = verify_ai_error_draft(published / name / "required_20_errors_for_review.csv", draft_path,
                                        name, files["best.pt"]["sha256"])
        if ai_check is not None:
            ai_drafts[name] = ai_check
    return {"notebook": notebook_runner.validate_notebook(member / "src/sentiment.ipynb"),
            "fresh_inference_verified": True, "weights_verified": True, "error_rows_per_model": 20,
            "student_reviewed_rows": review_status, "ai_draft_linkage": ai_drafts}


def inventory(root, directories):
    root = Path(root)
    return {path.relative_to(root).as_posix(): metadata(path) for folder in directories
            for path in sorted(Path(folder).rglob("*")) if path.is_file()}


def frozen_inputs(root, evidence):
    """Record data, runtime and predeclared recipe bytes for later archive-only gates."""
    root = Path(root)
    files = {root / relative for relative in RUNTIME_FILES}
    for folder, extension in ((evidence["data_dir"], "jsonl"), (evidence["feature_cache"], "npz")):
        files.add(folder / "manifest.json")
        files.update(folder / f"{split}.{extension}" for split in ("train", "validation", "test"))
    parent = (root / evidence["run"]).parent
    if (parent / "desktop_plan.json").is_file():
        files.add(parent / "desktop_plan.json")
        files.update((parent / "recipes").glob("*.json"))
    return {path.relative_to(root).as_posix(): metadata(path) for path in sorted(files)}


def technical_publication_files(root):
    """Freeze displayed metrics, source metadata, figures and notebook, allowing review additions."""
    root = Path(root)
    member = root / "task2_sentiment/srinidhi"
    files = [member / "src/sentiment.ipynb", member / "metrics_report.csv", member / "checkpoints/manifest.json"]
    files.extend(path for path in (member / "outputs/full").rglob("*") if path.is_file()
                 and path.name != "required_20_errors_for_review.csv" and "review_history" not in path.parts)
    return {path.relative_to(root).as_posix(): metadata(path) for path in sorted(files)}


def verify_recorded_files(root, recorded, label):
    require(bool(recorded), f"Missing {label} hash inventory")
    root = Path(root).resolve()
    for relative, expected in recorded.items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative
                and str(path) == relative, f"Noncanonical {label} inventory path")
        actual = root / relative
        require(actual.is_file() and not actual.is_symlink() and actual.resolve().is_relative_to(root)
                and metadata(actual) == expected, f"{label} bytes changed: {relative}")


def package_files(root, evidence):
    root = Path(root).resolve()
    paths = {}
    def add(path):
        path = Path(path)
        if path.is_file():
            require(not path.is_symlink() and path.resolve().is_relative_to(root), "Package file escapes repository")
            paths[path.relative_to(root).as_posix()] = path
    for folder in (root / "src/lab1", root / "task2_sentiment/srinidhi"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
                if "data_processed" in path.parts and not path.is_relative_to(evidence["data_dir"]) and "full_encoded" not in path.parts and path.name != "README.md":
                    continue
                add(path)
    for folder in (root / "task2_sentiment", root / "task2_sentiment/data"):
        for path in folder.glob("*.md"):
            add(path)
    source_runs = {root / selected["source_run"] for selected in evidence["selection"]["selected"].values()}
    source_runs.add(root / evidence["run"])
    for folder in source_runs:
        for path in folder.rglob("*"):
            if path.is_file():
                # Keep personal-path originals local; publisher preserves portable
                # exact epoch/metric copies and records the omitted console hash.
                if path.name == "RUN_LOG.txt" and publisher.re.search(rb"[A-Za-z]:[\\/]+Users[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/", path.read_bytes()):
                    continue
                add(path)
    # Preserve the plan/recipes adjacent to the selected suite when available.
    parent = (root / evidence["run"]).parent
    add(parent / "desktop_plan.json")
    for path in (parent / "recipes").glob("*.json"):
        add(path)
    for name in ("AI_USE.md", "PART2_FINALIZATION.md", ".gitattributes", ".gitignore"):
        add(root / name)
    for name in ("publish_sentiment_results.py", "verify_part2_notebook.py", "finalize_part2.py", "tune_sentiment.py", "run_sentiment_desktop.py",
                 "select_sentiment_candidates.py", "finalize_sentiment_selection.py", "sentiment_data_analysis.py"):
        add(root / "scripts" / name)
    add(root / "scripts/watch_sentiment_desktop.ps1")
    for pattern in ("verification/part2*.json", "verification/part2*.xml", "tests/test_*sentiment*.py", "tests/test_part2*.py"):
        for path in root.glob(pattern):
            add(path)
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    requirements = "# Part 2 direct dependencies; install the tested CUDA torch wheel first.\n" + "".join(
        f"{name}=={version.split('+')[0]}\n" for name, version in versions.items())
    paths["requirements.txt"] = requirements.encode()
    deps = ", ".join(json.dumps(f"{name}=={version.split('+')[0]}") for name, version in versions.items())
    paths["pyproject.toml"] = ("[build-system]\nrequires = [\"setuptools>=77,<82\"]\nbuild-backend = \"setuptools.build_meta\"\n\n"
        "[project]\nname = \"data266-part2-2342\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n"
        f"dependencies = [{deps}]\n\n[project.optional-dependencies]\ntest = [\"pytest>=8\"]\n\n"
        "[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.pytest.ini_options]\npythonpath = [\"src\"]\ntestpaths = [\"tests\"]\n").encode()
    paths["README.md"] = ("# Part 2 — Srinidhi's desktop Yelp classifiers\n\n"
        "Extract this complete Part 2 folder and use it as the working directory. Repository: https://github.com/Mrnidhi/Data266_Lab\n\n"
        "Create a Python 3.12 virtual environment. On Windows:\n\n```powershell\n"
        "py -3.12 -m venv .venv\n.venv\\Scripts\\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128\n"
        ".venv\\Scripts\\python.exe -m pip install -r requirements.txt\n.venv\\Scripts\\python.exe -m pip install --no-deps -e .\n"
        ".venv\\Scripts\\python.exe -m lab1.run --task sentiment --mode smoke --device cpu\n```\n\n"
        "The last command is the one-command offline CPU smoke test. It uses synthetic data; the recorded metrics use all official rows. "
        "Linux/macOS use python3.12 and .venv/bin/python. Open task2_sentiment/srinidhi/src/sentiment.ipynb to see executed results. "
        "Select the virtual-environment kernel; the notebook checks CPU and available CUDA saved-model inference without retraining.\n\n"
        f"To re-finalize the existing saved run:\n\n```powershell\n.venv\\Scripts\\python.exe scripts/finalize_part2.py --run-dir {evidence['run']} "
        f"--data-dir {evidence['data_dir'].relative_to(root).as_posix()}\n```\n\n"
        "Use --device cpu when CUDA is unavailable. Exact three training recipes, frozen selection, original logs, best/last weights, "
        "processed JSONL data and encoded features are included. See member results.md for commands that retrain into new folders; "
        "create a new validation-only selection before evaluating those newly trained models.\n\n"
        "Prior published results and annotations are retained under outputs/publication_history and review_history. They are historical "
        "evidence, separate from this fresh desktop run. Error interpretations require honest student review; new rows remain unreviewed. "
        "This Part 2 technical bundle is one component of the eventual three-part team Canvas ZIP and combined Report.pdf.\n").encode()
    if (root / "PART2_FINALIZATION.md").is_file():
        paths["README.md"] = (root / "PART2_FINALIZATION.md").read_bytes()
    derived_metadata = []
    # The preserved evaluator writes machine-local cache paths. Portable copies
    # change those paths only; original local bytes and hashes remain recorded.
    for filename in ("config.json", "resolved_config.json", "run_summary.json"):
        relative = f"{evidence['run']}/{filename}"
        if relative in paths:
            original = paths[relative].read_bytes()
            document = json.loads(original)
            portable = publisher.normalize_repository_paths(document, root)
            if portable != document:
                published = (json.dumps(portable, indent=2, ensure_ascii=False) + "\n").encode()
                paths[relative] = published
                derived_metadata.append({"path": relative, "original_sha256": hashlib.sha256(original).hexdigest(),
                                         "published_sha256": hashlib.sha256(published).hexdigest(),
                                         "scope": "Derived archive copy: exact repository prefix normalized in string values; original local bytes unchanged."})
    return paths, versions, derived_metadata


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP paths")
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            require(not name.is_absolute() and ".." not in name.parts and "\\" not in item.filename
                    and str(name) == item.filename and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16),
                    "Unsafe or noncanonical ZIP path")
        require(archive.testzip() is None, "Part 2 ZIP CRC verification failed")
        manifest_path = f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json"
        manifest = json.loads(archive.read(manifest_path))
        require(manifest["format_version"] == 1 and manifest["scope"] == "Srinidhi Part 2 only", "Unsupported package manifest")
        files = manifest["files"]
        require(set(names) == {manifest_path, *(f"{ARCHIVE_ROOT}/{name}" for name in files)}, "ZIP inventory differs from manifest")
        for relative, expected in files.items():
            entry = archive.getinfo(f"{ARCHIVE_ROOT}/{relative}")
            value = hashlib.sha256()
            with archive.open(entry) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(chunk)
            require(entry.file_size == expected["bytes"] and value.hexdigest() == expected["sha256"], f"ZIP SHA256/size mismatch: {relative}")
        required = {"pyproject.toml", "requirements.txt", "README.md", "src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                    "task2_sentiment/srinidhi/src/sentiment.py", "task2_sentiment/srinidhi/src/sentiment.ipynb",
                    "task2_sentiment/srinidhi/config.json", "task2_sentiment/srinidhi/metrics_report.csv"}
        required |= {f"task2_sentiment/srinidhi/checkpoints/{name}/{filename}" for name in s.MODEL_NAMES for filename in ("best.pt", "last.pt")}
        required |= {f"{manifest['source_run']}/{filename}" for filename in ("selection_manifest.json", "summary.json", "run_summary.json", "data_audit.json")}
        required |= {f"{manifest['data_dir']}/{filename}" for filename in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")}
        required |= {f"{manifest['feature_cache']}/{filename}" for filename in ("manifest.json", "train.npz", "validation.npz", "test.npz")}
        required |= {f"task2_sentiment/srinidhi/outputs/full/{name}/required_20_errors_for_review.csv" for name in s.MODEL_NAMES}
        require(required <= files.keys(), "Package lacks required Part 2 runtime/evidence/weights")
        selection = json.loads(archive.read(f"{ARCHIVE_ROOT}/{manifest['source_run']}/selection_manifest.json"))
        for name in s.MODEL_NAMES:
            checkpoint = selection["selected"][name]["checkpoint"]
            require(checkpoint in files and files[checkpoint]["sha256"] == selection["selected"][name]["sha256"],
                    f"Archived source checkpoint disagrees with selection: {name}")
        require(all(not name.startswith(("task1_", "task3_", "report/")) for name in files), "Part 2 ZIP includes another part or report")
        notebook = nbformat.reads(archive.read(f"{ARCHIVE_ROOT}/task2_sentiment/srinidhi/src/sentiment.ipynb").decode(), as_version=4)
        notebook_runner.validate_notebook(notebook)
    return {"files": len(files) + 1, "crc_and_content_hashes_verified": True, "contains_trained_final_weights": True,
            "contains_executed_notebook": True, "scope": "Srinidhi Part 2 only; not combined Canvas submission"}


def create_archive(root, evidence, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(destination.suffix.lower() == ".zip", "Archive destination must end in .zip")
    protected_runs = {root / evidence["run"], *(root / chosen["source_run"] for chosen in evidence["selection"]["selected"].values())}
    require(not any(destination.is_relative_to(folder.resolve()) for folder in protected_runs),
            "Archive destination cannot be inside immutable training/evaluation evidence")
    files, versions, derived_metadata = package_files(root, evidence)
    inventory = {name: metadata(value) if isinstance(value, Path) else {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                 for name, value in sorted(files.items())}
    manifest = {"format_version": 1, "scope": "Srinidhi Part 2 only", "source_run": evidence["run"],
                "data_dir": evidence["data_dir"].relative_to(root).as_posix(),
                "feature_cache": evidence["feature_cache"].relative_to(root).as_posix(),
                "derived_metadata_copies": derived_metadata, "runtime_packages": versions, "files": inventory}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted(files.items()):
                if isinstance(value, Path):
                    require(metadata(value) == inventory[name], f"File changed before packaging: {name}")
                    archive.write(value, f"{ARCHIVE_ROOT}/{name}")
                    require(metadata(value) == inventory[name], f"File changed during packaging: {name}")
                else:
                    archive.writestr(f"{ARCHIVE_ROOT}/{name}", value)
            archive.writestr(f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        result = verify_archive(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(archive=destination.relative_to(root).as_posix() if destination.is_relative_to(root) else destination.name,
                  **metadata(destination))
    publisher.write_json(root / "dist/part2_package_verification.json", result)
    (root / "dist/Part2_SHA256SUMS.txt").write_text(result["sha256"] + "  " + destination.name + "\n", encoding="utf-8")
    return result


def finalize(root, run, data_dir, *, device="auto", destination=None):
    root, run = Path(root).resolve(), Path(run).resolve()
    evidence = verify_evaluation(root, run, data_dir)
    raw_directories = [run, *(root / row["source_run"] for row in evidence["selection"]["selected"].values())]
    before = inventory(root, raw_directories)
    publisher.ROOT = root
    publisher.publish(run, data_dir, verify=False)
    notebook = notebook_runner.execute_notebook(root, device)
    checks = verify_publication(root, evidence)
    require(inventory(root, raw_directories) == before, "Raw training/evaluation evidence changed during publication")
    publisher.write_json(root / "verification/part2_finalization.json", {"source_run": evidence["run"],
        "raw_evidence_unchanged": True, "raw_files": before, "recomputed_test_metrics": True,
        "data_dir": evidence["data_dir"].relative_to(root).as_posix(),
        "feature_cache": evidence["feature_cache"].relative_to(root).as_posix(),
        "frozen_inputs": frozen_inputs(root, evidence),
        "published_technical_files": technical_publication_files(root),
        "notebook": notebook, "publication_checks": checks, "manual_error_review_complete": False})
    result = create_archive(root, evidence, destination or root / "dist/Part2_Srinidhi_2342.zip")
    require(inventory(root, raw_directories) == before, "Raw evidence changed during packaging")
    return result


def archive_only(root, run, data_dir, *, destination=None):
    """Rebuild after documentation/review additions without repeating publication."""
    root, run, data_dir = (Path(path).resolve() for path in (root, run, data_dir))
    require(run.is_relative_to(root) and data_dir.is_relative_to(root), "Run/data must remain within repository")
    receipt = read_json(root / "verification/part2_finalization.json")
    require(receipt["source_run"] == run.relative_to(root).as_posix()
            and receipt.get("raw_evidence_unchanged") is True and receipt.get("recomputed_test_metrics") is True,
            "Archive-only requires a successful full finalization of this exact run")
    require(receipt.get("data_dir") == data_dir.relative_to(root).as_posix(), "Archive-only data directory changed")
    verify_recorded_files(root, receipt.get("raw_files"), "Raw training/evaluation")
    verify_recorded_files(root, receipt.get("frozen_inputs"), "Frozen data/runtime/recipes")
    verify_recorded_files(root, receipt.get("published_technical_files"), "Published technical evidence")
    summary, config, provenance = (read_json(run / name) for name in ("summary.json", "config.json", "run_summary.json"))
    selection = read_json(run / "selection_manifest.json")
    require(provenance["status"] == "completed" and provenance["summary"] == summary
            and set(selection["selected"]) == set(summary["models"]) == set(s.MODEL_NAMES)
            and digest(run / "selection_manifest.json") == provenance["selection_manifest_sha256"] == summary["selection_manifest_sha256"],
            "Completed selection/evaluation metadata differs from the verified run")
    feature_cache = Path(config["encoded_cache"])
    feature_cache = feature_cache.resolve() if feature_cache.is_absolute() else (root / feature_cache).resolve()
    require(feature_cache.is_relative_to(root) and receipt.get("feature_cache") == feature_cache.relative_to(root).as_posix(),
            "Archive-only encoded feature directory changed")
    evidence = {"run": receipt["source_run"], "summary": summary, "config": config,
                "selection": selection, "data_dir": data_dir, "feature_cache": feature_cache}
    directories = [run, *(root / row["source_run"] for row in selection["selected"].values())]
    require(inventory(root, directories) == receipt["raw_files"], "Raw evidence inventory changed after full finalization")
    require(frozen_inputs(root, evidence) == receipt["frozen_inputs"], "Frozen input inventory changed after full finalization")
    checks = verify_publication(root, evidence)
    portability_path = root / "verification/part2_package_portability.json"
    if portability_path.is_file():
        portability = read_json(portability_path)
        require(portability.get("runtime_sha256") == {name: digest(root / name) for name in RUNTIME_FILES}
                and portability.get("original_notebook_unchanged") is True
                and portability.get("imports_isolated_with_extracted_src") is True,
                "Runtime differs from the extracted package portability proof")
    publisher.write_json(root / "verification/part2_archive_only.json", {
        "source_run": evidence["run"], "prior_full_metric_recomputation_verified": True,
        "raw_evidence_hashes_unchanged": True, "frozen_data_runtime_recipes_unchanged": True,
        "published_technical_evidence_unchanged": True,
        "notebook_reexecuted": False, "publication_recreated": False, "bootstrap_recomputed": False,
        "publication_checks": checks,
        "scope": "Archive rebuild after documentation/review additions; original completed-run evidence remains unchanged."})
    result = create_archive(root, evidence, destination or root / "dist/Part2_Srinidhi_2342.zip")
    require(inventory(root, directories) == receipt["raw_files"], "Raw evidence changed during archive-only rebuild")
    require(frozen_inputs(root, evidence) == receipt["frozen_inputs"], "Frozen inputs changed during archive-only rebuild")
    return result


def verify_portability(root, archive):
    """Execute extracted notebook/smoke in child processes with isolated imports."""
    root, archive = Path(root).resolve(), Path(archive).resolve()
    archive_checks = verify_archive(archive)
    original_notebook = root / "task2_sentiment/srinidhi/src/sentiment.ipynb"
    before = digest(original_notebook)
    with TemporaryDirectory(prefix="part2-portability-") as temporary:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temporary)  # Safe names/symlinks were validated above.
        extracted = Path(temporary) / ARCHIVE_ROOT
        hashes = {name: digest(extracted / name) for name in RUNTIME_FILES}
        for name, checksum in hashes.items():
            require(digest(root / name) == checksum, f"Extracted runtime differs from tested source: {name}")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(extracted / "src")
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["MPLBACKEND"] = "Agg"
        commands = [
            [sys.executable, "scripts/verify_part2_notebook.py", "--device", "cpu"],
            [sys.executable, "-m", "lab1.run", "--task", "sentiment", "--mode", "smoke", "--device", "cpu",
             "--output", "runs/part2-package-smoke"],
        ]
        for command in commands:
            outcome = subprocess.run(command, cwd=extracted, env=environment, capture_output=True,
                                     text=True, encoding="utf-8", timeout=900)
            if outcome.returncode:
                raise RuntimeError("Extracted Part 2 portability command failed:\n" + outcome.stdout + outcome.stderr)
        notebook = read_json(extracted / "verification/part2_notebook.json")
        smoke = read_json(extracted / "runs/part2-package-smoke/run_summary.json")
        require(smoke["status"] == "completed" and smoke["mode"] == "smoke"
                and set(smoke["summary"]["models"]) == set(s.MODEL_NAMES), "Standalone smoke did not complete all three models")
        fresh = read_json(extracted / "verification/part2_checkpoint_inference.json")
        require(set(fresh["models"]) == set(s.MODEL_NAMES) and all(any(device["device"] == "cpu" and device["finite_logits"]
                        for device in value["devices"]) for value in fresh["models"].values()),
                "Extracted notebook failed fresh CPU weight inference")
        require(digest(original_notebook) == before, "Original notebook changed during isolated portability checks")
        receipt = {"scope": "Extracted standalone Part 2 notebook and offline CPU smoke; recorded pinned interpreter",
                   "runtime_sha256": hashes, "package_crc_and_hashes_verified": archive_checks["crc_and_content_hashes_verified"],
                   "extracted_notebook": notebook, "standalone_cpu_smoke": {"status": smoke["status"], "models": list(s.MODEL_NAMES)},
                   "fresh_cpu_checkpoint_inference": fresh, "imports_isolated_with_extracted_src": True,
                   "original_notebook_unchanged": True}
    publisher.write_json(root / "verification/part2_package_portability.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-only", action="store_true", help="Verify saved finalization hashes/publication, then rebuild ZIP without republishing or executing")
    parser.add_argument("--verify-archive", type=Path)
    parser.add_argument("--verify-portability", type=Path, help="Verify ZIP, then run its extracted CPU notebook/smoke in isolated child processes")
    args = parser.parse_args()
    if args.verify_portability:
        print(json.dumps(verify_portability(ROOT, args.verify_portability), indent=2))
    elif args.verify_archive:
        print(json.dumps(verify_archive(args.verify_archive), indent=2))
    else:
        if args.run_dir is None or args.data_dir is None:
            parser.error("--run-dir and --data-dir are required")
        action = archive_only if args.archive_only else finalize
        options = {"destination": args.archive}
        if not args.archive_only:
            options["device"] = args.device
        print(json.dumps(action(ROOT, args.run_dir, args.data_dir, **options), indent=2))


if __name__ == "__main__":
    main()
