"""Publish a completed Part 1 run without altering its training evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys

import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import gpt


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            f"Inconsistent or nonfinite {label}")


def relative_run(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    require(run.is_relative_to(root), "Run must be inside the repository")
    require(run.is_dir(), "Run directory does not exist")
    return run.relative_to(root).as_posix()


def verify_cached_data(root, config, manifest, summary, vocabulary):
    """Stream the frozen split and validate every index/text hash without downloading."""
    paths = {name: Path(value) for name, value in gpt.data_cache_paths(config).items()}
    paths = {name: path if path.is_absolute() else Path(root) / path for name, path in paths.items()}
    require(all(path.is_file() and path.resolve().is_relative_to(Path(root).resolve())
                for path in paths.values()), "Processed data cache is absent or outside repository")
    require(read_json(paths["manifest"]) == manifest, "Processed cache manifest differs from training evidence")
    occupied, training_chars = set(), set()
    measurements = {}
    for split in ("train", "validation"):
        counts = {"stories": 0, "characters": 0, "targets": 0, "windows": 0, "unknown": 0}
        with paths[split].open(encoding="utf-8") as handle:
            for number, line in enumerate(handle):
                require(bool(line.strip()), f"Blank record in processed {split} cache")
                record = json.loads(line)
                require(number < len(manifest[split]), f"Extra processed {split} story")
                expected = manifest[split][number]
                text = record["text"]
                checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
                require(record["index"] == expected["index"] and checksum == expected["sha256"],
                        f"Processed {split} story hash/index mismatch at {number}")
                require(text and checksum not in occupied, "Empty, duplicate, or overlapping processed story")
                occupied.add(checksum)
                counts["stories"] += 1
                counts["characters"] += len(text)
                counts["targets"] += len(text) + 1
                counts["windows"] += math.ceil((len(text) + 1) / config["context_length"])
                if split == "train":
                    training_chars.update(text)
                else:
                    counts["unknown"] += sum(char not in training_chars for char in text)
        require(counts["stories"] == len(manifest[split]), f"Missing processed {split} stories")
        measurements[split] = counts
    require(vocabulary["tokens"] == gpt.SPECIALS + sorted(training_chars),
            "Vocabulary was not built from precisely the frozen training characters")
    require(measurements["train"]["windows"] == summary["train_windows_per_epoch"]
            and measurements["train"]["targets"] == summary["train_targets_per_epoch"]
            and measurements["validation"]["windows"] == summary["validation_windows"]
            and measurements["validation"]["unknown"] == summary["validation_unknown_characters"]
            and measurements["validation"]["characters"] == summary["validation_character_count"],
            "Processed character/window counts disagree with run summary")
    return paths, measurements


def verify_training_evidence(root, run, *, verify_cache=True):
    """Cross-check history, raw events, checkpoint states, data, and runner provenance."""
    root, run = Path(root).resolve(), Path(run).resolve()
    relative = relative_run(root, run)
    summary, config, history = (read_json(run / name) for name in ("summary.json", "config.json", "history.json"))
    require(config["mode"] == summary["mode"] == "full" and config["dataset"] == "roneneldan/TinyStories"
            and config.get("max_steps") is None and summary["is_final_training_run"] is True
            and summary["inference_reload_verified"] is True,
            "Only a completed, verified, uncapped real TinyStories full run can be published")
    epochs = config["epochs"]
    require(epochs >= 10 and summary["completed_full_epochs"] == epochs
            and summary["train_stories"] == config["train_stories"] >= 100000
            and summary["validation_stories"] == config["validation_stories"] >= 10000,
            "Full run does not satisfy required epochs/story counts")
    require([row["epoch"] for row in history] == list(range(1, epochs + 1))
            and all(row["epoch_complete"] and row["epoch_fraction"] == 1 for row in history),
            "Training history does not contain every complete epoch")
    batches = math.ceil(summary["train_windows_per_epoch"] / config["batch_size"])
    require(summary["global_steps"] == summary["planned_steps"] == batches * epochs,
            "Completed step count disagrees with full epoch coverage")
    for row in history:
        require(row["step"] == row["epoch"] * batches and row["mode"] == "full", "Epoch step/mode mismatch")
        for split in ("train", "validation"):
            metrics = row[split]
            loss = metrics["cross_entropy"]
            require(isinstance(loss, (int, float)) and math.isfinite(loss) and loss >= 0,
                    "Invalid cross-entropy in history")
            close(metrics["perplexity"], math.exp(loss), f"{split} perplexity")
            close(metrics["bits_per_character"], loss / math.log(2), f"{split} bits-per-character")
            accuracy = metrics["next_character_accuracy"]
            require(isinstance(accuracy, (int, float)) and math.isfinite(accuracy) and 0 <= accuracy <= 1,
                    "Invalid next-character accuracy")
            targets = (summary["train_targets_per_epoch"] if split == "train" else
                       summary["validation_character_count"] + summary["validation_stories"])
            require(metrics["scored_targets_including_eos"] == targets, "Epoch omitted or duplicated scored targets")
        close(row["generalization_gap_cross_entropy"],
              row["validation"]["cross_entropy"] - row["train"]["cross_entropy"], "generalization gap")
        require(math.isfinite(row["mean_gradient_norm_before_clipping"]), "Nonfinite gradient norm")
    selected = min(history, key=lambda row: row["validation"]["cross_entropy"])
    require(summary["best_checkpoint_epoch"] == selected["epoch"]
            and summary["best_validation"] == selected["validation"]
            and summary["last_epoch_metrics"] == history[-1], "Selected/final metrics differ from history")
    close(summary["best_validation_cross_entropy"], selected["validation"]["cross_entropy"], "best validation loss")
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require([event for event in events if event.get("kind") == "epoch"] == history,
            "Raw epoch events differ from saved history")
    steps = [event for event in events if event.get("kind") == "step"]
    require(all(math.isfinite(event["loss"]) and math.isfinite(event["gradient_norm_before_clipping"])
                for event in steps), "Raw step events include nonfinite training loss/gradient")
    require([row["step"] for row in steps] == sorted(set(row["step"] for row in steps)),
            "Raw step events have duplicate or unordered steps")
    manifest, vocabulary = read_json(run / "data_manifest.json"), read_json(run / "vocabulary.json")
    require(gpt._digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
            == manifest["manifest_sha256"] == summary["manifest_sha256"], "Data manifest checksum mismatch")
    require(len(manifest["train"]) == summary["train_stories"]
            and len(manifest["validation"]) == summary["validation_stories"], "Manifest story count mismatch")
    states = {}
    for name in ("best.pt", "last.pt"):
        state = torch.load(run / "checkpoints" / name, map_location="cpu", weights_only=False)
        require(state["task"] == "character_gpt" and state["config"] == config
                and state["vocabulary"] == vocabulary and state["data_manifest"] == manifest,
                f"{name} has mismatched config, vocabulary, or split manifest")
        require(all(torch.isfinite(value).all().item() for value in state["model"].values()),
                f"Nonfinite checkpoint tensors: {name}")
        expected_epochs = selected["epoch"] if name == "best.pt" else epochs
        progress = state["progress"]
        require(progress["history"] == history[:expected_epochs] and progress["epoch"] == expected_epochs
                and progress["global_step"] == batches * expected_epochs
                and progress["evaluation_pending"] is False and progress["next_batch"] == 0,
                f"{name} disagrees with completed epoch history")
        close(progress["best_validation_loss"], summary["best_validation_cross_entropy"], f"{name} best loss")
        states[name] = state
    model = gpt.CharacterGPT(len(vocabulary["tokens"]), config)
    model.load_state_dict(states["best.pt"]["model"], strict=True)
    require(sum(parameter.numel() for parameter in model.parameters()) == summary["parameter_count"],
            "Model parameter count disagrees with checkpoint/summary")
    provenance = read_json(run / "run_summary.json")
    require(provenance["status"] == "completed" and provenance["task"] == "gpt"
            and provenance["mode"] == "full" and provenance["config"] == config
            and provenance["summary"] == summary and isinstance(provenance["environment"], dict),
            "Runner completion provenance disagrees with training evidence")
    canonical_sources = {name.replace("\\", "/"): value for name, value in provenance["source_sha256"].items()}
    required_sources = {"src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                        "task1_llm/srinidhi/src/gpt.py", "task1_llm/srinidhi/config.json"}
    require(required_sources <= canonical_sources.keys(), "Runner provenance lacks required Part 1 source hashes")
    for canonical, sha256 in canonical_sources.items():
        if canonical.startswith(("src/lab1/", "task1_llm/srinidhi/src/")) or canonical == "task1_llm/srinidhi/config.json":
            require(digest(root / canonical) == sha256, f"Training source changed since run: {canonical}")
    # Startup library warnings can contain host-specific paths. The original
    # console is retained untouched locally; portable evidence uses metrics.jsonl.
    if (run / "RUN_LOG.txt").is_file():
        log = (run / "RUN_LOG.txt").read_text(encoding="utf-8")
        require('"event": "completed"' in log and "Traceback (most recent call last)" not in log,
                "Original console lacks successful runner completion or includes a traceback")
    generations = read_json(run / "generations.json")
    require(len(generations) == 2 * config["generation_prompts"], "Missing required generation outputs")
    for item in generations:
        require(item["full_text"] == item["prompt"] + item["continuation"]
                and item["generated_character_tokens"] == len(item["continuation"])
                and math.isfinite(item["generation_seconds"]) and item["generation_seconds"] > 0,
                "Generation text/count/timing mismatch")
        close(item["generation_tokens_per_second"], len(item["continuation"]) / item["generation_seconds"],
              "generation throughput")
        require(item["metrics"] == gpt.diversity([item["continuation"]]), "Generation diversity differs from text")
    expected_diversity = {name: gpt.diversity([item["continuation"] for item in generations if item["decoding"] == name])
                          for name in ("greedy", "sampled")}
    require(summary["generation_metrics"] == expected_diversity, "Aggregate generation diversity differs from text")
    close(summary["generation_tokens_per_second"],
          sum(len(item["continuation"]) for item in generations) / sum(item["generation_seconds"] for item in generations),
          "aggregate generation throughput")
    for name in ("total_training_seconds", "training_targets_per_second", "elapsed_seconds_including_prior_sessions"):
        require(math.isfinite(summary[name]) and summary[name] > 0, f"Invalid timing/throughput: {name}")
    close(summary["training_targets_per_second"],
          summary["train_targets_per_epoch"] * epochs / summary["total_training_seconds"], "training throughput")
    cache_paths, measurements = verify_cached_data(root, config, manifest, summary, vocabulary) if verify_cache else ({}, {})
    return {"run": relative, "summary": summary, "config": config, "history": history,
            "selected": selected, "steps": steps, "generations": generations,
            "cache_paths": cache_paths, "processed_measurements": measurements}


def verify_failure_analysis(path, generations):
    """Require three grounded excerpts and observations; this cannot judge their interpretation."""
    text = Path(path).read_text(encoding="utf-8")
    require(not re.search(r"\[(?:select|quote|your|insert|TODO)|Student review required", text, re.I),
            "Failure analysis still contains template placeholders")
    sections = re.split(r"(?m)^## Case\s+\d+[^\n]*\n", text)[1:]
    require(len(sections) == 3, "Failure analysis must include exactly three Case sections")
    checks = []
    for index, section in enumerate(sections, 1):
        identifier = re.search(r"Generation ID:\s*`?(\d+)`?", section)
        excerpt = re.search(r"```text\s*\n([\s\S]+?)\n```", section)
        require(identifier is not None and excerpt is not None, f"Case {index}: missing Generation ID or fenced text excerpt")
        number = int(identifier.group(1))
        snippet = excerpt.group(1)
        require(0 <= number < len(generations) and snippet in generations[number]["continuation"],
                f"Case {index}: excerpt is not an exact generated continuation substring")
        require(re.search(r"(?:Failure type|What failed):\s*\S", section)
                and re.search(r"Observation:\s*\S", section), f"Case {index}: missing failure type or observation")
        checks.append({"case": index, "generation_id": number, "excerpt_sha256": hashlib.sha256(snippet.encode()).hexdigest()})
    return checks


def metric_rows(evidence):
    summary, selected, steps = evidence["summary"], evidence["selected"], evidence["steps"]
    rows = []
    def metric(split, name, value):
        rows.append({"model": "character_gpt", "split": split, "metric": name, "value": value,
                     "status": "measured_full_run", "run_id": evidence["run"],
                     "checkpoint": "task1_llm/srinidhi/checkpoints/best.pt"})
    for split in ("train", "validation"):
        for name, value in selected[split].items():
            metric(split, name, value)
    for name in ("generalization_gap_cross_entropy", "mean_gradient_norm_before_clipping"):
        metric("selected_epoch", name, selected[name])
    if steps:
        metric("logged_steps", "maximum_gradient_norm_before_clipping", max(row["gradient_norm_before_clipping"] for row in steps))
        metric("logged_steps", "loss_increases_over_0.5_from_previous_logged_step",
               sum(after["loss"] > before["loss"] + 0.5 for before, after in zip(steps, steps[1:])))
    metric("training", "detected_nonfinite_failures", 0)
    for name in ("parameter_count", "training_targets_per_second", "total_training_seconds",
                 "elapsed_seconds_including_prior_sessions", "generation_tokens_per_second",
                 "cuda_peak_allocated_mb", "host_peak_rss_mb"):
        metric("run", name, summary[name])
    for decoding, values in summary["generation_metrics"].items():
        for name, value in values.items():
            if name != "tokenization":
                metric(decoding, name, value)
    return rows


def build_notebook(root, evidence):
    member = Path(root) / "task1_llm/srinidhi"
    cells = [
        nbformat.v4.new_markdown_cell("# Part 1: character GPT from scratch\n\n"
            "Srinidhi, Lab Pair 49, seed 2342. The preserved complete desktop training run appears below. "
            "This notebook loads the saved weights and executes preprocessing, causal masking, and inference checks. "
            "It displays training evidence without starting another training run. The complete implementation is in `gpt.py`."),
        nbformat.v4.new_code_cell('''from pathlib import Path
import os, sys, json, hashlib, inspect
import pandas as pd
import torch
from IPython.display import display, Image, Code
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "pyproject.toml").is_file() and (p / "src/lab1").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Extract the complete Part 1 folder and open this notebook inside it.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from lab1 import gpt
MEMBER = ROOT / "task1_llm/srinidhi"
''' + f'RUN = ROOT / {evidence["run"]!r}\n' + '''summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
assert summary["is_final_training_run"] and summary["completed_full_epochs"] >= 10
torch.set_num_threads(2)
display({"training_device": summary["device_name"], "torch": summary["torch_version"],
         "cuda": summary["cuda_version"], "epochs": summary["completed_full_epochs"],
         "training_stories": summary["train_stories"], "validation_stories": summary["validation_stories"],
         "parameter_count": summary["parameter_count"], "precision": summary["precision_dtype"]})
display(config)'''),
        nbformat.v4.new_markdown_cell("## Preprocessing and fixed input–target windows\n\n"
            "TinyStories uses this member's seeded permutation of the official training and validation splits. "
            "After normalizing line endings and outer whitespace, empty/duplicate stories are excluded across both splits. "
            "The frozen split records each source index and the normalized story's SHA256. The vocabulary contains training "
            "Unicode characters plus PAD, UNK, BOS, and EOS. Unseen validation characters map to UNK. Each story receives "
            "BOS/EOS tokens and fixed context windows; targets shift inputs by one position. Final windows pad on the right, "
            "and cross-entropy ignores PAD. Windows never cross stories, and each character/EOS target is scored once per epoch."),
        nbformat.v4.new_code_cell('''vocabulary = json.loads((RUN / "vocabulary.json").read_text(encoding="utf-8"))
char_to_idx = {char: index for index, char in enumerate(vocabulary["tokens"])}
idx_to_char = {index: char for char, index in char_to_idx.items()}
assert all(idx_to_char[char_to_idx[char]] == char for char in vocabulary["tokens"])
with pd.option_context("display.max_rows", None):
    display(pd.DataFrame({"idx": list(idx_to_char), "char": list(idx_to_char.values())}))
cache = gpt.data_cache_paths(config)
with (ROOT / cache["train"]).open(encoding="utf-8") as handle:
    first_story = json.loads(next(handle))["text"]
windows = gpt.StoryWindows([first_story], vocabulary, config["context_length"])
x, y = windows[0]
encoded = [gpt.BOS] + [char_to_idx.get(char, gpt.UNK) for char in first_story] + [gpt.EOS]
valid = y.ne(gpt.PAD)
count = int(valid.sum())
assert x[valid].tolist() == encoded[:count]
assert y[valid].tolist() == encoded[1:count+1]
assert sum(int(windows[index][1].ne(gpt.PAD).sum()) for index in range(len(windows))) == len(first_story)+1
print("First normalized training story:", first_story[:300])
display(pd.DataFrame({"input_id": x[:24].tolist(), "input_char": [idx_to_char[int(i)] for i in x[:24]],
                      "target_id": y[:24].tolist(), "target_char": [idx_to_char[int(i)] for i in y[:24]]}))
print("Character dictionaries invert correctly; shifted targets and complete single-story coverage verified.")
print("Validation characters mapped to UNK:", summary["validation_unknown_characters"],
      "of", summary["validation_character_count"])'''),
        nbformat.v4.new_markdown_cell("## Decoder architecture and training implementation\n\n"
            "The decoder sums learned character and learned position embeddings. Each pre-normalized block applies manual "
            "multi-head attention followed by a GELU feed-forward network; both sublayers have residual connections. "
            "The attention code explicitly projects Q/K/V, computes scaled QKᵀ, fills future positions with −∞, applies "
            "softmax and dropout, multiplies by V, and joins/projects heads. The final normalization and linear language "
            "head produce next-character logits. No built-in Transformer or attention module is used.\n\n"
            "AdamW minimizes next-character cross-entropy. The learning rate ramps linearly during the configured warm-up "
            "fraction, then follows cosine decay to the configured minimum ratio. Gradient clipping and finite loss/gradient "
            "checks guard the optimization. Mixed precision is disclosed in the config and hardware outputs."),
        nbformat.v4.new_code_cell('''for component in (gpt.StoryWindows, gpt.ManualCausalAttention, gpt.TransformerBlock,
                  gpt.CharacterGPT, gpt.evaluate, gpt.generate, gpt.run):
    print(component.__name__)
    display(Code(inspect.getsource(component), language="python"))'''),
        nbformat.v4.new_markdown_cell("## All measured metrics and complete epoch history\n\n"
            "Perplexity is exp(cross-entropy), and bits per character is cross-entropy/ln(2). PAD targets are excluded; "
            "EOS and mapped UNK targets are included. The generalization gap is validation eval-mode CE minus online "
            "training-mode CE for the selected checkpoint epoch. Distinct-1/2/3 and repeated 4-gram rate use lowercase "
            "word n-grams in continuations, excluding prompts. Reported timing/memory describe training on the recorded "
            "GPU. Fresh inference checks below have separate timing and do not replace those metrics."),
        nbformat.v4.new_code_cell('''metrics = pd.read_csv(MEMBER / "metrics_report.csv")
with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
    display(metrics[["split", "metric", "value"]])
display(pd.DataFrame({"epoch": [row["epoch"] for row in history],
                      "complete": [row["epoch_complete"] for row in history],
                      "train_ce": [row["train"]["cross_entropy"] for row in history],
                      "validation_ce": [row["validation"]["cross_entropy"] for row in history],
                      "validation_perplexity": [row["validation"]["perplexity"] for row in history],
                      "validation_accuracy": [row["validation"]["next_character_accuracy"] for row in history]}))
for figure in sorted((RUN / "figures").glob("*.png")):
    print(figure.name)
    display(Image(filename=str(figure), width=950))'''),
        nbformat.v4.new_markdown_cell("## Preserved raw training output\n\n"
            "The complete unedited metrics.jsonl training log is displayed here and included in the package. "
            "It records each logged update and every completed epoch. The original local console log is retained "
            "unchanged; its startup warnings include host paths, so that console is omitted from the portable package."),
        nbformat.v4.new_code_cell('print((RUN / "metrics.jsonl").read_text(encoding="utf-8"))'),
        nbformat.v4.new_markdown_cell("## Actual greedy and temperature-sampled text\n\nEach generation below comes from the saved best validation checkpoint. IDs match failure_analysis.md."),
        nbformat.v4.new_code_cell('''generations = json.loads((RUN / "generations.json").read_text(encoding="utf-8"))
for index, item in enumerate(generations):
    print(f"Generation ID: {index} | {item['decoding']} | temperature={item['temperature']} | top_k={item['top_k']} | seed={item['seed']}")
    print(item["full_text"])
    print()'''),
        nbformat.v4.new_markdown_cell((member / "failure_analysis.md").read_text(encoding="utf-8")),
        nbformat.v4.new_markdown_cell("## Saved weights, causal-mask check, and fresh CPU/CUDA inference\n\n"
            "Both best and last saved weights are checked against their hashes. The selected model is reconstructed "
            "from its checkpoint config in a new process. Changing future tokens must leave earlier logits exactly "
            "unchanged. Independently loaded instances must produce identical logits on each available device. "
            "CUDA is checked when available; CPU loading is always checked. This verifies portability without retraining."),
        nbformat.v4.new_code_cell('''folder = MEMBER / "checkpoints"
checkpoint_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
for name, info in checkpoint_manifest.items():
    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == info["sha256"]
saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
assert all(torch.isfinite(value).all() for value in saved["model"].values())
verification = {"checkpoint_sha256": checkpoint_manifest["best.pt"]["sha256"], "devices": []}
devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
for device in devices:
    model = gpt.CharacterGPT(len(saved["vocabulary"]["tokens"]), saved["config"]).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    independent = gpt.CharacterGPT(len(saved["vocabulary"]["tokens"]), saved["config"]).to(device).eval()
    independent.load_state_dict(saved["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == summary["parameter_count"]
    probe = x[:16].unsqueeze(0).to(device)
    modified = probe.clone()
    modified[:, 8:] = (modified[:, 8:] + 1) % len(saved["vocabulary"]["tokens"])
    with torch.no_grad():
        original_logits, altered_logits = model(probe), model(modified)
        assert torch.equal(original_logits, independent(probe))
        assert torch.equal(original_logits[:, :8], altered_logits[:, :8])
        assert torch.isfinite(original_logits).all()
    prompt = "Once upon a time, a little bird"
    continuation = gpt.generate(model, saved["vocabulary"], prompt, 100, device, seed=2342, temperature=0)
    print(f"Fresh {device} inference; reload logits and causal masking verified:")
    print(prompt + continuation)
    verification["devices"].append({"device": str(device), "reload_logits_identical": True,
                                   "causal_mask_verified": True, "finite_logits": True,
                                   "generated_characters": len(continuation)})
    del independent, model
destination = ROOT / "verification/part1_checkpoint_inference.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(verification, indent=2)+"\\n", encoding="utf-8")
print("Both checkpoint hashes verified; fresh inference verification saved.")'''),
        nbformat.v4.new_markdown_cell("## Reproduce and demo\n\n"
            "From the extracted Part 1 root, install the dependencies and package described in README.md. "
            "A quick CPU smoke test uses `python -m lab1.run --task gpt --mode smoke --device cpu`. "
            "The complete frozen data split is included: a fresh full run uses "
            "`python -m lab1.run --task gpt --mode full --device cuda --set offline=true`. "
            "To re-execute only this evidence notebook and package the existing completed run, use "
            f"`python scripts/finalize_gpt.py --run-dir {evidence['run']}`. The individual Part 1 bundle is one "
            "component of the eventual three-part team Canvas submission. Review the model rationale and observed "
            "failure analysis in your own words before the viva."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "lab1": {"task": "gpt", "mode": "completed_full_run_evidence",
                                                        "source_run": evidence["run"]}})
    nbformat.write(notebook, member / "src/gpt.ipynb")


def publish(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    evidence = verify_training_evidence(root, run)
    member = root / "task1_llm/srinidhi"
    failures = verify_failure_analysis(member / "failure_analysis.md", evidence["generations"])
    rows = metric_rows(evidence)
    with (member / "metrics_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    checkpoints = {}
    (member / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt"):
        source, destination = run / "checkpoints" / name, member / "checkpoints" / name
        shutil.copy2(source, destination)
        require(digest(source) == digest(destination), "Checkpoint copy SHA256 mismatch")
        checkpoints[name] = {"path": destination.relative_to(root).as_posix(),
                             "source": source.relative_to(root).as_posix(),
                             "sha256": digest(destination), "bytes": destination.stat().st_size}
    write_json(member / "checkpoints/manifest.json", checkpoints)
    outputs = member / "outputs/full"
    outputs.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "history.json", "generations.json", "vocabulary.json", "config.json", "data_manifest.json"):
        shutil.copy2(run / name, outputs / name)
    shutil.copytree(run / "figures", outputs / "figures", dirs_exist_ok=True)
    summary, config, selected = evidence["summary"], evidence["config"], evidence["selected"]
    (member / "results.md").write_text(
        "# Srinidhi — Part 1 character GPT results\n\n"
        f"Completed {summary['completed_full_epochs']} full epochs on the desktop {summary['device_name']}, "
        f"using {summary['train_stories']:,} training and {summary['validation_stories']:,} validation TinyStories. "
        f"Seed {config['seed']}; no pretrained model, tokenizer, or attention/Transformer module is used. "
        "The frozen member-specific split and its story hashes are in data_processed/full and the raw run data manifest.\n\n"
        f"Architecture: {config['layers']} pre-normalized decoder blocks, {config['heads']} manual attention heads, "
        f"embedding width {config['embedding_dim']}, feed-forward width {config['feedforward_dim']}, "
        f"context {config['context_length']}, learned character and positional embeddings, "
        f"dropout {config['dropout']}, and {summary['parameter_count']:,} parameters. "
        "The compact width/depth balance desktop throughput and model capacity; multiple heads let distinct "
        "positions contribute to a character prediction, while causal masking preserves the autoregressive task. "
        "Pre-normalized residual blocks support stable optimization. The context bounds memory and limits long-range story coherence.\n\n"
        f"Training: AdamW, batch {config['batch_size']}, learning rate {config['learning_rate']}, "
        f"betas {config['adam_betas']}, weight decay {config['weight_decay']}, gradient clip {config['gradient_clip']}, "
        f"linear warm-up over {config['warmup_fraction']:.0%} of updates, then cosine decay to "
        f"{config['minimum_lr_ratio']:.0%} of the base rate. Precision: {summary['precision_dtype']}. "
        "Cross-entropy ignores PAD targets and includes character/EOS targets. The warm-up moderates initial updates; "
        "dropout and weight decay regularize the model. All resolved parameters and exact library versions are preserved.\n\n"
        f"Best validation checkpoint: epoch {summary['best_checkpoint_epoch']}. Training CE at that epoch: "
        f"{selected['train']['cross_entropy']:.6f}; validation CE: {summary['best_validation']['cross_entropy']:.6f}; "
        f"validation perplexity: {summary['best_validation']['perplexity']:.6f}; bits per character: "
        f"{summary['best_validation']['bits_per_character']:.6f}; next-character accuracy: "
        f"{summary['best_validation']['next_character_accuracy']:.4%}; generalization gap: "
        f"{selected['generalization_gap_cross_entropy']:.6f}. All required metrics are in metrics_report.csv.\n\n"
        "Metric interpretation: perplexity=exp(CE); bits per character=CE/ln(2). The gap compares validation "
        "eval-mode CE with the selected epoch's online training-mode CE, so dropout and model updates affect that comparison. "
        "Unseen validation characters map to UNK and are counted. Distinct-1/2/3 and repeated 4-grams use lowercase regex "
        "word tokens in generated continuations; prompts are excluded. Greedy and sampled generation scores are separate. "
        "Sampling parameters and seeds are in generations.json. Training throughput counts scored targets; generation "
        "throughput counts emitted characters and excludes prompts/EOS. CUDA peak memory is allocator-tracked and host "
        "peak RSS is reported separately. The loss-spike count uses successive logged steps and a 0.5 CE increase; "
        "it does not examine every intermediate update. A successful run detected no nonfinite loss/gradient failures.\n\n"
        f"Raw unedited evidence: ../../{evidence['run']}. Both best.pt and last.pt are supplied, with hashes in "
        "checkpoints/manifest.json. The executed gpt.ipynb displays the complete raw metrics.jsonl training log, loss curves, metrics, "
        "and generated texts, then independently verifies fresh checkpoint inference on CPU and CUDA when available. "
        "failure_analysis.md contains three exact observed generation excerpts.\n\n"
        "This completes Srinidhi's Part 1 artifacts. The combined report, cross-member comparisons, and Parts 2/3 "
        "belong to the overall team submission. Review the architecture rationale and failure interpretations before the viva.\n",
        encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "matplotlib", "datasets", "nbformat", "nbclient", "ipykernel")}
    write_json(outputs / "export_environment.json", {"python": sys.version.split()[0], "packages": versions})
    build_notebook(root, evidence)
    report = {"source_run": evidence["run"], "metric_rows": len(rows), "checkpoint_copies": checkpoints,
              "failure_excerpts_verified": failures, "processed_split": evidence["processed_measurements"],
              "raw_training_evidence_unchanged": True}
    write_json(root / "verification/part1_export.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(ROOT, args.run_dir), indent=2))


if __name__ == "__main__":
    main()
