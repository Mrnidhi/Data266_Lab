"""Export a completed GPT run without changing its raw evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    relative = run.relative_to(ROOT).as_posix()
    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())
    if not (summary["is_final_training_run"] and summary["completed_full_epochs"] >= 10
            and summary["train_stories"] >= 100000 and summary["validation_stories"] >= 10000
            and summary["inference_reload_verified"]):
        raise ValueError("Only a completed, verified full run can be published as results")
    selected = next(row for row in history if row["epoch"] == summary["best_checkpoint_epoch"])
    member = ROOT / "task1_llm/srinidhi"
    checkpoint = "task1_llm/srinidhi/checkpoints/best.pt"
    rows = []

    def metric(split, name, value):
        rows.append({"model": "character_gpt", "split": split, "metric": name,
                     "value": value, "status": "measured_full_run", "run_id": relative,
                     "checkpoint": checkpoint})

    for split in ("train", "validation"):
        for name, value in selected[split].items():
            metric(split, name, value)
    metric("selected_epoch", "generalization_gap_cross_entropy", selected["generalization_gap_cross_entropy"])
    metric("selected_epoch", "mean_gradient_norm_before_clipping", selected["mean_gradient_norm_before_clipping"])
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [row for row in events if row.get("kind") == "step"]
    if steps:
        metric("logged_steps", "maximum_gradient_norm_before_clipping", max(r["gradient_norm_before_clipping"] for r in steps))
        metric("logged_steps", "loss_increases_over_0.5_from_previous_logged_step",
               sum(b["loss"] > a["loss"] + 0.5 for a, b in zip(steps, steps[1:])))
    metric("training", "detected_nonfinite_failures", 0)
    for name in ("parameter_count", "training_targets_per_second", "total_training_seconds",
                 "elapsed_seconds_including_prior_sessions", "generation_tokens_per_second",
                 "cuda_peak_allocated_mb", "host_peak_rss_mb"):
        metric("run", name, summary[name])
    for decoding, values in summary["generation_metrics"].items():
        for name, value in values.items():
            if name != "tokenization":
                metric(decoding, name, value)
    with (member / "metrics_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    destinations = {}
    for name in ("best.pt", "last.pt"):
        source = run / "checkpoints" / name
        destination = member / "checkpoints" / name
        shutil.copy2(source, destination)
        checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
        if checksum != hashlib.sha256(source.read_bytes()).hexdigest():
            raise RuntimeError("Checkpoint copy checksum mismatch")
        destinations[name] = {"path": destination.relative_to(ROOT).as_posix(), "sha256": checksum,
                              "bytes": destination.stat().st_size}
    (member / "checkpoints/manifest.json").write_text(json.dumps(destinations, indent=2) + "\n")
    outputs = member / "outputs/full"
    outputs.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "history.json", "generations.json", "vocabulary.json"):
        shutil.copy2(run / name, outputs / name)
    shutil.copytree(run / "figures", outputs / "figures", dirs_exist_ok=True)
    (member / "results.md").write_text(
        "# Srinidhi - Character GPT results\n\n"
        f"Completed {summary['completed_full_epochs']} full epochs on {summary['device_name']} "
        f"using {summary['train_stories']:,} training and {summary['validation_stories']:,} validation stories. "
        "These are cloud GPU results; they do not claim a college-lab run.\n\n"
        f"The model has {config['layers']} blocks, {config['heads']} heads, width {config['embedding_dim']}, "
        f"context {config['context_length']}, dropout {config['dropout']}, batch {config['batch_size']}, "
        f"and {summary['parameter_count']:,} parameters. AdamW uses learning rate {config['learning_rate']}, "
        "5% warm-up and cosine decay. See README.md for the design and batch-size benchmark.\n\n"
        f"Best validation checkpoint: epoch {summary['best_checkpoint_epoch']}. "
        f"Validation cross-entropy: {summary['best_validation']['cross_entropy']:.6f}; "
        f"perplexity: {summary['best_validation']['perplexity']:.6f}; "
        f"next-character accuracy: {summary['best_validation']['next_character_accuracy']:.4%}.\n\n"
        "All metrics are in metrics_report.csv. Character metrics include EOS targets. "
        "The generalization gap compares validation eval-mode loss with online training-mode loss "
        "in the selected epoch, rather than reevaluating the training split. Generation speed counts "
        "emitted characters, excluding the prompt and EOS. The loss-spike count uses successive logged "
        "steps and a 0.5 cross-entropy increase; it is a diagnostic, not proof of instability. "
        "The trainer stops on nonfinite losses or gradients.\n\n"
        f"Raw, unedited run evidence: ../../{relative}. Checkpoint checksums: checkpoints/manifest.json. "
        "The executed notebook displays the recorded GPU run; rendering it locally does not retrain the model. "
        "Failure analysis and interpretation should be reviewed by the student before submission.\n")

    cells = [
        nbformat.v4.new_markdown_cell("# Character GPT - completed GPU run\n\n"
            "Srinidhi, Lab Pair 49. This executed notebook displays preserved training outputs and verifies "
            "the saved checkpoint. Training ran on the recorded RunPod GPU; notebook rendering can run locally. "
            "The accompanying gpt.py contains the from-scratch model and training implementation."),
        nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
''' + f'RUN = ROOT / {relative!r}\n' + '''summary = json.loads((RUN / "summary.json").read_text())
config = json.loads((RUN / "config.json").read_text())
assert summary["is_final_training_run"]
display({"training_gpu": summary["device_name"], "torch": summary["torch_version"],
         "cuda": summary["cuda_version"], "epochs": summary["completed_full_epochs"],
         "train_stories": summary["train_stories"], "validation_stories": summary["validation_stories"]})
display(config)'''),
        nbformat.v4.new_markdown_cell("## Measured metrics\n\nMetrics come from the selected validation checkpoint. "
            "Timing and memory describe the whole run. See results.md for metric definitions and limitations."),
        nbformat.v4.new_code_cell('display(pd.read_csv(ROOT / "task1_llm/srinidhi/metrics_report.csv")[["split", "metric", "value"]])'),
        nbformat.v4.new_code_cell('''for figure in sorted((RUN / "figures").glob("*.png")):
    print(figure.name)
    display(Image(filename=str(figure), width=850))'''),
        nbformat.v4.new_markdown_cell("## Actual generated stories"),
        nbformat.v4.new_code_cell('''generations = json.loads((RUN / "generations.json").read_text())
for index, item in enumerate(generations):
    print(f"{index}: {item['decoding']} | {item['generation_tokens_per_second']:.1f} characters/sec")
    print(item["full_text"])
    print()'''),
        nbformat.v4.new_markdown_cell((member / "failure_analysis.md").read_text()),
        nbformat.v4.new_markdown_cell("## Checkpoint and raw evidence"),
        nbformat.v4.new_code_cell('''import hashlib
import torch
folder = ROOT / "task1_llm/srinidhi/checkpoints"
manifest = json.loads((folder / "manifest.json").read_text())
for name, info in manifest.items():
    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == info["sha256"]
saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
assert all(torch.isfinite(value).all() for value in saved["model"].values())
print("Both checkpoint checksums verified; best model tensors are finite.")
print("Recorded same-device GPU inference reload:", summary["inference_reload_verified"])
print("Raw evidence:", RUN.relative_to(ROOT))
print((RUN / "RUN_LOG.txt").read_text()[-4000:])'''),
        nbformat.v4.new_markdown_cell("## Reload and run the saved model locally\n\n"
            "This fresh CPU inference is a portability check. It is separate from the recorded GPU timing metrics."),
        nbformat.v4.new_code_cell('''from lab1.gpt import CharacterGPT, generate
torch.set_num_threads(2)
model = CharacterGPT(len(saved["vocabulary"]["tokens"]), saved["config"])
model.load_state_dict(saved["model"], strict=True)
model.eval()
assert sum(p.numel() for p in model.parameters()) == summary["parameter_count"]
print(model)
prompt = "Once upon a time, a little bird"
continuation = generate(model, saved["vocabulary"], prompt, 120, torch.device("cpu"),
                        seed=2342, temperature=0)
print("Fresh CPU checkpoint inference:")
print(prompt + continuation)'''),
        nbformat.v4.new_markdown_cell("## Reproduce\n\nRun from the repository root after setup and preparing the frozen data:\n\n"
            "```bash\n.venv/bin/python -m lab1.run --task gpt --mode full --device cuda --set offline=true\n```\n\n"
            "This starts a new full training run. The one-command CPU smoke test is documented in the root README. "
            "See failure_analysis.md for observed errors and review the explanation before the viva."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.write(notebook, member / "src/gpt.ipynb")
    print(json.dumps({"run": relative, "metric_rows": len(rows), "checkpoints": destinations}, indent=2))


if __name__ == "__main__":
    main()
