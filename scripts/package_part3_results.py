"""Package the completed October 4 CycleGAN results and recovery checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "runs/part3-push-work/results/Vast_54065624_full_backup_20261004/Part3_Push_5090"
DESTINATION = ROOT / "reproducibility/packages/part3-20261004"
BEST_SHA = "62d80f7ef2752b190fa496b7775b32dd77684bfceeb3b9f4c7356642848abc14"
EVALUATOR_SHA = "702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef"
ARMS = ("b1_c5", "b8_c5", "b8_c5_r1")
SOURCE_FILES = (
    "run_all.py",
    "scripts/finetune_part3_push.py",
    "src/lab1/__init__.py",
    "src/lab1/common.py",
    "task3_gan/srinidhi/src/cyclegan.py",
    "task3_gan/srinidhi/src/cyclegan_ema.py",
    "task3_gan/srinidhi/src/cyclegan_push.py",
    "tests/test_cyclegan_push.py",
    "requirements.txt",
    "pytest.ini",
    "arms.json",
)
PERSONAL_PATH = re.compile(rb"/Users/[^/\s]+/|[A-Z]:\\Users\\")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n")


def copy_file(source: Path, target: Path, records: list, portable: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    original = digest(source)
    if portable:
        text = source.read_text().replace("/workspace/Part3_Push_5090/", "")
        target.write_text(text)
    else:
        shutil.copyfile(source, target)
    if target.suffix in {".json", ".jsonl", ".py", ".md", ".txt", ".ipynb", ".log", ".out"}:
        if PERSONAL_PATH.search(target.read_bytes()):
            raise ValueError(f"Personal path in publication: {target.name}")
    records.append(
        {
            "source": source.relative_to(BACKUP).as_posix(),
            "publication": target.as_posix(),
            "source_sha256": original,
            "sha256": digest(target),
            "derived": original != digest(target),
        }
    )


def tensor_fingerprint(value) -> dict:
    import numpy as np
    import torch

    hashes = {}

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().view(torch.uint8).numpy()
            header = f"{item.dtype}:{tuple(item.shape)}:".encode()
            hashes[path] = hashlib.sha256(header + array.tobytes()).hexdigest()
        elif isinstance(item, np.ndarray):
            hashes[path] = hashlib.sha256(str(item.dtype).encode() + item.tobytes()).hexdigest()
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}/{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(value, "")
    return hashes


def portable_init(source: Path, target: Path, records: list) -> dict:
    import torch

    saved = torch.load(source, map_location="cpu", weights_only=False)
    before = tensor_fingerprint(saved)
    path = saved["state"]["init_checkpoint"]
    if "/Users/" not in path:
        raise ValueError("Expected the original warm-start provenance path")
    saved["state"]["init_checkpoint"] = "historical_checkpoints/" + Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, target)
    reopened = torch.load(target, map_location="cpu", weights_only=False)
    if tensor_fingerprint(reopened) != before:
        raise ValueError("Checkpoint tensor fingerprint changed")
    with zipfile.ZipFile(target) as archive:
        for name in archive.namelist():
            if name.endswith("data.pkl") and PERSONAL_PATH.search(archive.read(name)):
                raise ValueError("Checkpoint still contains a personal path")
    receipt = {
        "source_sha256": digest(source),
        "sha256": digest(target),
        "metadata_changes": ["state.init_checkpoint: historical filename only"],
        "tensor_count": len(before),
        "every_tensor_unchanged": True,
        "tensor_manifest_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest(),
    }
    records.append(
        {"source": "inputs/init.pt", "publication": target.as_posix(), "derived": True, **receipt}
    )
    return receipt


def copy_sources(destination: Path, records: list) -> None:
    for name in SOURCE_FILES:
        copy_file(BACKUP / name, destination / name, records)


def metrics_rows(results: dict, summary: dict, parameters: dict, training: list) -> list:
    rows = []

    def add(scope, direction, metric, value, status="computed", note=""):
        rows.append([scope, direction, metric, value, status, note])

    for metric, value in results["official_notebook"].items():
        add("unchanged_supplied_notebook", "directional_mean", metric, value)
    for direction, values in results["export_metrics"]["directions"].items():
        for metric in ("FID", "MiFID"):
            add(
                "export_scorer",
                direction,
                metric,
                values[metric],
                note="Directional companion scorer; official mean is above.",
            )
    checks = results["checks"]
    for direction, suffix in (("monet_to_photo", "a2b"), ("photo_to_monet", "b2a")):
        add(
            "diagnostic",
            direction,
            "KID",
            checks["class_protocol"][f"kid_{suffix}"],
            note="Class images also occurred in training; not a held-out estimate.",
        )
        add(
            "diagnostic",
            direction,
            "content_cosine",
            checks["content_cosine_input_vs_output"][direction],
        )
        for metric in (
            "generative_precision",
            "generative_recall",
            "generative_density",
            "generative_coverage",
            "LPIPS",
            "cycle_reconstruction_L1",
        ):
            add(
                "selected_checkpoint",
                direction,
                metric,
                "",
                "not_measured",
                "Not measured for this checkpoint; older model metrics do not apply.",
            )
        for metric in ("human_style", "human_content", "human_artifacts", "inter_rater_agreement"):
            add(
                "human_audit",
                direction,
                metric,
                "",
                "pending",
                "Requires new blinded ratings for this checkpoint.",
            )
    for metric in (
        "generator_total",
        "discriminator_photo",
        "discriminator_monet",
        "cycle_photo_l1",
        "cycle_monet_l1",
    ):
        values = [row[metric] for row in training]
        add(
            "training",
            "b1_c5",
            metric + "_logged_mean",
            sum(values) / len(values),
            note="Mean over logged updates, not all updates or selected-checkpoint evaluation.",
        )
        add("training", "b1_c5", metric + "_final_logged", values[-1])
    add(
        "training",
        "b1_c5",
        "identity_loss_weight",
        0,
        note="Identity loss disabled; no measured identity reconstruction loss.",
    )
    add("training", "b1_c5", "nan_events", summary["nan_events"])
    add("training", "b1_c5", "completed_updates", summary["completed_updates"])
    add(
        "training",
        "b1_c5",
        "training_seconds",
        summary["training_seconds"],
        note="Trainer-timed update work; excludes evaluation and setup.",
    )
    add(
        "training",
        "b1_c5",
        "images_per_second",
        2 * summary["completed_updates"] / summary["training_seconds"],
        note="Two domain images per batch-1 update; timed update work, not end-to-end throughput.",
    )
    for name, count in parameters.items():
        add("architecture", name, "parameter_count", count)
    for metric in ("gradient_norms", "peak_gpu_memory_bytes", "peak_host_memory_bytes"):
        add(
            "training",
            "b1_c5",
            metric,
            "",
            "not_recorded",
            "No verified per-arm measurement; shared GPU snapshots are not this metric.",
        )
    for metric in ("public_score", "private_score", "leaderboard_rank"):
        add(
            "kaggle",
            "team",
            metric,
            "",
            "not_submitted",
            "Local supplied-notebook evaluation only.",
        )
    return rows


def make_notebook(package: Path) -> Path:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    member = "task3_gan/srinidhi"
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Part 3: CycleGAN results\n\nThe final supplied-notebook score is **47.5604**. This notebook reads the recorded results and runs one CPU translation in each direction. It does not train or rescore FID. Batch-1 completed; the batch-8 and R1 trials were interrupted by host memory exhaustion. No Kaggle submission or current human audit is claimed."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport hashlib\nimport json\nimport sys\nimport torch\nfrom PIL import Image\nfrom IPython.display import display\n\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'PACKAGE.json').exists())\nmember = root / 'task3_gan/srinidhi'\nsys.path.insert(0, str(root / 'src'))\nfrom lab1 import cyclegan as cg\nfrom lab1 import cyclegan_push as cp\n\ntorch.set_num_threads(4)\ncheckpoint = member / 'checkpoints/best.pt'\nwith checkpoint.open('rb') as stream:\n    assert hashlib.file_digest(stream, 'sha256').hexdigest() == '"
            + BEST_SHA
            + "'\nsaved = torch.load(checkpoint, map_location='cpu', weights_only=False)\nprint('Verified selected checkpoint; device: CPU')"
        ),
        nbformat.v4.new_code_cell(
            "results = json.loads((member / 'outputs/RESULTS.json').read_text())\nprint(json.dumps(results['official_notebook'], indent=2))\nprint('Selection: batch-1, slow EMA, update 221250; 192 scored candidates.')\nprint('Score uses class images seen during training, not a held-out test.')"
        ),
        nbformat.v4.new_code_cell(
            "examples = json.loads((member / 'data_processed/examples/manifest.json').read_text())\nwith torch.inference_mode():\n    for direction, example in examples.items():\n        source = member / 'data_processed/examples' / example['file']\n        model = cg.Generator(saved['config'].get('base_channels', 64), saved['config'].get('residual_blocks', 9))\n        model.load_state_dict(saved['models']['G_' + direction], strict=True)\n        model.eval()\n        output = model(cp.load_image(source, 256, None).unsqueeze(0))[0].float()\n        assert torch.isfinite(output).all() and tuple(output.shape) == (3, 256, 256)\n        panel = Image.new('RGB', (512, 256))\n        panel.paste(Image.open(source).convert('RGB').resize((256, 256)), (0, 0))\n        panel.paste(cg._pil(output), (256, 0))\n        print(direction + ': original left, fresh CPU output right')\n        display(panel)\nprint('Two CPU translations passed. Published JPEGs remain unchanged.')"
        ),
        nbformat.v4.new_markdown_cell(
            "## Remaining work\n\nThe under-44 target was not met. Read `full_metrics_report.csv` for measured, missing and pending items. This model still needs blinded human ratings and agreement, missing full metrics, student review, the team comparison/report and official submission. `evaluation/Part3_Evaluation_Script.executed.ipynb` preserves the actual full scoring run; its matching `submission.csv` is authoritative."
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
        },
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    NotebookClient(
        notebook, timeout=180, km=manager, resources={"metadata": {"path": str(package)}}
    ).execute()
    errors = [
        out
        for cell in notebook.cells
        if cell.cell_type == "code"
        for out in cell.outputs
        if out.output_type == "error"
    ]
    if errors:
        raise ValueError("Results notebook execution failed")
    path = package / member / "src/cyclegan.ipynb"
    nbformat.write(notebook, path)
    return path


def plot_losses(package: Path, training: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for key in ("generator_total", "discriminator_photo", "discriminator_monet"):
        axes[0].plot(
            [r["step"] for r in training],
            [r[key] for r in training],
            label=key,
            linewidth=0.6,
            alpha=0.75,
        )
    for key in ("cycle_photo_l1", "cycle_monet_l1"):
        axes[1].plot(
            [r["step"] for r in training],
            [r[key] for r in training],
            label=key,
            linewidth=0.6,
            alpha=0.75,
        )
    for ax in axes:
        ax.set_xlabel("Update")
        ax.set_ylabel("Logged loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)
    figure.savefig(package / "task3_gan/srinidhi/outputs/training_curves.png", dpi=160)
    plt.close(figure)


def finish_manifest(folder: Path, records: list) -> dict:
    normalized = []
    for record in records:
        row = dict(record)
        row["publication"] = Path(row["publication"]).relative_to(folder).as_posix()
        normalized.append(row)
    files = {
        p.relative_to(folder).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(folder.rglob("*"))
        if publishable(p)
    }
    manifest = {
        "files": files,
        "source_mapping": normalized,
        "raw_logs": "Copied byte-for-byte; generic original container paths are historical provenance.",
        "source_code": "Exact run source; readable working repository source may contain documented post-run cleanup.",
    }
    write_json(folder / "MANIFEST.json", manifest)
    return manifest


def publishable(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and ".ipynb_checkpoints" not in path.parts
        and path.suffix not in {".pyc", ".pyo", ".tmp"}
    )


def archive_and_verify(folder: Path, output: Path) -> dict:
    print(f"Packing {output.name}", flush=True)
    with zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True
    ) as archive:
        for path in sorted(folder.rglob("*")):
            if publishable(path):
                archive.write(path, path.relative_to(folder.parent).as_posix())
    if output.stat().st_size >= 2_000_000_000:
        raise ValueError(f"Archive exceeds 2 GB: {output.name}")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failed")
        manifest = json.loads(archive.read(folder.name + "/MANIFEST.json"))
        for name, info in manifest["files"].items():
            with archive.open(folder.name + "/" + name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != info["sha256"]:
                raise ValueError(f"Archive member hash failed: {name}")
        entries = len(archive.namelist())
    return {
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest(output),
        "zip_crc": "passed",
        "all_member_hashes": "passed",
        "entries": entries,
    }


def build(destination: Path) -> None:
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    final = BACKUP / "work/final"
    if digest(final / "best.pt") != BEST_SHA:
        raise ValueError("Selected checkpoint differs from the verified final model")
    evaluator = BACKUP / "reproducibility/packages/part3-20261002/Part3_Evaluation_Script.ipynb"
    if digest(evaluator) != EVALUATOR_SHA:
        raise ValueError("Supplied evaluator differs")
    results = json.loads((final / "RESULTS.json").read_text())
    summary = json.loads((BACKUP / "work/arms/b1_c5/summary.json").read_text())
    if results["official_notebook"]["composite"] != 47.56042586442388:
        raise ValueError("Unexpected official score")
    staging = ROOT / "runs/part3-publication-20261004"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="package-", dir=staging) as temporary:
        package = Path(temporary) / "Part3"
        recovery = Path(temporary) / "Part3_Recovery"
        member = package / "task3_gan/srinidhi"
        records, recovery_records = [], []
        copy_sources(package, records)
        copy_sources(recovery, recovery_records)
        copy_file(final / "best.pt", member / "checkpoints/best.pt", records)
        for name in ("RESULTS.json", "check.json"):
            copy_file(final / name, member / "outputs" / name, records, portable=True)
        copy_file(evaluator, member / "evaluation/Part3_Evaluation_Script.ipynb", records)
        copy_file(
            final / "official/Part3_Evaluation_Script.executed.ipynb",
            member / "evaluation/Part3_Evaluation_Script.executed.ipynb",
            records,
        )
        copy_file(final / "official/submission.csv", member / "submission.csv", records)
        copy_file(
            BACKUP / "work/environment.json", package / "reproducibility/environment.json", records
        )
        copy_file(
            BACKUP / "work/b8_c5_r1",
            package / "reproducibility/resolved_schedule.json",
            records,
            portable=True,
        )
        copy_file(
            BACKUP / "work/run_all.log", package / "reproducibility/raw_logs/run_all.log", records
        )
        copy_file(
            BACKUP / "work/environment.json", recovery / "work/environment.json", recovery_records
        )
        copy_file(
            BACKUP / "work/b8_c5_r1",
            recovery / "work/resolved_schedule.json",
            recovery_records,
            portable=True,
        )
        copy_file(BACKUP / "work/run_all.log", recovery / "work/run_all.log", recovery_records)
        for label, expected in (("pred_A2B", 300), ("pred_B2A", 7038)):
            paths = sorted((final / "export" / label).glob("*.jpg"))
            if len(paths) != expected:
                raise ValueError(f"Wrong {label} count")
            for path in paths:
                copy_file(path, member / "outputs" / label / path.name, records)
            copy_file(
                final / "export" / f"{label}_export_manifest.json",
                member / "outputs" / f"{label}_export_manifest.json",
                records,
            )
        checkpoint_rows = []
        for arm in ARMS:
            saved = torch.load(
                BACKUP / f"work/arms/{arm}/last.pt", map_location="cpu", weights_only=False
            )
            checkpoint_rows.append(
                {
                    "arm": arm,
                    "last_step": saved["state"]["step"],
                    "scheduled_steps": saved["push_config"]["steps"],
                    "status": "completed" if arm == "b1_c5" else "interrupted_oom",
                    "nan_events": saved["state"]["nan_events"],
                }
            )
            write_json(package / "reproducibility/configs" / f"{arm}.json", saved["push_config"])
            write_json(recovery / "configs" / f"{arm}.json", saved["push_config"])
            del saved
            for filename in ("best.pt", "last.pt", "training_log.jsonl"):
                source = BACKUP / f"work/arms/{arm}/{filename}"
                copy_file(source, recovery / f"work/arms/{arm}/{filename}", recovery_records)
            copy_file(
                BACKUP / f"work/arms/{arm}/training_log.jsonl",
                package / f"reproducibility/raw_logs/{arm}/training_log.jsonl",
                records,
            )
            copy_file(
                BACKUP / f"work/arms/{arm}.out",
                package / f"reproducibility/raw_logs/{arm}/console.log",
                records,
            )
            copy_file(
                BACKUP / f"work/arms/{arm}.out", recovery / f"work/arms/{arm}.out", recovery_records
            )
        copy_file(
            BACKUP / "work/arms/b1_c5/summary.json",
            package / "reproducibility/b1_summary.json",
            records,
            portable=True,
        )
        copy_file(
            BACKUP / "work/arms/b1_c5/summary.json",
            recovery / "work/arms/b1_c5/summary.json",
            recovery_records,
            portable=True,
        )
        init_receipt = portable_init(
            BACKUP / "inputs/init.pt", recovery / "inputs/init.pt", recovery_records
        )
        examples = {}
        for domain, direction in (("monet_jpg", "monet_to_photo"), ("photo_jpg", "photo_to_monet")):
            source = sorted((BACKUP / "task3_gan/data" / domain).glob("*.jpg"))[0]
            name = domain + ".jpg"
            copy_file(source, member / "data_processed/examples" / name, records)
            examples[direction] = {
                "file": name,
                "original_filename": source.name,
                "source_sha256": digest(source),
            }
        write_json(member / "data_processed/examples/manifest.json", examples)
        saved = torch.load(final / "best.pt", map_location="cpu", weights_only=False)
        parameters = {
            name: sum(t.numel() for t in weights.values())
            for name, weights in saved["models"].items()
        }
        del saved
        log = BACKUP / "work/arms/b1_c5/training_log.jsonl"
        training = [
            row for line in log.read_text().splitlines() if "step" in (row := json.loads(line))
        ]
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["scope", "direction", "metric", "value", "status", "note"])
        writer.writerows(metrics_rows(results, summary, parameters, training))
        write_text(member / "full_metrics_report.csv", output.getvalue())
        shutil.copyfile(member / "full_metrics_report.csv", member / "metrics_report.csv")
        write_json(
            package / "PACKAGE.json",
            {
                "selected_checkpoint_sha256": BEST_SHA,
                "official_notebook": results["official_notebook"],
                "trials": checkpoint_rows,
                "kaggle_submitted": False,
                "current_model_human_audit": "pending",
            },
        )
        plot_losses(package, training)
        write_text(member / "results.md", RESULT_TEXT)
        write_text(member / "failure_analysis.md", FAILURE_TEXT)
        write_text(package / "README.md", PACKAGE_TEXT)
        write_text(
            member / "data_processed/README.md",
            "Two unchanged source examples support the CPU notebook. Full data are in the separate existing dataset.zip; this run trained on all 300 Monet and 7,038 photo images.",
        )
        write_text(recovery / "README.md", RECOVERY_TEXT)
        write_json(
            recovery / "RECOVERY.json",
            {
                "trials": checkpoint_rows,
                "warm_start_publication": init_receipt,
                "automatic_resume_tested": False,
                "no_training_launched": True,
            },
        )
        notebook = make_notebook(package)
        finish_manifest(package, records)
        finish_manifest(recovery, recovery_records)
        archives = [
            archive_and_verify(package, destination / "Part3.zip"),
            archive_and_verify(recovery, destination / "Recovery.zip"),
        ]
        for source, name in (
            (notebook, "Part3_Results.ipynb"),
            (member / "submission.csv", "submission.csv"),
            (
                member / "evaluation/Part3_Evaluation_Script.executed.ipynb",
                "Part3_Evaluation_Script.executed.ipynb",
            ),
        ):
            shutil.copyfile(source, destination / name)
        write_json(
            destination / "receipt.json",
            {
                "archives": archives,
                "selected_checkpoint_sha256": BEST_SHA,
                "official_evaluator_sha256": EVALUATOR_SHA,
                "official_notebook": results["official_notebook"],
                "image_counts": {"pred_A2B": 300, "pred_B2A": 7038},
                "results_notebook": {"cpu_inferences": 2, "executed_code_cells": 3, "errors": 0},
                "trials": checkpoint_rows,
                "warm_start_publication": init_receipt,
                "no_new_training": True,
                "no_kaggle_submission": True,
                "dataset": "../part3-20261002/dataset.zip",
                "dataset_sha256": digest(
                    ROOT / "reproducibility/packages/part3-20261002/dataset.zip"
                ),
                "original_backup_unchanged": True,
            },
        )
        write_text(destination / "README.md", LANDING_TEXT)
    print(json.dumps(archives, indent=2), flush=True)


RESULT_TEXT = """
# CycleGAN results — October 4, 2026

The unchanged supplied scoring notebook reports FID **94.71603584640337**, MiFID
**0.4048158824443817**, and composite **47.56042586442388**. The requested score
below 44 was not reached. This is local notebook evaluation, not a Kaggle rank.

The model has two nine-residual-block, width-64 generators and two PatchGAN
discriminators at 256 pixels. The selected batch-1 continuation used Adam with
LR 0.0001, betas (0.5, 0.999), cycle weight 5, identity weight 0, translation
augmentation for the Monet discriminator, BF16, 1,000 warm-up updates and linear
decay after half the scheduled updates. Generator EMA rates were 0.999 and 0.9999.
The retained checkpoint is the slower EMA at update 221250, not the final update.
Exact recipes and original training source accompany this package.

Batch-1 completed 370,000 updates on an RTX 5090. Batch-8 and batch-8 with R1 were
killed by host memory exhaustion; their recoverable states are at updates 55,000
and 53,000. They were not resumed. Three arms initially shared the GPU, so elapsed
time and utilization do not describe independent single-model benchmarks.

Selection compared 192 candidates using the class scorer on images also used
in training. It is not a held-out comparison and may be optimistically selected.
The full supplied notebook was run afterward on the unchanged exported JPEGs.
Its CSV is authoritative; the companion export scorer differs slightly numerically.
The diagnostic fresh-sample estimate is not an official score or unseen test.

The images are direct model JPEGs in alphabetical source order: A=Monet,
B=Photo; 300 A→B and 7,038 B→A. No images were edited, filtered or substituted.
Read full_metrics_report.csv for metric scope and missing entries. Earlier model
metrics and human ratings cannot be reused for this checkpoint. Student review,
the current human audit, remaining metrics, team report and submissions are pending.
"""

FAILURE_TEXT = """
# Failure analysis

This is a results-grounded draft for student review, not a completed human audit.

- The official score improved from the smoke run's 48.9991 to 47.5604, but missed
  the below-44 target. Smoke and final checks share class images, not an unseen test.
- Continuing batch-1 past its selected checkpoint did not improve the best saved
  score. Its final slow EMA interim score was 48.5550 versus 47.5443 at update
  221250. Checkpoint selection matters; the final training state is not the winner.
- Running three processes exhausted host RAM. Batch-8 and R1 failed with exit -9;
  finite logged losses do not mean these trials completed successfully.
- The fixed evaluation images were observed throughout selection. Lower local
  scores do not establish generalization. Future tuning needs a declared validation
  protocol and independent final evaluation.
- No checkpoint-specific human visual failure labels are asserted here. Inspect
  fixed blinded outputs, record two raters' scores and agreement, and explain
  actual content/style/artifact failures before final reporting.
"""

PACKAGE_TEXT = """
# Part 3 — selected CycleGAN result

Open `task3_gan/srinidhi/src/cyclegan.ipynb` for recorded results and two verified
CPU translations. This notebook does not start training. The authoritative final
scorer is `task3_gan/srinidhi/evaluation/Part3_Evaluation_Script.executed.ipynb`;
all its cells executed without error. Both the supplied evaluator and executed
copy are unchanged from the verified run. The authoritative CSV is
`task3_gan/srinidhi/submission.csv`. Score: **47.56042586442388**.

Use Python 3.12, install PyTorch 2.11.0 / torchvision 0.26.0 for your platform,
then `python -m pip install -r requirements.txt`. A Jupyter Python kernel is
included in those requirements. Run the notebook from this extracted package.
For an RTX 5090, use the PyTorch CUDA 12.8 wheels. CPU inference needs no GPU.

The original full dataset is separately preserved in the repository at
`reproducibility/packages/part3-20261002/dataset.zip` through Git LFS. Extract it
to a temporary folder and place its `monet_jpg` and `photo_jpg` folders under
this package's `task3_gan/data/` for complete rescoring or an authorized new run.
The original scoring notebook retains its class environment path assumptions;
inspect its path-setting cell when reproducing in another environment. Do not
change metric code or presented historical outputs.

Source files here are the exact version that produced the run, including known
historical limitations; cleaned working source in the Git repository is a later
readability revision. `MANIFEST.json` maps original files to published copies.
Raw logs remain byte-identical. Metadata copies use relative container paths;
selected weights, JPEGs and the evaluator were not modified.

Do not launch this historical `run_all.py` blindly or use its `--resume`: its
saved schedule filename is wrong. Recovery.zip preserves full arm checkpoints
and documents the constraints. This package is not a completed Canvas/team
submission: missing metrics, new human ratings, student review and team work
remain explicitly marked in the member results and metric table.
"""

RECOVERY_TEXT = """
# Part 3 recovery evidence

This archive preserves the exact six arm `best.pt`/`last.pt` files, raw logs,
matching original trainer/helpers, frozen recipes, schedule and the warm-start
model. It does not resume or rent compute. Restore the existing repository's
`reproducibility/packages/part3-20261002/dataset.zip` to `task3_gan/data/` first.

Batch-1 completed 370,000/370,000 updates. Batch-8 last.pt is at 55,000/85,000;
R1 last.pt is at 53,000/79,000. Both were OOM-killed. `best.pt` is an inference
candidate, while `last.pt` includes optimizer, EMA, replay and random state.
R1 was explicitly retired. Further training requires a new user decision.

Do not run the original `run_all.py --resume`: the schedule filename was
shadowed. Resume, if authorized, must use the direct trainer with the identical
saved push configuration and original data ordering. Check host RAM first and
do not repeat the three-process allocation that failed. The old deadline and
cloud supervisor are historical and are deliberately not included.

Keep a separate copy of every existing best.pt before any continuation: a last.pt
may contain an older best-score record, and the original trainer can overwrite
a stronger later best.pt. Batch-8's saved best is update 55,250, newer than its
55,000 recovery state. Compare authentic saved candidates before export.

The warm-start `inputs/init.pt` is a metadata-only portable copy: only its
historical personal-path string was replaced by a relative historical filename.
Every tensor was fingerprint-verified unchanged. RECOVERY.json records original
and published hashes. The six arm checkpoint files remain byte-identical; their
recorded initialization hash therefore refers to the original warm-start bytes.
The archived original trainer loads the warm-start architecture when resuming;
no checkpoint metadata or recipe has been silently rewritten to hide this change.
This recovery package has been hash/CRC checked, not resumed in a new GPU run.
"""

LANDING_TEXT = """
# Latest Part 3 — October 4, 2026

**Final supplied-notebook score: 47.56042586442388** (FID 94.71603584640337,
MiFID 0.4048158824443817). The below-44 target was not reached.

- [Part3.zip](Part3.zip): selected weights, all 7,338 unchanged predictions,
  original run source, executed results and scoring notebooks, metrics and logs.
- [Recovery.zip](Recovery.zip): six arm best/last checkpoints, portable warm-start,
  original matching code, settings and logs. Read its recovery caveats first.
- [Results notebook](Part3_Results.ipynb), [executed supplied scoring notebook](Part3_Evaluation_Script.executed.ipynb),
  [authoritative submission CSV](submission.csv), [verification receipt](receipt.json).
- [Existing dataset backup](../part3-20261002/dataset.zip), reused without duplication.

After cloning, run `git lfs install` and `git lfs pull`, then extract Part3.zip.
Open its `task3_gan/srinidhi/src/cyclegan.ipynb`; the standalone notebook displayed
here is for viewing and must be run with the extracted package's files.
The notebook completed two fresh CPU translations without errors. The full
unchanged evaluator already ran on the GPU server; packaging does not rescore it.
Both archives have verified CRCs and every member hash; see receipt.json.

The winner is batch-1 slow EMA at update 221250. Batch-1 completed; batch-8 and
R1 were interrupted by host OOM. All 192 scored candidates used class images
also seen during training, so this is not a held-out score. Historical packages
and original backups remain unchanged. Readable working code is a post-run
cleanup; the archives preserve exact source from the experiment.

This is the current individual result and recovery package, not a completed
team submission. Missing full metrics, a new blinded human audit, student
review, the remaining teammate work/report and official Kaggle/Canvas submission
are not claimed. No old checkpoint's human ratings or metrics were reused.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    build(parser.parse_args().output)
