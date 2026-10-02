"""Verify, execute, and package only completed Part 1 artifacts; never train a model."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from tempfile import TemporaryDirectory
import time
import zipfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

import publish_gpt_results as publisher

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "Part 1"
MANIFEST = "PACKAGE_MANIFEST.json"
RUNTIME_PACKAGES = ("torch", "numpy", "pandas", "matplotlib", "datasets", "nbformat", "nbclient", "ipykernel")


def file_metadata(path):
    return {"bytes": Path(path).stat().st_size, "sha256": publisher.digest(path)}


def raw_inventory(run):
    """The original console is included in local integrity verification only."""
    return {path.relative_to(run).as_posix(): file_metadata(path)
            for path in sorted(Path(run).rglob("*")) if path.is_file()}


def execute_notebook(root):
    root = Path(root).resolve()
    path = root / "task1_llm/srinidhi/src/gpt.ipynb"
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    os.environ["MPLBACKEND"] = "Agg"
    # Use this Python interpreter rather than a stale user kernelspec.
    # The stable repository cwd avoids Windows directory-rename kernel locks.
    with TemporaryDirectory(prefix="part1-kernel-") as temporary:
        kernel = Path(temporary) / "python3"
        kernel.mkdir()
        publisher.write_json(kernel / "kernel.json", {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Part 1 verification", "language": "python"})
        manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[temporary], ensure_native_kernel=False))
        client = NotebookClient(notebook, km=manager, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(root)}})
        try:
            client.execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook, path)
    record = verify_notebook(path)
    record.update(seconds=time.perf_counter() - started, notebook=path.relative_to(root).as_posix(),
                  mode="completed_full_run_evidence", sha256=publisher.digest(path),
                  interpreter_version=sys.version.split()[0])
    publisher.write_json(root / "verification/part1_notebook.json", record)
    return record


def verify_notebook(path):
    notebook = path if isinstance(path, nbformat.NotebookNode) else nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    publisher.require(code and all(cell.execution_count is not None and cell.outputs for cell in code),
                      "Every notebook code cell must be executed with visible output")
    errors = [output for cell in code for output in cell.outputs if output.output_type == "error"]
    publisher.require(not errors, "Notebook contains execution errors")
    publisher.require(notebook.metadata.get("lab1", {}).get("mode") == "completed_full_run_evidence",
                      "Notebook does not identify completed full-run evidence")
    return {"executed_code_cells": len(code), "visible_output_cells": len(code), "errors": 0,
            "all_outputs_visible": True}


def verify_published(root, evidence):
    root = Path(root)
    member = root / "task1_llm/srinidhi"
    manifest = publisher.read_json(member / "checkpoints/manifest.json")
    for name in ("best.pt", "last.pt"):
        original = root / evidence["run"] / "checkpoints" / name
        copied = member / "checkpoints" / name
        publisher.require(file_metadata(copied) == {key: manifest[name][key] for key in ("bytes", "sha256")}
                          and publisher.digest(original) == publisher.digest(copied),
                          f"Published {name} differs from original")
    with (member / "metrics_report.csv").open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    expected = publisher.metric_rows(evidence)
    publisher.require(len(actual) == len(expected), "Published metric count mismatch")
    for recorded, measured in zip(actual, expected):
        publisher.require(all(recorded[key] == ("" if value is None else str(value))
                              for key, value in measured.items()), "Published metric CSV differs from raw run")
    for name in ("summary.json", "history.json", "generations.json", "vocabulary.json", "config.json", "data_manifest.json"):
        publisher.require(publisher.digest(member / "outputs/full" / name)
                          == publisher.digest(root / evidence["run"] / name), f"Published {name} differs from raw evidence")
    publisher.require(publisher.read_json(member / "outputs/full/reproduction_config.json") ==
                      {"full": evidence["config"]},
                      "Published reproduction recipe differs from the selected run")
    raw_figures = {path.name: publisher.digest(path) for path in (root / evidence["run"] / "figures").glob("*.png")}
    copied_figures = {path.name: publisher.digest(path) for path in (member / "outputs/full/figures").glob("*.png")}
    publisher.require({"learning_curves.png", "training_diagnostics.png"} <= raw_figures.keys()
                      and raw_figures == copied_figures, "Published figures differ from raw evidence or required curves are missing")
    failures = publisher.verify_failure_analysis(member / "failure_analysis.md", evidence["generations"])
    inference = publisher.read_json(root / "verification/part1_checkpoint_inference.json")
    publisher.require(inference["checkpoint_sha256"] == manifest["best.pt"]["sha256"],
                      "Notebook inference checked a different checkpoint")
    publisher.require(any(item["device"] == "cpu" for item in inference["devices"])
                      and all(item["reload_logits_identical"] and item["causal_mask_verified"] and item["finite_logits"]
                              for item in inference["devices"]), "Fresh checkpoint inference verification failed")
    if publisher.torch.cuda.is_available():
        publisher.require(any(item["device"] == "cuda" for item in inference["devices"]),
                          "CUDA was available but notebook did not check CUDA inference")
    return {"metrics_match_raw_evidence": True, "checkpoints_match_raw_evidence": True,
            "failure_excerpts": failures, "fresh_inference": inference,
            "notebook": verify_notebook(member / "src/gpt.ipynb")}


def selected_experiment_files(root, evidence):
    """Include only the selected run's frozen recipe and matching comparison."""
    root = Path(root).resolve()
    selected = []
    experiments = root / "task1_llm/srinidhi/experiments"
    for plan_path in sorted(experiments.glob("*/plan.json")):
        plan = publisher.read_json(plan_path)
        if plan.get("run") != evidence["run"]:
            continue
        relative = PurePosixPath(plan["config"])
        publisher.require(not relative.is_absolute() and ".." not in relative.parts
                          and "\\" not in str(relative), "Experiment config path is not portable")
        config_path = root / str(relative)
        publisher.require(config_path.is_file() and not config_path.is_symlink()
                          and config_path.resolve().is_relative_to(experiments),
                          "Selected experiment config is missing or outside the experiment directory")
        publisher.require(publisher.digest(config_path) == plan["config_sha256"],
                          "Selected experiment config differs from its frozen plan")
        publisher.require(publisher.read_json(config_path).get("full") == evidence["config"],
                          "Selected experiment recipe differs from the actual training configuration")
        selected.extend((plan_path, config_path))
    comparison_path = root / "verification/part1_quality_comparison.json"
    if comparison_path.is_file():
        comparison = publisher.read_json(comparison_path)
        # An unrelated comparison must not prevent the original baseline from
        # being packaged again, and must not be attached to that baseline.
        if comparison.get("promotion_rule", {}).get("preferred_run") == evidence["run"]:
            rows = [row for row in comparison.get("candidates", []) if row.get("run") == evidence["run"]]
            publisher.require(len(rows) == 1 and rows[0].get("checkpoint_sha256") ==
                              publisher.digest(root / evidence["run"] / "checkpoints/best.pt"),
                              "Selected comparison refers to a different checkpoint")
            publisher.require(comparison.get("protocol", {}).get("manifest_sha256") ==
                              evidence["summary"]["manifest_sha256"],
                              "Selected comparison refers to a different frozen dataset")
            selected.append(comparison_path)
    return selected


def package_files(root, evidence):
    root = Path(root).resolve()
    files = {}
    def add(path):
        path = Path(path)
        if not path.is_file():
            return
        publisher.require(not path.is_symlink() and path.resolve().is_relative_to(root),
                          f"Package file is a symlink or outside repository: {path.name}")
        files[path.relative_to(root).as_posix()] = path
    for name in ("pyproject.toml", "PART1_FINALIZATION.md", "AI_USE.md", ".gitattributes", ".gitignore"):
        add(root / name)
    for path in sorted((root / "src/lab1").glob("*.py")):
        add(path)
    member = root / "task1_llm/srinidhi"
    for path in sorted(member.glob("*")):
        add(path)
    for pattern in ("src/*.py", "src/*.ipynb", "outputs/full/*", "outputs/full/figures/*.png"):
        for path in sorted(member.glob(pattern)):
            add(path)
    for relative in ("checkpoints/best.pt", "checkpoints/last.pt", "checkpoints/manifest.json",
                     "checkpoints/README.md", "outputs/README.md", "data_processed/README.md"):
        add(member / relative)
    for path in selected_experiment_files(root, evidence):
        add(path)
    add(root / "task1_llm/README.md")
    add(root / "task1_llm/data/README.md")
    for name in ("publish_gpt_results.py", "finalize_gpt.py", "compare_gpt_candidates.py"):
        add(root / "scripts" / name)
    for name in ("test_gpt.py", "test_gpt_finalization.py", "test_compare_gpt_candidates.py"):
        add(root / "tests" / name)
    for path in evidence["cache_paths"].values():
        add(path)
    for path in sorted((root / evidence["run"]).rglob("*")):
        if path.is_file() and path.name != "RUN_LOG.txt":
            add(path)
    for name in ("part1_desktop_environment.json", "part1_export.json", "part1_notebook.json",
                 "part1_checkpoint_inference.json", "part1_finalization.json", "part1_preflight_pytest.xml",
                 "part1_finalization_pytest.xml", "part1_desktop_preflight.xml", "part1_desktop_smoke.json",
                 "part1_package_portability.json"):
        add(root / "verification" / name)
    publisher.require("PART1_FINALIZATION.md" in files, "Standalone PART1_FINALIZATION.md setup guide is missing")
    # Portable install pins omit the CUDA local version suffix; the guide gives
    # the tested CUDA-wheel index. Full runtime versions remain in the receipt.
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    requirements = "# Part 1 direct dependencies; see README.md for CUDA wheel installation.\n" + "".join(
        f"{name}=={version.split('+')[0]}\n" for name, version in versions.items())
    files["requirements.txt"] = requirements.encode("utf-8")
    files["README.md"] = (root / "PART1_FINALIZATION.md").read_bytes()
    # Part 1 needs none of the image-classification/image-metric dependencies.
    dependencies = ", ".join(json.dumps(f"{name}=={version.split('+')[0]}") for name, version in versions.items())
    files["pyproject.toml"] = ("[build-system]\nrequires = [\"setuptools>=77,<82\"]\n"
        "build-backend = \"setuptools.build_meta\"\n\n[project]\n"
        "name = \"data266-part1-2342\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n"
        f"dependencies = [{dependencies}]\n\n[project.optional-dependencies]\ntest = [\"pytest>=8\"]\n\n"
        "[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.pytest.ini_options]\n"
        "pythonpath = [\"src\"]\ntestpaths = [\"tests\"]\n").encode("utf-8")
    return files, versions


def entry_metadata(value):
    if isinstance(value, Path):
        return file_metadata(value)
    return {"bytes": len(value), "sha256": publisher.hashlib.sha256(value).hexdigest()}


def verify_archive(path):
    """Validate safe canonical names, complete inventory, CRC and SHA256 hashes."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        publisher.require(len(names) == len(set(names)), "Duplicate package ZIP paths")
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            publisher.require(not relative.is_absolute() and ".." not in relative.parts
                              and "\\" not in item.filename and str(relative) == item.filename
                              and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16),
                              "Unsafe or noncanonical package ZIP path")
        publisher.require(archive.testzip() is None, "Package ZIP CRC verification failed")
        manifest_path = f"{ARCHIVE_ROOT}/{MANIFEST}"
        manifest = json.loads(archive.read(manifest_path))
        publisher.require(manifest.get("format_version") == 1 and manifest.get("scope") == "Srinidhi Part 1 only",
                          "Unsupported Part 1 package manifest")
        files = manifest["files"]
        publisher.require(set(names) == {manifest_path, *(f"{ARCHIVE_ROOT}/{name}" for name in files)},
                          "Package ZIP inventory differs from manifest")
        for relative, expected in files.items():
            info = archive.getinfo(f"{ARCHIVE_ROOT}/{relative}")
            value = publisher.hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(chunk)
            publisher.require(info.file_size == expected["bytes"] and value.hexdigest() == expected["sha256"],
                              f"Package ZIP size/SHA256 mismatch: {relative}")
        required = {"pyproject.toml", "src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                    "task1_llm/srinidhi/src/gpt.py", "task1_llm/srinidhi/src/gpt.ipynb",
                    "task1_llm/srinidhi/checkpoints/best.pt", "task1_llm/srinidhi/checkpoints/last.pt",
                    "task1_llm/srinidhi/metrics_report.csv", "task1_llm/srinidhi/failure_analysis.md",
                    "scripts/finalize_gpt.py", "scripts/publish_gpt_results.py", "README.md", "requirements.txt"}
        publisher.require(required <= files.keys(), "Package lacks required Part 1 runtime/artifacts")
        raw_required = {f"{manifest['source_run']}/{name}" for name in (
            "metrics.jsonl", "summary.json", "history.json", "run_summary.json", "config.json",
            "resolved_config.json", "data_manifest.json", "vocabulary.json", "generations.json")}
        publisher.require(raw_required <= files.keys(), "Package lacks required raw training evidence")
        notebook = nbformat.reads(archive.read(f"{ARCHIVE_ROOT}/task1_llm/srinidhi/src/gpt.ipynb").decode("utf-8"), as_version=4)
        verify_notebook(notebook)
        publisher.require(all(not name.startswith(("task2_", "task3_", "report/")) for name in files),
                          "Part 1 package unexpectedly contains other parts or combined report")
        publisher.require(any(name.endswith("/train.jsonl") and "data_processed/full/" in name for name in files)
                          and any(name.endswith("/validation.jsonl") and "data_processed/full/" in name for name in files),
                          "Package lacks frozen processed training/validation stories")
    return {"files": len(files) + 1, "crc_and_content_hashes_verified": True,
            "contains_trained_final_weights": True, "contains_executed_notebook": True,
            "contains_frozen_processed_split": True}


def create_archive(root, evidence, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    publisher.require(not destination.is_relative_to(root / evidence["run"]),
                      "Archive destination cannot be inside immutable raw run")
    files, versions = package_files(root, evidence)
    metadata = {name: entry_metadata(value) for name, value in sorted(files.items())}
    manifest = {"format_version": 1, "scope": "Srinidhi Part 1 only", "source_run": evidence["run"],
                "completed_full_epochs": evidence["summary"]["completed_full_epochs"],
                "best_checkpoint_epoch": evidence["summary"]["best_checkpoint_epoch"], "runtime_packages": versions,
                "console_evidence": "Original RUN_LOG.txt retained locally unchanged; omitted because startup warnings contain host paths. Untouched metrics.jsonl and runner provenance are packaged.",
                "files": metadata}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted(files.items()):
                if isinstance(value, Path):
                    publisher.require(file_metadata(value) == metadata[name], f"File changed during packaging: {name}")
                    archive.write(value, f"{ARCHIVE_ROOT}/{name}")
                    publisher.require(file_metadata(value) == metadata[name], f"File changed during packaging: {name}")
                else:
                    archive.writestr(f"{ARCHIVE_ROOT}/{name}", value)
            archive.writestr(f"{ARCHIVE_ROOT}/{MANIFEST}", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        result = verify_archive(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(archive=destination.relative_to(root).as_posix() if destination.is_relative_to(root) else destination.name,
                  **file_metadata(destination), scope="Srinidhi Part 1 only; not the combined Canvas submission")
    publisher.write_json(root / "dist/part1_package_verification.json", result)
    (root / "dist/Part1_SHA256SUMS.txt").write_text(result["sha256"] + "  " + destination.name + "\n", encoding="utf-8")
    return result


def finalize(root, run, destination=None):
    root, run = Path(root).resolve(), Path(run).resolve()
    publisher.relative_run(root, run)
    before = raw_inventory(run)
    publisher.publish(root, run)
    notebook = execute_notebook(root)
    evidence = publisher.verify_training_evidence(root, run)
    published = verify_published(root, evidence)
    publisher.require(raw_inventory(run) == before, "Raw training evidence changed during finalization")
    receipt = {"source_run": evidence["run"], "completed_full_epochs": evidence["summary"]["completed_full_epochs"],
               "best_checkpoint_epoch": evidence["summary"]["best_checkpoint_epoch"],
               "raw_training_evidence_unchanged": True, "original_raw_files": before,
               "notebook": notebook, "published_checks": published,
               "scope": "Srinidhi Part 1 only; team submission remains separate"}
    publisher.write_json(root / "verification/part1_finalization.json", receipt)
    result = create_archive(root, evidence, destination or root / "dist/Part1_Srinidhi_2342.zip")
    publisher.require(raw_inventory(run) == before, "Raw training evidence changed during packaging")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--archive", type=Path, help="Destination archive (default dist/Part1_Srinidhi_2342.zip)")
    parser.add_argument("--verify-archive", type=Path, help="Verify an existing archive without writing/exporting")
    args = parser.parse_args()
    if args.verify_archive:
        print(json.dumps(verify_archive(args.verify_archive), indent=2))
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required unless --verify-archive is given")
        print(json.dumps(finalize(ROOT, args.run_dir, args.archive), indent=2))


if __name__ == "__main__":
    main()
