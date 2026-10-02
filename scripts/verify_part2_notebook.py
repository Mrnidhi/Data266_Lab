"""Execute only the completed Part 2 notebook in a fresh portable kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def validate_notebook(path):
    notebook = path if isinstance(path, nbformat.NotebookNode) else nbformat.read(path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not cells or not all(cell.execution_count is not None and cell.outputs for cell in cells):
        raise ValueError("All Part 2 code cells must be executed with visible output")
    if any(output.output_type == "error" for cell in cells for output in cell.outputs):
        raise ValueError("Part 2 notebook contains execution errors")
    if notebook.metadata.get("lab1", {}).get("mode") != "completed_full_run_evidence":
        raise ValueError("Part 2 notebook must identify completed full-run evidence")
    return {"executed_code_cells": len(cells), "visible_output_cells": len(cells), "errors": 0,
            "all_outputs_visible": True}


def execute_notebook(root=ROOT, device="auto"):
    root = Path(root).resolve()
    path = root / "task2_sentiment/srinidhi/src/sentiment.ipynb"
    provenance = json.loads((root / "task2_sentiment/srinidhi/outputs/full/run_summary.json").read_text(encoding="utf-8"))
    if provenance["status"] != "completed" or provenance["mode"] != "full":
        raise ValueError("Only a completed full evaluation may be executed as Part 2 results")
    notebook = nbformat.read(path, as_version=4)
    os.environ["MPLBACKEND"] = "Agg"
    prior_device = os.environ.get("LAB1_PART2_INFERENCE")
    os.environ["LAB1_PART2_INFERENCE"] = device
    started = time.perf_counter()
    try:
        with TemporaryDirectory(prefix="lab1-part2-kernel-") as temporary:
            kernel = Path(temporary) / "python3"
            kernel.mkdir()
            (kernel / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Part 2 verification", "language": "python"}), encoding="utf-8")
            manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[temporary], ensure_native_kernel=False))
            try:
                NotebookClient(notebook, km=manager, timeout=600, kernel_name="python3",
                               resources={"metadata": {"path": str(root)}}).execute(cleanup_kc=True)
            finally:
                nbformat.write(notebook, path)
    finally:
        if prior_device is None:
            os.environ.pop("LAB1_PART2_INFERENCE", None)
        else:
            os.environ["LAB1_PART2_INFERENCE"] = prior_device
    receipt = {"notebook": path.relative_to(root).as_posix(),
               "mode": "preserved_full_results_and_fresh_checkpoint_inference", **validate_notebook(notebook),
               "seconds": time.perf_counter() - started, "python": sys.version.split()[0],
               "inference_device": device, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    destination = root / "verification/part2_notebook.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(execute_notebook(device=args.device), indent=2))


if __name__ == "__main__":
    main()
