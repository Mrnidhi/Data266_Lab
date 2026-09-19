"""Execute the published Part 2 notebook and verify fresh CPU checkpoint inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "task2_sentiment/srinidhi/src/sentiment.ipynb"
provenance = json.loads((ROOT / "task2_sentiment/srinidhi/outputs/full/run_summary.json").read_text())
assert provenance["status"] == "completed" and provenance["mode"] == "full"
notebook = nbformat.read(path, as_version=4)
os.environ["MPLBACKEND"] = "Agg"
started = time.perf_counter()
with TemporaryDirectory(prefix="lab1-part2-kernel-") as temporary:
    kernel = Path(temporary) / "python3"
    kernel.mkdir()
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Part 2 verification", "language": "python"}))
    manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[temporary], ensure_native_kernel=False))
    NotebookClient(notebook, km=manager, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute(cleanup_kc=True)
errors = [output for cell in notebook.cells if cell.cell_type == "code"
          for output in cell.get("outputs", []) if output.output_type == "error"]
assert not errors
nbformat.write(notebook, path)
receipt = {"notebook": path.relative_to(ROOT).as_posix(),
           "mode": "preserved_full_results_and_fresh_cpu_checkpoint_inference",
           "executed_code_cells": sum(cell.cell_type == "code" and cell.execution_count is not None
                                      for cell in notebook.cells),
           "errors": len(errors), "seconds": time.perf_counter() - started,
           "python": sys.version}
(ROOT / "verification/part_b_notebook.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
