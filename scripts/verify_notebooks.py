"""Execute portable notebooks locally; defaults remain synthetic CPU smoke checks."""
from pathlib import Path
import json
import os
import sys
import time
from tempfile import TemporaryDirectory
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLBACKEND"] = "Agg"
records = []
notebooks = sorted(ROOT.glob("task*/srinidhi/src/*.ipynb"))
expected = {"task1_llm/srinidhi/src/gpt.ipynb", "task2_sentiment/srinidhi/src/sentiment.ipynb", "task3_gan/srinidhi/src/cyclegan.ipynb"}
actual = {str(path.relative_to(ROOT)) for path in notebooks}
if actual != expected:
    raise RuntimeError(f"Notebook set is incomplete or unexpected: {sorted(actual)}. Run scripts/build_notebooks.py.")
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    # Ignore unrelated user-level kernels, which can point to deleted environments.
    with TemporaryDirectory(prefix="lab1-kernel-") as temporary:
        kernel_dir = Path(temporary) / "python3"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Lab 1 verification", "language": "python"}))
        manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False))
        client = NotebookClient(notebook, km=manager, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook, path)
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    record = {"notebook": str(path.relative_to(ROOT)), "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells), "errors": len(errors), "seconds": time.perf_counter()-started, "mode": "synthetic_cpu_smoke"}
    records.append(record)
    print(json.dumps(record), flush=True)
destination = ROOT / "verification" / "notebooks.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps({"python": sys.version, "notebooks": records}, indent=2)+"\n")
