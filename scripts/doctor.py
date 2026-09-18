"""Inspect this machine. Does not connect to, start, or stop any cloud GPU."""
import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lab1.common import environment, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument("--image-metrics", action="store_true")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
info = environment()
info["image_metric_imports"] = {}
info["image_metric_errors"] = {}
for name in ["cleanfid", "lpips", "prdc"]:
    try:
        importlib.import_module(name)
        info["image_metric_imports"][name] = True
    except Exception as error:
        info["image_metric_imports"][name] = False
        info["image_metric_errors"][name] = f"{type(error).__name__}: {error}"
info["passed"] = (not args.require_cuda or info["cuda_available"]) and (
    not args.image_metrics or all(info["image_metric_imports"].values()))
print(json.dumps(info, indent=2))
if args.output:
    write_json(args.output, info)
raise SystemExit(0 if info["passed"] else 1)
