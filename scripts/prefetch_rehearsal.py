"""Prepare verified public text subsets locally, without training or allocating a GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import task_config_path, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tasks", nargs="+", choices=["gpt", "sentiment"], default=["gpt", "sentiment"])
args = parser.parse_args()
for task in args.tasks:
    config = json.loads(task_config_path(task).read_text())["rehearsal"]
    if task == "gpt":
        from lab1.gpt import prepare_data, data_cache_paths
        config["data_cache"] = str(ROOT / "data/rehearsal/tinystories")
        train, validation, manifest = prepare_data(config)
        result = {"task": task, "counts": {"train": len(train), "validation": len(validation)}, "paths": data_cache_paths(config), "manifest_sha256": manifest["manifest_sha256"]}
    else:
        from lab1.sentiment import prefetch_data
        result = {"task": task, **prefetch_data(config, ROOT / "data/rehearsal/yelp")}
    write_json(ROOT / "verification" / f"{task}_data_cache.json", result)
    print(json.dumps(result, indent=2), flush=True)
print("Copy data/rehearsal with the package. scripts/rehearsal.py detects these caches; no model was trained.")
