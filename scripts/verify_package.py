"""Extract the small preparation ZIP elsewhere and verify imports, caches and smoke runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / "dist/lab1-2342-prepared.zip"
with tempfile.TemporaryDirectory(prefix="lab1-relocated-") as temporary:
    destination = Path(temporary)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise RuntimeError("Corrupt ZIP")
        for name in handle.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe archive member")
        handle.extractall(destination)
    project = destination / "lab1-2342"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project / "src")
    command = [sys.executable, "scripts/rehearsal.py", "--mode", "smoke", "--device", "cpu", "--hourly-rate", "0", "--max-minutes", "10", "--output", str(project / "runs/relocated-check")]
    completed = subprocess.run(command, cwd=project, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    (ROOT / "dist/relocated-smoke.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError("Relocated smoke failed; inspect dist/relocated-smoke.log")
    check = '''import json
from pathlib import Path
import lab1
from lab1.common import task_config_path
from lab1.gpt import prepare_data
from lab1.sentiment import prefetch_data
root = Path.cwd()
assert Path(lab1.__file__).resolve().is_relative_to(root)
caches = {}
if (root / "data/rehearsal/tinystories").is_dir():
    cfg = json.loads(task_config_path("gpt").read_text())["rehearsal"]
    cfg.update(data_cache=str(root / "data/rehearsal/tinystories"), offline=True)
    a, b, _ = prepare_data(cfg)
    assert (len(a), len(b)) == (512, 64)
    caches["tinystories"] = "passed"
else:
    caches["tinystories"] = "not_in_source_bundle"
if (root / "data/rehearsal/yelp/manifest.json").is_file():
    cfg = json.loads(task_config_path("sentiment").read_text())["rehearsal"]
    manifest = prefetch_data(cfg, root / "data/rehearsal/yelp")
    assert manifest["split_counts"] == {"train": 2048, "validation": 256, "test": 512}
    caches["yelp"] = "passed"
else:
    caches["yelp"] = "not_in_source_bundle"
print(json.dumps(caches))
'''
    cache_check = subprocess.run([sys.executable, "-c", check], cwd=project, env=env, check=True, timeout=120, capture_output=True, text=True)
    cache_status = json.loads(cache_check.stdout.strip().splitlines()[-1])
result = {"status": "passed", "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "checks": ["ZIP CRC", "relocated source imports", "all five synthetic CPU model pipelines"], "optional_rehearsal_caches": cache_status, "cuda_verified": False}
(ROOT / "dist/relocation_verification.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
