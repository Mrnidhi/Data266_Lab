"""Bundle the Part 3 runner, dataset, checkpoint and scoring weights.

The output must be new. Existing packages are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCES = ROOT / "scripts" / "part3_push_package"
CODE = [
    "src/lab1/__init__.py",
    "src/lab1/common.py",
    "task3_gan/srinidhi/src/cyclegan.py",
    "task3_gan/srinidhi/src/cyclegan_ema.py",
    "task3_gan/srinidhi/src/cyclegan_push.py",
    "scripts/finetune_part3_push.py",
    "tests/test_cyclegan_push.py",
    "tests/test_part3_push_runner.py",
]
TOP_LEVEL = ["run_all.py", "README_5090.txt", "requirements.txt"]
EVALUATOR = "reproducibility/packages/part3-20261002/Part3_Evaluation_Script.ipynb"
EVALUATOR_SHA256 = "702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef"
DATASET = ROOT / "reproducibility/packages/part3-20261002/dataset.zip"
INCEPTION = Path.home() / ".cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"
STORED = {".zip", ".pt", ".pth", ".jpg"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(init: Path, arms: Path, out: Path, name: str) -> Path:
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError("Package name must be a single directory name.")
    staging = out / name
    archive = out / f"{name}.zip"
    if staging.exists() or archive.exists():
        raise FileExistsError("Package output already exists; choose a new --out or --name.")
    sources = [ROOT / relative for relative in (*CODE, EVALUATOR)]
    sources += [PACKAGE_SOURCES / relative for relative in TOP_LEVEL]
    sources += [init, arms, DATASET, INCEPTION]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if sha256(ROOT / EVALUATOR) != EVALUATOR_SHA256:
        raise ValueError("The evaluation notebook differs from the supplied one.")
    for relative in (*CODE, EVALUATOR):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for relative in TOP_LEVEL:
        shutil.copy2(PACKAGE_SOURCES / relative, staging / relative)
    shutil.copy2(arms, staging / "arms.json")
    (staging / "pytest.ini").write_text(
        "[pytest]\npythonpath = src\ntestpaths = tests\n", encoding="utf-8"
    )
    (staging / "task3_gan/data").mkdir(parents=True, exist_ok=True)
    inputs = staging / "inputs"
    (inputs / "torch_home/hub/checkpoints").mkdir(parents=True)
    shutil.copy2(DATASET, inputs / "dataset.zip")
    shutil.copy2(init, inputs / "init.pt")
    shutil.copy2(INCEPTION, inputs / "torch_home/hub/checkpoints" / INCEPTION.name)
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "init_source": str(init),
        "init_sha256": sha256(init),
        "sha256": {p.relative_to(staging).as_posix(): sha256(p) for p in files},
    }
    (inputs / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            bundle.write(
                path,
                Path(name) / path.relative_to(staging),
                compress_type=(
                    zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
                ),
            )
    print(
        json.dumps(
            {
                "package": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "files": len(files) + 1,
                "init_sha256": manifest["init_sha256"],
            },
            indent=2,
        )
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--init",
        type=Path,
        required=True,
        help="starting checkpoint with all four networks",
    )
    parser.add_argument("--arms", type=Path, default=PACKAGE_SOURCES / "arms.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/part3-push-work/package")
    parser.add_argument("--name", default="Part3_Push_5090")
    args = parser.parse_args()
    build(args.init.resolve(), args.arms.resolve(), args.out.resolve(), args.name)


if __name__ == "__main__":
    main()
