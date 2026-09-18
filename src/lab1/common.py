"""Run provenance and configuration utilities shared by the three tasks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


TASK_DIRECTORIES = {"gpt": "task1_llm", "sentiment": "task2_sentiment", "cyclegan": "task3_gan"}


def task_config_path(task: str) -> Path:
    """Return Srinidhi's task config from the shared team repository."""
    try:
        folder = TASK_DIRECTORIES[task]
    except KeyError as error:
        raise ValueError(f"Unknown task {task!r}; choose from {tuple(TASK_DIRECTORIES)}") from error
    return project_root() / folder / "srinidhi" / "config.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def code_manifest(root: Path | None = None) -> dict:
    root = root or project_root()
    files = sorted([*root.glob("src/**/*.py"), *root.glob("task*/srinidhi/src/*.py"),
                    *root.glob("task*/srinidhi/config.json")])
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def environment() -> dict:
    import torch
    package_names = ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy",
                     "matplotlib", "datasets", "Pillow", "clean-fid", "lpips", "prdc"]
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": prop.name, "memory_bytes": prop.total_memory,
                            "compute_capability": [prop.major, prop.minor]})
    info = {"captured_utc": utc_now(), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages, "cuda_available": cuda, "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpus": devices,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())}
    if cuda:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                     "--format=csv"], capture_output=True, text=True, timeout=10)
            info["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            info["nvidia_smi_error"] = str(error)
    return info


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu for local checks.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return str(device)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    # Values must be valid JSON: --set epochs=2 --set 'data.monet_dir="data/monet"'.
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid override {item!r}; expected dotted.key=JSON_VALUE")
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            if not isinstance(target[part], dict):
                raise ValueError(f"Cannot descend through non-object {part!r}")
            target = target[part]
        target[parts[-1]] = json.loads(raw)
    return config


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
