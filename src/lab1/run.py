"""One entry point for smoke, rehearsal, and full training."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

from .common import (Tee, apply_overrides, code_manifest, environment, project_root,
                     resolve_device, task_config_path, utc_now, write_json)


def execute(task: str, mode: str, output_dir: Path, device: str = "auto",
            config_path: Path | None = None, overrides: list[str] | None = None,
            resume: Path | None = None) -> dict:
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    path = config_path or task_config_path(task)
    all_configs = json.loads(path.read_text())
    if mode not in all_configs:
        raise ValueError(f"Missing profile {mode!r} in {path.name}")
    config = apply_overrides(all_configs[mode], overrides or [])
    config["mode"] = mode
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    manifest = {"task": task, "mode": mode, "device": device,
                "started_utc": utc_now(), "status": "running",
                "is_final_result": False, "config": config, "source_sha256": code_manifest(),
                "environment": environment(), "resumed": resume is not None}
    # A new invocation has its own provenance, including resumed invocations.
    invocation_name = manifest["started_utc"].replace(":", "-")
    manifest_path = output_dir / "provenance" / f"{invocation_name}.json"
    write_json(manifest_path, manifest)
    if resume is None or not (output_dir / "resolved_config.json").exists():
        write_json(output_dir / "resolved_config.json", config)
    start = time.perf_counter()
    with (output_dir / "RUN_LOG.txt").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(json.dumps({"event": "start", "task": task, "mode": mode, "utc": utc_now(), "device": device}))
            try:
                module = importlib.import_module(f"lab1.{task}")
                summary = module.run(config, output_dir, device, resume=resume)
                write_json(output_dir / "resolved_config.json", config)
                manifest.update(status="completed", ended_utc=utc_now(),
                                elapsed_seconds=time.perf_counter() - start)
                manifest["summary"] = summary
                write_json(output_dir / "run_summary.json", manifest)
                print(json.dumps({"event": "completed", "task": task, "mode": mode,
                                  "elapsed_seconds": manifest["elapsed_seconds"]}))
            except BaseException as error:
                manifest.update(status="failed", ended_utc=utc_now(),
                                elapsed_seconds=time.perf_counter() - start,
                                error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
                write_json(output_dir / "run_summary.json", manifest)
                raise
            finally:
                write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["gpt", "sentiment", "cyclegan"])
    parser.add_argument("--mode", choices=["smoke", "rehearsal", "full"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    stamp = utc_now().replace(":", "-")
    output = args.output or project_root() / "runs" / f"{args.mode}-{stamp}" / args.task
    execute(args.task, args.mode, output, args.device, args.config, args.overrides, args.resume)


if __name__ == "__main__":
    main()
