"""Train, evaluate and package the Part 3 experiments. See README_5090.txt.

Use --smoke for a short rehearsal, --resume to continue saved experiments,
or --finalize-only to export and score the best saved checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
WORK = ROOT / "work"
DATA = ROOT / "task3_gan" / "data"
NOTEBOOK = (
    ROOT / "reproducibility" / "packages" / "part3-20261002" / "Part3_Evaluation_Script.ipynb"
)
TRAINER = ROOT / "scripts" / "finetune_part3_push.py"
TORCH_HOME = INPUTS / "torch_home"
STOP = WORK / "STOP"
COUNTS = {"monet_jpg": 300, "photo_jpg": 7038}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "run_all.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def child_env() -> dict:
    env = dict(os.environ)
    env["TORCH_HOME"] = str(TORCH_HOME)  # evaluator Inception weights ship in the package
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (no effect elsewhere)."""
    if platform.system() == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001
        )  # CONTINUOUS | SYSTEM_REQUIRED


def check_environment(requested: str) -> dict:
    missing = []
    for module in ("torch", "torchvision", "numpy", "scipy", "PIL", "pandas"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            f"Missing Python packages: {missing}. Install with: {sys.executable} -m pip install "
            "-r requirements.txt (see README_5090.txt)"
        )
    import numpy, scipy, torch, torchvision, PIL

    device = requested
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    env = {
        "captured_utc": now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "device": device,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    for optional in ("tqdm", "nbclient", "nbformat", "ipykernel", "pytest"):
        try:
            module = __import__(optional)
            env[optional] = getattr(module, "__version__", "present")
        except ImportError:
            env[optional] = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but torch.cuda.is_available() is False. Use the CUDA build of "
                "PyTorch (README_5090.txt) or pass --device cpu for a slow test."
            )
        properties = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu": properties.name,
                "gpu_memory_gb": round(properties.total_memory / 2**30, 1),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
        try:
            env["nvidia_smi"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            env["nvidia_smi"] = None
        # A real CUDA op catches wheels built without kernels for this GPU (e.g. sm_120 on old builds).
        probe = torch.randn(64, 64, device="cuda")
        env["cuda_matmul_ok"] = bool(torch.isfinite(probe @ probe).all())
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    env["disk_free_gb"] = round(free_gb, 1)
    if free_gb < 6:
        raise SystemExit(
            f"Only {free_gb:.1f} GB free next to the package; at least 6 GB is needed."
        )
    log(
        "environment: "
        + ", ".join(
            f"{k}={env[k]}"
            for k in (
                "python",
                "torch",
                "device",
                "gpu",
                "gpu_memory_gb",
                "bf16_supported",
                "disk_free_gb",
            )
            if k in env
        )
    )
    return env


def verify_inputs() -> None:
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(
                f"Checksum mismatch for {relative}: the package is incomplete or corrupted. "
                "Re-copy the ZIP and extract it again."
            )
    log(f"verified {len(manifest['sha256'])} package files by SHA-256")


def prepare_data() -> None:
    if all(
        (DATA / folder).is_dir() and len(list((DATA / folder).glob("*.jpg"))) == count
        for folder, count in COUNTS.items()
    ):
        log("class images already extracted")
        return
    for folder in COUNTS:
        shutil.rmtree(DATA / folder, ignore_errors=True)
        (DATA / folder).mkdir(parents=True)
    with zipfile.ZipFile(INPUTS / "dataset.zip") as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if (
                member.is_dir()
                or "__MACOSX" in parts
                or len(parts) != 3
                or parts[0] != "dataset"
                or parts[1] not in COUNTS
                or not parts[2].lower().endswith(".jpg")
            ):
                continue
            (DATA / parts[1] / parts[2]).write_bytes(archive.read(member))
    for folder, count in COUNTS.items():
        found = len(list((DATA / folder).glob("*.jpg")))
        if found != count:
            raise SystemExit(f"{folder}: expected {count} images after extraction, found {found}")
    log("extracted class images: 300 Monet, 7038 photos")


def run_tests() -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        log("pytest not installed; skipping unit tests (optional)")
        return
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
    )
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-3:])
    log("unit tests: " + tail.replace("\n", " | "))
    if result.returncode != 0:
        (WORK / "unit_tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        raise SystemExit("Unit tests failed; see work/unit_tests.log. Nothing was trained.")


def arm_flags(
    arm: dict, defaults: dict, device: str, smoke: bool, resolved: dict | None = None
) -> dict:
    flags = {**defaults, **arm.get("flags", {}), **(resolved or {})}
    if device != "cuda":
        flags["precision"] = "fp32"
    if smoke:
        # A rehearsal, not a result: tiny schedule and at most batch 2, so it also runs on small GPUs.
        flags.update(
            {
                "steps": 40,
                "eval_every": 40,
                "checkpoint_every": 20,
                "log_every": 10,
                "batch_size": min(int(flags.get("batch_size", 1)), 2),
            }
        )
    return flags


def arm_command(
    arm: dict,
    flags: dict,
    device: str,
    output: Path,
    resume: bool,
    stop: Path = STOP,
    extra: tuple[str, ...] = (),
) -> list[str]:
    command = [
        sys.executable,
        str(TRAINER),
        "train",
        "--init",
        str(INPUTS / arm.get("init", "init.pt")),
        "--output",
        str(output),
        "--device",
        device,
        "--stop-file",
        str(stop),
        *extra,
    ]
    for key, value in flags.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command + (["--resume"] if resume else [])


def pid_alive(pid: int) -> bool:
    """True if a process with this PID is running. (On Windows os.kill(pid, 0) would terminate it.)"""
    if platform.system() == "Windows":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(
            0x1000, False, pid
        )  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def isolation() -> dict:
    # Own process group: a console Ctrl+C reaches only this runner, which then stops arms cleanly.
    return (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if platform.system() == "Windows"
        else {"start_new_session": True}
    )


def benchmark(plan: dict, device: str, seconds: float, smoke: bool = False) -> dict[str, float]:
    """Run every arm at once without scoring and return each arm's steady updates/second."""
    root = WORK / "benchmark"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    stop = root / "STOP"
    processes = []
    for arm in plan["arms"]:
        flags = arm_flags(
            arm,
            plan.get("defaults", {}),
            device,
            False,
            {
                "steps": 10_000_000,
                "eval_every": 10_000_000,
                "checkpoint_every": 10_000_000,
                "log_every": 5,
            },
        )
        if smoke:
            flags["batch_size"] = min(int(flags.get("batch_size", 1)), 2)
        command = arm_command(
            arm,
            flags,
            device,
            root / arm["name"],
            False,
            stop,
            ("--no-score", "--stop-after", "3000"),
        )  # ends on its own if orphaned
        stream = (root / f"{arm['name']}.out").open("w", encoding="utf-8")
        processes.append(
            (
                arm["name"],
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=child_env(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    **isolation(),
                ),
                stream,
            )
        )
    log(
        f"benchmarking {len(processes)} arm(s) together for about {seconds:.0f} s to size the schedules"
    )

    def logged_steps(name: str) -> int:
        path = root / name / "training_log.jsonl"
        return path.read_text(encoding="utf-8").count('{"step"') if path.exists() else 0

    started = time.time()
    # Run at least `seconds`, and until every arm has logged enough steps (slow machines), up to 4x.
    while time.time() - started < seconds or (
        min(logged_steps(n) for n, _, _ in processes) < 8 and time.time() - started < 4 * seconds
    ):
        time.sleep(5)
        if all(process.poll() is not None for _, process, _ in processes):
            break
    stop.write_text("stop", encoding="utf-8")
    rates = {}
    for name, process, stream in processes:
        try:
            process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
        stream.close()
        lines = (
            (root / name / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
            if (root / name / "training_log.jsonl").exists()
            else []
        )
        steps = [json.loads(line) for line in lines if line.startswith('{"step"')]
        steady = steps[2:] if len(steps) > 4 else steps  # skip cuDNN autotuning and warm-up
        if len(steady) < 2:
            raise SystemExit(
                f"Benchmark of {name} produced too few steps; see {root / (name + '.out')}"
            )
        rates[name] = (steady[-1]["step"] - steady[0]["step"]) / (
            steady[-1]["elapsed"] - steady[0]["elapsed"]
        )
    shutil.rmtree(root, ignore_errors=True)
    log(
        "benchmark it/s with all arms running: "
        + ", ".join(f"{k} {v:.2f}" for k, v in rates.items())
    )
    return rates


def resolve_schedule(
    plan: dict, rates: dict[str, float], hours: float, filename: str = "schedule.json"
) -> dict[str, dict]:
    """Per arm, pick steps so its full constant-then-decay schedule ends inside the budget."""
    defaults = plan.get("defaults", {})
    eval_minutes = float(
        plan.get("eval_minutes_per_arm", 25)
    )  # time all scoring rounds of one arm take
    train_seconds = max(600.0, hours * 3600 - eval_minutes * 60 - 600)
    resolved = {}
    for arm_name, rate in rates.items():
        steps = int(rate * train_seconds * 0.9 // 1000 * 1000)
        steps = max(
            int(plan.get("min_steps", 10_000)),
            min(int(plan.get("max_steps", 400_000)), steps),
        )
        resolved[arm_name] = {}
        if defaults.get("steps") == "auto":
            resolved[arm_name]["steps"] = steps
        if defaults.get("eval_every") == "auto":
            rounds = int(
                plan.get("eval_rounds", 10)
            )  # scoring rounds per arm (each scores every variant)
            resolved[arm_name]["eval_every"] = max(
                500, (resolved[arm_name].get("steps", steps) // rounds) // 250 * 250
            )
    log(f"schedule (budget {hours:.2f} h): " + ", ".join(f"{k} {v}" for k, v in resolved.items()))
    (WORK / filename).write_text(
        json.dumps({"rates": rates, "hours": hours, "arms": resolved}, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def load_schedule(path: Path, plan: dict) -> dict[str, dict]:
    if not path.is_file():
        raise SystemExit(
            f"Cannot resume without {path}. Restore the original verified schedule; "
            "re-benchmarking would change the saved training recipe."
        )
    resolved = json.loads(path.read_text(encoding="utf-8"))["arms"]
    if set(resolved) != {arm["name"] for arm in plan["arms"]}:
        raise SystemExit("The saved schedule does not match the experiments in arms.json.")
    for flags in resolved.values():
        for key in ("steps", "eval_every"):
            if plan.get("defaults", {}).get(key) == "auto":
                value = flags.get(key)
                if type(value) is not int or value <= 0:
                    raise SystemExit(f"The saved schedule needs a positive integer for {key}.")
    return resolved


def read_progress(output: Path) -> dict:
    progress = {"step": 0, "best": None, "rate": None, "status": "running"}
    log_path = output / "training_log.jsonl"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        steps = [json.loads(line) for line in lines if line.startswith('{"step"')]
        evals = [
            json.loads(line)["evaluation"] for line in lines if line.startswith('{"evaluation"')
        ]
        if steps:
            progress["step"] = steps[-1]["step"]
            recent = steps[-10:]
            # Elapsed restarts at zero in a resumed session, so only use the latest increasing run.
            while len(recent) >= 2 and recent[0]["elapsed"] >= recent[-1]["elapsed"]:
                recent = recent[1:]
            if len(recent) >= 2:
                progress["rate"] = (recent[-1]["step"] - recent[0]["step"]) / (
                    recent[-1]["elapsed"] - recent[0]["elapsed"]
                )
        if evals:
            progress["best"] = min(evals, key=lambda e: e["composite"])
    summary = output / "summary.json"
    if summary.exists():
        progress["status"] = json.loads(summary.read_text(encoding="utf-8"))["status"]
    return progress


def train_arms(
    plan: dict,
    device: str,
    parallel: int,
    hours: float,
    resume: bool,
    smoke: bool,
    resolved: dict | None = None,
) -> None:
    root = WORK / ("smoke" if smoke else "arms")
    if smoke:
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + hours * 3600
    isolate = isolation()
    queue, running = [], {}
    for arm in plan["arms"]:
        output = root / arm["name"]
        pid_file = root / f"{arm['name']}.pid"
        if pid_file.exists():
            text = pid_file.read_text(encoding="utf-8").strip()
            if text.isdigit() and pid_alive(int(text)):
                raise SystemExit(
                    f"{arm['name']} still appears to be running as PID {text}. Let it finish or stop it "
                    f"(create {STOP}), or delete {pid_file} if that PID is not a training process."
                )
            pid_file.unlink()
        summary = output / "summary.json"
        if (
            summary.exists()
            and json.loads(summary.read_text(encoding="utf-8"))["status"] == "completed"
        ):
            log(f"{arm['name']}: already completed, skipping")
            continue
        continuing = (output / "last.pt").exists()
        if output.exists() and not continuing:
            shutil.rmtree(output)  # started but never checkpointed: nothing to keep
        if continuing and not resume and not smoke:
            raise SystemExit(
                f"{output} already has a checkpoint. Use --resume to continue or --finalize-only."
            )
        queue.append((arm, output, continuing))
    log(
        f"training {len(queue)} arm(s), up to {parallel} at once, budget {hours:.2f} h; stop early with {STOP}"
    )
    stop_sent = False
    totals = {}
    try:
        while queue or running:
            while queue and len(running) < parallel and not stop_sent:
                arm, output, continuing = queue.pop(0)
                flags = arm_flags(
                    arm,
                    plan.get("defaults", {}),
                    device,
                    smoke,
                    (resolved or {}).get(arm["name"]),
                )
                totals[arm["name"]] = flags.get("steps")
                command = arm_command(arm, flags, device, output, continuing)
                stream = (root / f"{arm['name']}.out").open("a", encoding="utf-8")
                stream.write(f"\n# {now()} {' '.join(command)}\n")
                stream.flush()
                running[arm["name"]] = (
                    subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=child_env(),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        **isolate,
                    ),
                    output,
                    stream,
                )
                (root / f"{arm['name']}.pid").write_text(
                    str(running[arm["name"]][0].pid), encoding="utf-8"
                )
                log(
                    f"{arm['name']}: {'resumed' if continuing else 'started'} ({arm.get('note', '')})"
                )
            for _ in range(60 if not smoke else 5):
                time.sleep(1)
                if any(process.poll() is not None for process, _, _ in running.values()):
                    break
            for name, (process, output, stream) in list(running.items()):
                if process.poll() is not None:
                    stream.close()
                    del running[name]
                    (root / f"{name}.pid").unlink(missing_ok=True)
                    progress = read_progress(output)
                    best = progress["best"]
                    log(
                        f"{name}: exited with code {process.returncode}, status {progress['status']}, "
                        f"best composite {best['composite']:.3f} at step {best['step']} ({best['variant']})"
                        if best
                        else f"{name}: exited with code {process.returncode} before any score"
                    )
                    if process.returncode != 0:
                        log(
                            f"{name}: see {root / (name + '.out')} for the error; other arms continue"
                        )
            if running:
                rows = []
                for name, (_, output, _) in running.items():
                    p = read_progress(output)
                    best = p["best"]
                    rows.append(
                        f"{name} {p['step']}/{totals.get(name)}"
                        + (f" {p['rate']:.1f} it/s" if p["rate"] else "")
                        + (f" best {best['composite']:.3f}@{best['step']}" if best else "")
                    )
                log(" | ".join(rows))
            if not stop_sent and (STOP.exists() or time.time() > deadline):
                STOP.parent.mkdir(parents=True, exist_ok=True)
                STOP.write_text("stop", encoding="utf-8")
                stop_sent = True
                queue.clear()
                log("time budget reached or STOP requested: arms will score, checkpoint and exit")
    except KeyboardInterrupt:
        STOP.write_text("stop", encoding="utf-8")
        log(
            "Ctrl+C: asking arms to score, checkpoint and exit (press Ctrl+C again only if they hang)"
        )
        for name, (process, _, stream) in running.items():
            try:
                process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                process.kill()
            stream.close()
        raise SystemExit(
            "Stopped. Run again with --resume to continue, or --finalize-only to export."
        )


def best_candidate(root: Path) -> tuple[Path, dict]:
    import torch

    candidates = []
    for path in sorted(root.glob("*/best.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append(
            (
                saved["class_scores"]["composite"],
                path,
                saved["class_scores"],
                saved["state"],
            )
        )
    if not candidates:
        raise SystemExit(f"No best.pt found under {root}; nothing to finalize.")
    for composite, path, _, state in sorted(candidates, key=lambda c: c[0]):
        log(
            f"candidate {path.parent.name}: composite {composite:.4f} (step {state['global_step']}, "
            f"{state['candidate_variant']})"
        )
    composite, path, scores, state = min(candidates, key=lambda c: c[0])
    return path, {
        "arm": path.parent.name,
        "class_scores_during_training": scores,
        "state": state,
    }


def run_official_notebook(export: Path, official: Path) -> dict | None:
    """Execute the supplied notebook unchanged in the folder layout it expects ("Part 3/Data")."""
    try:
        import nbclient, nbformat  # noqa: F401
        import ipykernel  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError as error:
        log(
            f"official notebook run skipped ({error.name} not installed); export/metrics.json has the same arithmetic"
        )
        return None
    import csv
    import nbformat
    from nbclient import NotebookClient

    # A private kernel bound to this interpreter: a machine's default "python3" kernelspec may point
    # at another (or deleted) environment. The kernel inherits TORCH_HOME, so no weight download.
    kernel_dir = WORK / "jupyter" / "kernels" / "part3_run_all"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Part 3 run_all",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_PATH"] = (
        str(WORK / "jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    )
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    base = official / "Part 3" / "Data"
    shutil.rmtree(official, ignore_errors=True)
    base.mkdir(parents=True)
    try:
        for folder in COUNTS:
            shutil.copytree(DATA / folder, base / folder)
        for folder in ("pred_A2B", "pred_B2A"):
            shutil.copytree(export / folder, base / folder)
        notebook_copy = official / "Part3_Evaluation_Script.ipynb"
        shutil.copy2(NOTEBOOK, notebook_copy)
        notebook = nbformat.read(notebook_copy, as_version=4)
        log(
            "running the supplied evaluation notebook unchanged (this re-extracts Inception features)"
        )
        NotebookClient(
            notebook,
            timeout=7200,
            kernel_name="part3_run_all",
            resources={"metadata": {"path": str(official)}},
        ).execute()
        nbformat.write(notebook, official / "Part3_Evaluation_Script.executed.ipynb")
        notebook_copy.unlink()
        with (official / "submission.csv").open(encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
    finally:
        shutil.rmtree(
            official / "Part 3", ignore_errors=True
        )  # exports are stored once, under export/
    result = {"FID": float(row["FID"]), "MiFID": float(row["MiFID"])}
    result["composite"] = (result["FID"] + result["MiFID"]) / 2
    return result


def finalize(device: str, env: dict, smoke: bool) -> Path:
    root = WORK / ("smoke" if smoke else "arms")
    final = WORK / ("final_smoke" if smoke else "final")
    shutil.rmtree(final, ignore_errors=True)
    final.mkdir(parents=True)
    chosen, details = best_candidate(root)
    shutil.copy2(chosen, final / "best.pt")
    export = final / "export"
    command = [
        sys.executable,
        str(TRAINER),
        "export",
        "--checkpoint",
        str(final / "best.pt"),
        "--output",
        str(export),
        "--device",
        device,
        "--data-root",
        str(DATA),
    ] + (["--max-photos", "300"] if smoke else [])
    log(
        f"exporting {details['arm']} (step {details['state']['global_step']}, {details['state']['candidate_variant']})"
    )
    result = subprocess.run(command, cwd=ROOT, env=child_env(), capture_output=True, text=True)
    (final / "export.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"Export failed; see {final / 'export.log'}")
    metrics = json.loads((export / "metrics.json").read_text(encoding="utf-8"))
    log(
        f"export scored with the notebook's metric cells: FID {metrics['directional_means']['FID']:.4f}, "
        f"MiFID {metrics['directional_means']['MiFID']:.4f}, composite {metrics['local_composite']:.4f}"
    )
    check_cmd = [
        sys.executable,
        str(TRAINER),
        "check",
        "--checkpoint",
        str(final / "best.pt"),
        "--output",
        str(final / "check.json"),
        "--device",
        device,
        "--data-root",
        str(DATA),
    ]
    checked = subprocess.run(check_cmd, cwd=ROOT, env=child_env(), capture_output=True, text=True)
    (final / "check.log").write_text(checked.stdout + checked.stderr, encoding="utf-8")
    check = (
        json.loads((final / "check.json").read_text(encoding="utf-8"))
        if checked.returncode == 0
        else None
    )
    if check:
        fresh, memo = check["fresh_sample"], check["memorisation"]
        log(
            f"fresh-sample check (photos 301-600): FID A2B {fresh['fid_a2b_vs_photos_301_600']:.2f}, "
            f"B2A {fresh['fid_b2a_from_photos_301_600']:.2f}, composite ~{fresh['composite_estimate']:.2f}; "
            f"content cosine {check['content_cosine_input_vs_output']}; outputs closer to a training Monet than "
            f"any real pair: {memo['outputs_closer_to_a_training_monet_than_any_real_pair']}"
        )
    else:
        log(f"check step failed; see {final / 'check.log'}")
    candidates = sum(
        path.read_text(encoding="utf-8").count('{"evaluation"')
        for path in root.glob("*/training_log.jsonl")
    )
    log(
        f"checkpoint chosen among {candidates} scored candidates (report this: the class score is optimistic)"
    )
    official = None
    if not smoke:
        try:
            official = run_official_notebook(export, final / "official")
        except Exception as error:  # never lose the export over a Jupyter kernel problem
            log(
                f"official notebook run failed ({type(error).__name__}: {error}); "
                "export/metrics.json holds the same metric cells' result"
            )
    if official:
        log(
            f"supplied notebook submission.csv: FID {official['FID']:.4f}, MiFID {official['MiFID']:.4f}, "
            f"(FID+MiFID)/2 = {official['composite']:.4f}"
        )
    results = {
        "finished_utc": now(),
        "selected": details,
        "export_metrics": metrics,
        "official_notebook": official,
        "checks": check,
        "scored_candidates": candidates,
        "environment": env,
        "plan": json.loads((ROOT / "arms.json").read_text(encoding="utf-8")),
        "integrity": "Submitted images are direct JPEG outputs of the selected CycleGAN generators, one per "
        "source image in alphabetical source order; none were edited, reordered or picked. "
        "Checkpoint selection used the supplied class evaluator's score.",
    }
    (final / "RESULTS.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    archive = ROOT / f"Part3_Push_Results_{'SMOKE_' if smoke else ''}{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(final.rglob("*")):
            if path.is_file():
                stored = path.suffix.lower() in (".jpg", ".pt")  # already compressed
                bundle.write(
                    path,
                    Path("final") / path.relative_to(final),
                    compress_type=(zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED),
                )
        for path in sorted(root.glob("*")):
            if path.is_file():
                bundle.write(path, Path("arms") / path.name)
        for path in sorted(root.glob("*/*")):
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                bundle.write(path, Path("arms") / path.relative_to(root))
        for name in ("run_all.log", "environment.json"):
            if (WORK / name).exists():
                bundle.write(WORK / name, Path(name))
        bundle.write(ROOT / "arms.json", "arms.json")
    log(f"RESULTS ZIP: {archive}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="training time budget (default from arms.json)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="arms trained at once (default from arms.json)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue unfinished arms from last.pt"
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="skip training; export the best so far",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end rehearsal (minutes, not a result)",
    )
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    keep_awake()
    log(f"run_all start: {' '.join(sys.argv[1:]) or '(full run)'}")
    plan = json.loads((ROOT / "arms.json").read_text(encoding="utf-8"))
    env = check_environment(args.device)
    (WORK / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    verify_inputs()
    prepare_data()
    if STOP.exists():
        STOP.unlink()
    device = env["device"]
    if not args.finalize_only:
        if not args.skip_tests:
            run_tests()
        hours = args.hours if args.hours is not None else plan.get("hours", 5.0)
        parallel = args.parallel or plan.get("parallel", 3)
        resolved = None
        auto = "auto" in (
            plan.get("defaults", {}).get("steps"),
            plan.get("defaults", {}).get("eval_every"),
        )
        if auto and not args.smoke:
            if len(plan["arms"]) > parallel:
                raise SystemExit(
                    "With automatic steps every arm must run at once: set parallel >= number of arms"
                )
            saved = WORK / "schedule.json"
            if args.resume:
                resolved = load_schedule(saved, plan)
                log(f"resuming with the saved schedule {resolved}")
            else:
                resolved = resolve_schedule(
                    plan,
                    benchmark(plan, device, float(plan.get("benchmark_seconds", 150))),
                    hours,
                )
        elif auto and args.smoke:
            # Exercises the benchmark path only; smoke keeps tiny steps and never touches schedule.json.
            resolve_schedule(
                plan,
                benchmark(plan, device, 25.0, smoke=True),
                1.0,
                "schedule_smoke.json",
            )
        train_arms(
            plan,
            device,
            parallel,
            hours if not args.smoke else 1.0,
            args.resume,
            args.smoke,
            resolved,
        )
    archive = finalize(device, env, args.smoke)
    log("done. Bring back " + archive.name)


if __name__ == "__main__":
    main()
