"""Fixed-sample CycleGAN cycle/LPIPS and distribution evaluation, without training."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
MEMBER = ROOT / "task3_gan/srinidhi"
PROTOCOL = ROOT / "reproducibility/manifests/srinidhi/part3-metrics-20261004/protocol.json"
CHECKPOINT_SHA = "62d80f7ef2752b190fa496b7775b32dd77684bfceeb3b9f4c7356642848abc14"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("The frozen protocol already exists.")
    checkpoint = MEMBER / "checkpoints/best.pt"
    assert digest(checkpoint) == CHECKPOINT_SHA
    domains = {
        d: sorted((ROOT / f"task3_gan/data/{d}_jpg").glob("*.jpg")) for d in ("monet", "photo")
    }
    assert len(domains["monet"]) == 300 and len(domains["photo"]) == 7038
    samples = {
        "monet": domains["monet"],
        "photo": sorted(random.Random(2342).sample(domains["photo"], 300)),
    }
    directions = {}
    for direction, label, source, target in (
        ("monet_to_photo", "pred_A2B", "monet", "photo"),
        ("photo_to_monet", "pred_B2A", "photo", "monet"),
    ):
        export = json.loads((MEMBER / f"outputs/{label}_export_manifest.json").read_text())
        mapping = {r["source"]: r for r in export["images"]}
        pairs = []
        for path in samples[source]:
            output = MEMBER / "outputs" / label / mapping[path.name]["jpeg"]
            assert digest(output) == mapping[path.name]["jpeg_sha256"]
            pairs.append({"input": record(path), "prediction": record(output)})
        directions[direction] = {"pairs": pairs, "references": [record(p) for p in samples[target]]}
    protocol = {
        "checkpoint": record(checkpoint),
        "source": record(Path(__file__)),
        "seed": 2342,
        "sample_count_per_direction": 300,
        "device": "cpu",
        "dtype": "float32",
        "threads": 4,
        "sampling": "All 300 Monet images; random.Random(2342).sample of sorted photos, sorted after sampling; matched target reference subsets.",
        "evaluation_scope": "Class training images; fixed post-training diagnostic, not held-out validation or test.",
        "cycle": "Mean absolute input minus G_reverse(G_forward(input)), uncompressed float tensors in [-1,1].",
        "LPIPS": "LPIPS v0.1 AlexNet, input versus cycle reconstruction; frozen metric weights only.",
        "distribution": "Unmodified published JPEGs; existing ClassScorer torchvision IMAGENET1K_V1 Inception pooled2048 features; PRDC nearest_k=5. Not Clean-FID preprocessing.",
        "batch_size": 4,
        "directions": directions,
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n")
    print("Protocol frozen:", PROTOCOL.relative_to(ROOT), digest(PROTOCOL), flush=True)


def evaluate(output, benchmark=False):
    import numpy as np
    import torch
    import lpips

    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    from prdc import compute_prdc
    from lab1 import cyclegan as cg
    from lab1 import cyclegan_push as cp

    output = output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh metrics output file.")
    protocol = json.loads(PROTOCOL.read_text())
    assert digest(Path(__file__)) == protocol["source"]["sha256"]
    assert digest(ROOT / protocol["checkpoint"]["path"]) == CHECKPOINT_SHA
    for direction in protocol["directions"].values():
        for row in [
            *(v[k] for v in direction["pairs"] for k in ("input", "prediction")),
            *direction["references"],
        ]:
            assert digest(ROOT / row["path"]) == row["sha256"]
    torch.set_num_threads(4)
    torch.manual_seed(2342)
    saved = torch.load(MEMBER / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    models = {}
    for name in ("G_monet_to_photo", "G_photo_to_monet"):
        model = cg.Generator(saved["config"]["base_channels"], saved["config"]["residual_blocks"])
        model.load_state_dict(saved["models"][name], strict=True)
        models[name] = model.eval().requires_grad_(False)
    del saved
    with redirect_stdout(io.StringIO()):
        perceptual = lpips.LPIPS(net="alex", version="0.1").eval().requires_grad_(False)
    features = cp.InceptionFeatures("cpu")
    results = {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "benchmark_only": benchmark,
        "directions": {},
    }
    started = time.perf_counter()
    with torch.inference_mode():
        for direction, selection in protocol["directions"].items():
            pairs = selection["pairs"][:4] if benchmark else selection["pairs"]
            reference = selection["references"][:4] if benchmark else selection["references"]
            source, target = direction.split("_to_")
            cycle, lpips_values = [], []
            phase = time.perf_counter()
            for i in range(0, len(pairs), 4):
                x = torch.stack(
                    [cp.load_image(ROOT / p["input"]["path"], 256, None) for p in pairs[i : i + 4]]
                )
                y = models[f"G_{direction}"](x)
                reconstruction = models[f"G_{target}_to_{source}"](y)
                cycle.extend((reconstruction - x).abs().mean((1, 2, 3)).tolist())
                lpips_values.extend(perceptual(x, reconstruction).flatten().tolist())
                if (i + 4) % 40 == 0:
                    print(direction, "cycle", min(i + 4, len(pairs)), "/", len(pairs), flush=True)
            pair_seconds = time.perf_counter() - phase
            phase = time.perf_counter()
            real = features([cp.read_rgb(ROOT / p["path"]) for p in reference], batch_size=4)
            fake = features(
                [cp.read_rgb(ROOT / p["prediction"]["path"]) for p in pairs], batch_size=4
            )
            assert np.isfinite(real).all() and np.isfinite(fake).all()
            feature_seconds = time.perf_counter() - phase
            row = {
                "count": len(pairs),
                "cycle_reconstruction_L1": float(np.mean(cycle)),
                "LPIPS_cycle": float(np.mean(lpips_values)),
                "cycle_lpips_seconds": pair_seconds,
                "feature_seconds": feature_seconds,
            }
            if not benchmark:
                row.update(
                    {
                        f"generative_{k}": float(v)
                        for k, v in compute_prdc(real, fake, nearest_k=5).items()
                    }
                )
            assert all(np.isfinite(v) for v in row.values())
            results["directions"][direction] = row
            print(direction, json.dumps(row), flush=True)
    results["elapsed_seconds"] = time.perf_counter() - started
    results["library_versions"] = {"torch": torch.__version__, "numpy": np.__version__}
    results["metric_weights_sha256"] = {
        p.name: digest(p)
        for p in (
            Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth",
            Path(torch.hub.get_dir()) / "checkpoints/inception_v3_google-0cc3c7bd.pth",
            Path(lpips.__file__).parent / "weights/v0.1/alex.pth",
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("Saved", output.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "benchmark", "evaluate"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        if args.output is None:
            parser.error("--output is required")
        evaluate(args.output, benchmark=args.action == "benchmark")
