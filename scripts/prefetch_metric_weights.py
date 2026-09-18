#!/usr/bin/env python3
"""Download/cache frozen metric weights and exercise real CPU metric APIs.

Synthetic fixtures below validate plumbing only. Their scores are not lab results.
Use --offline after copying the cache to a GPU machine to verify cached loading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("LAB1_METRIC_CACHE", ROOT / ".cache" / "metric_weights")))
    parser.add_argument("--output", type=Path, default=ROOT / "verification" / "metric_dependencies.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    os.environ["TORCH_HOME"] = str(args.cache_dir / "torch")
    os.environ["LAB1_METRIC_CACHE"] = str(args.cache_dir)
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18
    import lpips
    from cleanfid import fid
    from prdc import compute_prdc
    from lab1.cyclegan import (Generator, PatchDiscriminator,
                              build_metric_feature_extractor, configure_metric_cache)

    torch.set_num_threads(2)
    torch.manual_seed(2342)
    np.random.seed(2342)
    configure_metric_cache(args.cache_dir)
    required = [args.cache_dir / "cleanfid" / "inception-2015-12-05.pt",
                args.cache_dir / "torch" / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth",
                args.cache_dir / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"]
    if args.offline and any(not path.is_file() for path in required):
        parser.error("Offline cache incomplete: " + ", ".join(str(x) for x in required if not x.is_file()))
    if args.offline:
        def no_download(*unused_args, **unused_kwargs):
            raise RuntimeError("Offline verification attempted an unexpected download")
        torch.hub.download_url_to_file = no_download
    start = time.perf_counter()
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "seed": 2342,
              "purpose": "Real dependency/weight/API checks using synthetic fixtures; no lab-quality claim",
              "offline": args.offline, "cache_dir": str(args.cache_dir), "checks": {}, "versions": {}}
    for package in ("torch", "torchvision", "clean-fid", "lpips", "prdc", "numpy", "scipy", "Pillow"):
        report["versions"][package] = importlib.metadata.version(package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("Loading Clean-FID Inception evaluation weights...", flush=True)
        feature_model = build_metric_feature_extractor("cpu", args.cache_dir, download=not args.offline)
        with tempfile.TemporaryDirectory(prefix="lab1_metric_api_") as temporary:
            folder = Path(temporary)
            for i in range(8):
                pixels = np.random.default_rng(2342 + i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i:02d}.png")
            features = fid.get_folder_features(str(folder), model=feature_model, num_workers=0,
                                               batch_size=2, device=torch.device("cpu"), mode="clean", verbose=False)
        assert features.shape == (8, 2048) and np.isfinite(features).all()
        report["checks"]["cleanfid_features"] = {"status": "passed", "shape": list(features.shape), "mode": "clean"}
        fake = features + np.random.default_rng(2342).normal(0, .01, features.shape)
        kid = float(fid.kernel_distance(features, fake, num_subsets=2, max_subset_size=6))
        assert np.isfinite(kid)
        report["checks"]["kid_api"] = {"status": "passed", "finite": True, "subsets": 2, "subset_size": 6}
        prdc = compute_prdc(features, fake, nearest_k=3)
        assert set(prdc) == {"precision", "recall", "density", "coverage"}
        assert all(np.isfinite(v) for v in prdc.values())
        report["checks"]["prdc_api"] = {"status": "passed", "keys": sorted(prdc), "nearest_k": 3}
        # Check the API on low-dimensional fixtures, not a costly/singular 2048D covariance fit.
        small_real = np.random.default_rng(2342).normal(size=(20, 8))
        fid_check = float(fid.fid_from_feats(small_real, small_real.copy()))
        assert abs(fid_check) < 1e-8
        report["checks"]["fid_api"] = {"status": "passed", "same_distribution_zero_within": 1e-8,
                                         "dimension": 8, "full_2048d_fid_computed": False}
        del feature_model
        print("Loading LPIPS AlexNet evaluation weights...", flush=True)
        perceptual = lpips.LPIPS(net="alex", version="0.1").eval()
        perceptual.requires_grad_(False)
        x = torch.rand(2, 3, 64, 64) * 2 - 1
        with torch.inference_mode():
            distances = perceptual(x, x.clone())
        assert distances.shape == (2, 1, 1, 1) and torch.isfinite(distances).all()
        assert float(distances.abs().max()) < 1e-7
        report["checks"]["lpips_cycle_api"] = {"status": "passed", "shape": list(distances.shape),
                                                "identical_pairs_zero_within": 1e-7, "net": "alex", "version": "0.1"}
        del perceptual
        print("Loading frozen ResNet18 content evaluation weights...", flush=True)
        weights = ResNet18_Weights.IMAGENET1K_V1
        network = resnet18(weights=weights).eval()
        encoder = nn.Sequential(*list(network.children())[:-1]).eval()
        encoder.requires_grad_(False)
        with torch.inference_mode():
            encoded = encoder(weights.transforms()((x + 1) / 2)).flatten(1)
            cosine = torch.nn.functional.cosine_similarity(encoded, encoded)
        assert encoded.shape == (2, 512) and torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)
        report["checks"]["content_cosine_api"] = {"status": "passed", "shape": list(encoded.shape),
                                                   "identical_pairs_one_within": 1e-6, "weights": "ResNet18_IMAGENET1K_V1"}
        del network, encoder
        print("Verifying full CycleGAN architecture inference shapes...", flush=True)
        generator, discriminator = Generator(64, 9).eval(), PatchDiscriminator(64).eval()
        with torch.inference_mode():
            generated = generator(torch.zeros(1, 3, 256, 256))
            patches = discriminator(generated)
        assert tuple(generated.shape) == (1, 3, 256, 256) and tuple(patches.shape) == (1, 1, 30, 30)
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        architecture = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu",
                        "torch": torch.__version__, "purpose": "Full architecture forward check, not training/convergence",
                        "generator": {"blocks": 9, "base_channels": 64, "parameters_each": gen_params,
                                      "input_shape": [1, 3, 256, 256], "output_shape": list(generated.shape)},
                        "discriminator": {"receptive_field": 70, "parameters_each": disc_params, "output_shape": list(patches.shape)},
                        "two_generators_two_discriminators_total_parameters": 2 * (gen_params + disc_params), "status": "passed"}
        (ROOT / "verification" / "full_architecture_cyclegan.json").write_text(json.dumps(architecture, indent=2) + "\n")
        report["checks"]["full_architecture"] = {"status": "passed", "total_parameters": 2 * (gen_params + disc_params)}
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report["cached_files"] = [{"relative_path": str(path.relative_to(args.cache_dir)), "bytes": path.stat().st_size,
                                    "sha256": checksum(path)} for path in required if path.is_file()]
        report["total_cached_bytes"] = sum(x["bytes"] for x in report["cached_files"])
        report["transfer"] = "Optionally copy .cache/metric_weights/ to the same project-relative path on the GPU machine; run this script --offline. Otherwise first real evaluation downloads weights. Dependencies still need installation."
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
