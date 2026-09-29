"""Read-only PatchGAN diagnostics on fixed train/validation image subsets.

Raw least-squares discriminator scores are not probabilities. Differences across
groups are measurements, not a threshold-based diagnosis of overfitting. This
script performs no optimization, checkpoint writing, or test-split inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import cyclegan as cg
from lab1.common import environment, resolve_device, utc_now
import numpy as np
import torch

SEED = 2342
LIMIT = 128


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Diagnostics require nonempty finite per-image measurements")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9))}


def subset_indices(count, seed, limit=LIMIT):
    if count < 1:
        raise ValueError("Empty source split")
    return list(range(count)) if count <= limit else sorted(random.Random(seed).sample(range(count), limit))


def patch_measurements(output):
    patches = output.detach().float().cpu().reshape(-1)
    if not len(patches) or not torch.isfinite(patches).all():
        raise FloatingPointError("Non-finite or empty discriminator patch output")
    return {"patch_count": len(patches), "patch_score_mean": patches.mean().item(),
            "patch_mse_to_1": ((patches - 1) ** 2).mean().item(),
            "patch_mse_to_0": (patches ** 2).mean().item()}


@torch.inference_mode()
def measure_group(discriminator, data, entries, indices, device, generator=None):
    records = []
    for index in indices:
        image = data[index].unsqueeze(0).to(device)
        if generator is not None:
            image = generator(image)
        row = patch_measurements(discriminator(image))
        row.update(source_index=index, relative_path=entries[index]["relative_path"],
                   source_image_sha256=entries[index]["sha256"])
        records.append(row)
    names = ("patch_score_mean", "patch_mse_to_1", "patch_mse_to_0")
    target = 0 if generator is not None else 1
    return {"count": len(records), "target": target,
            "target_mse_field": f"patch_mse_to_{target}",
            "summary": {name: summary([row[name] for row in records]) for name in names},
            "per_image": records}


def diagnose(checkpoint, config_path, output, device="cuda"):
    checkpoint, output = Path(checkpoint).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new diagnostics JSON path; existing evidence is preserved")
    if output == checkpoint:
        raise ValueError("Diagnostics cannot overwrite a checkpoint")
    source_hash = digest(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format_version") != 1 or saved.get("config", {}).get("mode") != "full":
        raise ValueError("Diagnostics require a trusted full class-data checkpoint")
    source_step = saved["state"]["global_step"]
    supplied = json.loads(Path(config_path).read_text())
    supplied = supplied.get("full", supplied)
    config = dict(saved["config"])
    # Relocation is permitted; prepare_data verifies the exact frozen contents.
    config["data"] = dict(supplied["data"])
    for key in ("monet_dir", "photo_dir", "manifest_dir"):
        path = Path(config["data"][key]).expanduser()
        config["data"][key] = str(path if path.is_absolute() else ROOT / path)
    data, manifest = cg.prepare_data(config)
    if (manifest.get("class_results") is not True or manifest.get("kind") != "explicit_class_manifests"
            or manifest["manifest_fingerprint"] != saved["state"].get("manifest_fingerprint")):
        raise ValueError("Diagnostics data differ from the checkpoint's frozen class manifests")
    device = resolve_device(device)
    models = cg.build_models(saved["config"], device)
    cg.load_checkpoint(saved, models, restore_random=False)
    del saved
    for model in models.values():
        model.eval().requires_grad_(False)
    chosen, plain = {}, {}
    for position, key in enumerate(("train_photo", "train_monet", "val_photo", "val_monet")):
        chosen[key] = subset_indices(len(data[key]), SEED + position)
        plain[key] = cg.ImageDomain(data[key].records, config["image_size"], augment=False)
    started = time.perf_counter()
    results = {}
    for domain, other in (("photo", "monet"), ("monet", "photo")):
        name = f"D_{domain}"
        groups = {}
        for group, key, generator in (("real_train", f"train_{domain}", None),
                                      ("real_val", f"val_{domain}", None),
                                      ("fake_val", f"val_{other}", models[f"G_{other}_to_{domain}"])):
            print(f"diagnostics {name}/{group}: {len(chosen[key])} images from {key}", flush=True)
            groups[group] = measure_group(models[name], plain[key], manifest["entries"][key], chosen[key], device, generator)
            groups[group]["source_split"] = key
        results[name] = groups
    if digest(checkpoint) != source_hash:
        raise ValueError("Source checkpoint changed while diagnosing; no report was published")
    report = {"created_utc": utc_now(), "kind": "read_only_discriminator_diagnostics",
              "checkpoint": str(checkpoint), "checkpoint_sha256": source_hash,
              "checkpoint_step": source_step, "manifest_fingerprint": manifest["manifest_fingerprint"],
              "source_checkpoint_unchanged": True, "seed": SEED, "max_images_per_split": LIMIT,
              "selection": {key: {"available_count": len(data[key]), "selected_count": len(chosen[key]),
                                   "indices": chosen[key], "seed": SEED + i,
                                   "all_images": len(chosen[key]) == len(data[key])}
                            for i, key in enumerate(chosen)},
              "inference_precision": "fp32", "augmentation": False, "training_updates": 0,
              "test_split_inference": False, "device": device, "environment": environment(),
              "elapsed_inference_seconds": time.perf_counter() - started,
              "score_definition": "Per-image mean of raw PatchGAN outputs; these LSGAN scores are not probabilities.",
              "mse_definition": "Mean squared error over individual patch outputs, followed by equal-weight summaries across images.",
              "interpretation": "Compare groups descriptively alongside generated images. No universal threshold or automatic overfitting conclusion is used.",
              "discriminators": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "task3_gan/srinidhi/config.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = diagnose(args.checkpoint, args.config, args.output, args.device)
    print(json.dumps({"output": str(args.output), "checkpoint_sha256": report["checkpoint_sha256"],
                      "elapsed_inference_seconds": report["elapsed_inference_seconds"],
                      "discriminators": {name: {group: {"count": entry["count"], "summary": entry["summary"]}
                                                    for group, entry in groups.items()}
                                         for name, groups in report["discriminators"].items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
