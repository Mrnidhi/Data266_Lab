"""Export a verified project CycleGAN and run the supplied Canvas evaluator.

Class A=Monet, B=Photo. All inputs are exported in alphabetical order. Metrics
use the unmodified notebook definitions on JPEGs with fixed quality/subsampling.
This does not train, select a model, or submit anything to Kaggle.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import PIL
from PIL import Image
import scipy
import torch
import torchvision
from lab1.cyclegan import Generator, ImageDomain, _pil, prepare_data

EVALUATOR_SHA256 = "702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef"
DIRECTIONS = (("monet_to_photo", "pred_A2B", "monet_jpg", "photo_jpg"),
              ("photo_to_monet", "pred_B2A", "photo_jpg", "monet_jpg"))
JPEG_SETTINGS = {"quality": 95, "subsampling": 0, "optimize": False}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "task3_gan/data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--concurrent-workloads-note", default="Other workloads were not assessed by this runner; runtime is not an isolated benchmark.")
    args = parser.parse_args()
    started = time.time()
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint does not match the expected project artifact")
    if sha256(args.notebook) != EVALUATOR_SHA256:
        raise ValueError("Notebook differs from the manually reviewed Canvas evaluator")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory to preserve earlier evidence")
    if args.eval_batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        # Bound this inference process's allocator while other training runs.
        torch.cuda.set_per_process_memory_fraction(0.10, device=device)
    # Hash was verified against a previously recorded project checkpoint before
    # loading its trusted optimizer/RNG-containing pickle payload.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["format_version"] == 1
    assert not saved["state"]["manifest_fingerprint"].startswith("synthetic:")
    assert set(saved["models"]) == {"G_photo_to_monet", "G_monet_to_photo", "D_photo", "D_monet"}
    cfg = saved["config"]
    assert cfg["image_size"] == 256
    for network, parameters in saved["models"].items():
        for name, value in parameters.items():
            if torch.is_tensor(value) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite checkpoint tensor: {network}/{name}")
    data_config = {**cfg, "data": {**cfg["data"],
        "monet_dir": str(args.data_root / "monet_jpg"),
        "photo_dir": str(args.data_root / "photo_jpg"),
        "manifest_dir": str(ROOT / "task3_gan/srinidhi/data_processed/part3_manifests")}}
    _, provenance = prepare_data(data_config)
    assert provenance["manifest_fingerprint"] == saved["state"]["manifest_fingerprint"]
    inputs = {domain: image_paths(args.data_root / domain) for domain in ("monet_jpg", "photo_jpg")}
    assert len(inputs["monet_jpg"]) == 300 and len(inputs["photo_jpg"]) == 7038
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.notebook, args.output / "supplied_evaluator.ipynb")
    result = {
        "status": "export_in_progress", "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": saved["state"]["global_step"],
        "checkpoint_selection": saved["state"].get("best_selection"),
        "manifest_fingerprint": provenance["manifest_fingerprint"],
        "notebook_sha256": EVALUATOR_SHA256, "runner_sha256": sha256(__file__),
        "n_eval": 300, "evaluation_batch_size": args.eval_batch_size,
        "inference_batch_size": 1, "inference_precision": "FP32, eval mode, no gradients",
        "jpeg_settings": JPEG_SETTINGS,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "scipy": scipy.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "concurrent_workloads": args.concurrent_workloads_note,
        "is_held_out_evaluation": False,
        "evaluation_note": "Full supplied class references include training images. Preserve separate held-out reports.",
        "kaggle_submitted": False, "official_rank": None,
        "directions": {},
    }
    save_json(args.output / "metrics.json", result)
    log = (args.output / "run_log.txt").open("a", encoding="utf-8")

    def report(message):
        line = f"[{time.time() - started:.1f}s] {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    generator_states = {key: value for key, value in saved["models"].items() if key.startswith("G_")}
    del saved
    gc.collect()
    for direction, label, source_domain, target_domain in DIRECTIONS:
        png_dir, jpeg_dir = args.output / "lossless_png" / label, args.output / label
        png_dir.mkdir(parents=True)
        jpeg_dir.mkdir()
        sources = inputs[source_domain]
        (args.output / f"{source_domain}_inference_manifest.txt").write_text(
            "\n".join(p.name for p in sources) + "\n", encoding="utf-8")
        model = Generator(cfg["base_channels"], cfg["residual_blocks"]).to(device).eval()
        model.load_state_dict(generator_states[f"G_{direction}"], strict=True)
        dataset = ImageDomain(sources, cfg["image_size"], augment=False)
        records = []
        report(f"Exporting {direction}: {len(sources)} inputs")
        with torch.no_grad():
            for i, source in enumerate(sources):
                tensor = model(dataset[i].unsqueeze(0).to(device))[0]
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite inference output: {direction}/{source.name}")
                output = _pil(tensor)
                png = png_dir / f"{i:06d}.png"
                jpeg = jpeg_dir / f"{i:06d}.jpg"
                output.save(png)
                output.save(jpeg, format="JPEG", **JPEG_SETTINGS)
                with Image.open(jpeg) as check:
                    check.load()
                    assert check.mode == "RGB" and check.size == (256, 256)
                records.append({"source": source.name, "source_sha256": sha256(source),
                    "png": png.name, "png_sha256": sha256(png),
                    "jpeg": jpeg.name, "jpeg_sha256": sha256(jpeg)})
                if (i + 1) % 250 == 0 or i + 1 == len(sources):
                    report(f"{direction}: {i + 1}/{len(sources)} exported")
        assert len(image_paths(jpeg_dir)) == len(sources)
        save_json(args.output / f"{label}_export_manifest.json", {
            "direction": direction, "class_label": label, "checkpoint_sha256": checkpoint_hash,
            "count": len(sources), "source_order": "alphabetical source filename",
            "jpeg_settings": JPEG_SETTINGS, "images": records})
        result["directions"][direction] = {"class_label": label, "export_count": len(sources)}
        save_json(args.output / "metrics.json", result)
        del tensor, model, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del generator_states
    gc.collect()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    ns = {"__name__": "canvas_metric_definitions", "device": device}
    for idx in (2, 4, 5, 6, 7):
        exec(compile("".join(notebook["cells"][idx]["source"]),
                     f"supplied_evaluator.ipynb:cell{idx}", "exec"), ns)
    result["status"] = "evaluation_in_progress"
    save_json(args.output / "metrics.json", result)
    for direction, label, source_domain, target_domain in DIRECTIONS:
        real = ns["take_n"](ns["list_images"](str(args.data_root / target_domain)), 300)
        generated = ns["take_n"](ns["list_images"](str(args.output / label)), 300)
        assert len(real) == len(generated) == 300
        report(f"Evaluating final JPEGs for {direction}: 300 real and 300 generated")
        fid, mifid = ns["calculate_fid_mifid"](real, generated, batch_size=args.eval_batch_size)
        assert np.isfinite(fid) and np.isfinite(mifid)
        result["directions"][direction].update({"FID": fid, "MiFID": mifid,
            "evaluated_real_count": len(real), "evaluated_generated_count": len(generated),
            "evaluated_real_filenames": [Path(p).name for p in real],
            "evaluated_generated_filenames": [Path(p).name for p in generated]})
        report(f"{direction}: FID={fid:.8f}, MiFID={mifid:.8f}")
        save_json(args.output / "metrics.json", result)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["directional_means"] = {
        metric: sum(row[metric] for row in result["directions"].values()) / 2
        for metric in ("FID", "MiFID")}
    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "FID", "MiFID"])
        writer.writeheader()
        writer.writerow({"ID": 1, **result["directional_means"]})
    result.update(status="complete_local_class_evaluation", elapsed_seconds=time.time() - started,
                  submission_csv_sha256=sha256(args.output / "submission.csv"))
    if device.type == "cuda":
        result["peak_process_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_process_reserved_gpu_bytes"] = torch.cuda.max_memory_reserved(device)
    save_json(args.output / "metrics.json", result)
    report("COMPLETE: " + json.dumps(result["directional_means"]))
    log.close()


if __name__ == "__main__":
    main()
