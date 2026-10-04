"""Warm-start CycleGAN fine-tuning for the class leaderboard, then export and score.

Train:   .venv/bin/python scripts/finetune_part3_push.py train --init CKPT --output DIR [recipe flags]
Resume:  .venv/bin/python scripts/finetune_part3_push.py train --init CKPT --output DIR --resume [same flags]
Export:  .venv/bin/python scripts/finetune_part3_push.py export --checkpoint DIR/best.pt --output EXPORT_DIR

Export writes pred_A2B (all 300 Monet -> Photo) and pred_B2A (all photos -> Monet)
as 256x256 RGB JPEGs named in alphabetical source order, then runs the supplied
notebook's unmodified metric cells to produce submission.csv. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVALUATOR_SHA256 = "702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef"
DEFAULT_NOTEBOOK = ROOT / "reproducibility/packages/part3-20261002/Part3_Evaluation_Script.ipynb"
DEFAULT_DATA = ROOT / "task3_gan/data"
DEFAULT_MANIFESTS = ROOT / "task3_gan/srinidhi/data_processed/part3_manifests"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def add_recipe_arguments(parser: argparse.ArgumentParser) -> None:
    from lab1.cyclegan_push import PushConfig

    for field in fields(PushConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(field.default), default=field.default
        )


def recipe_from(args: argparse.Namespace):
    from lab1.cyclegan_push import PushConfig

    return PushConfig(**{field.name: getattr(args, field.name) for field in fields(PushConfig)})


def train(args: argparse.Namespace) -> None:
    from lab1 import cyclegan_push as cp

    push, device = recipe_from(args), resolve_device(args.device)
    scorer = (
        None
        if args.no_score
        else cp.ClassScorer(args.data_root, device, 256, push.jpeg_quality, push.jpeg_subsampling)
    )
    print(
        json.dumps({"device": device, "recipe": asdict(push), "init": str(args.init)}, indent=2),
        flush=True,
    )
    summary = cp.run(
        push,
        args.init,
        args.output,
        device,
        args.data_root,
        args.manifest_dir,
        scorer=scorer,
        resume=args.resume,
        stop_after=args.stop_after,
        stop_file=args.stop_file,
    )
    print(json.dumps(summary, indent=2), flush=True)


def export(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from lab1 import cyclegan as cg
    from lab1 import cyclegan_push as cp

    if sha256(args.notebook) != EVALUATOR_SHA256:
        raise ValueError("Notebook differs from the supplied Canvas evaluator")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty export directory")
    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, push = saved["config"], saved.get("push_config", {})
    quality = int(push.get("jpeg_quality", 95))
    subsampling = int(push.get("jpeg_subsampling", 0))
    output.mkdir(parents=True)
    started, results = time.time(), {}
    inputs = {
        domain: cp.image_paths(Path(args.data_root) / domain)
        for domain in ("monet_jpg", "photo_jpg")
    }
    if args.max_photos is not None:
        # The supplied evaluator only reads the first 300 sorted files of pred_B2A.
        inputs["photo_jpg"] = inputs["photo_jpg"][: max(300, args.max_photos)]
    for direction, label, source in (
        ("monet_to_photo", "pred_A2B", "monet_jpg"),
        ("photo_to_monet", "pred_B2A", "photo_jpg"),
    ):
        generator = cg.Generator(config.get("base_channels", 64), config.get("residual_blocks", 9))
        generator.load_state_dict(saved["models"][f"G_{direction}"], strict=True)
        generator = generator.to(device).eval()
        folder = output / label
        folder.mkdir()
        records = []
        with torch.no_grad():
            for index, path in enumerate(inputs[source]):
                tensor = generator(cp.load_image(path, 256, None).unsqueeze(0).to(device))[
                    0
                ].float()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite output for {path.name}")
                target = folder / f"{index:06d}.jpg"
                cg._pil(tensor).save(
                    target, format="JPEG", quality=quality, subsampling=subsampling, optimize=False
                )
                with Image.open(target) as check:
                    assert check.mode == "RGB" and check.size == (256, 256)
                records.append(
                    {"source": path.name, "jpeg": target.name, "jpeg_sha256": sha256(target)}
                )
        (output / f"{label}_export_manifest.json").write_text(
            json.dumps(
                {
                    "direction": direction,
                    "count": len(records),
                    "checkpoint_sha256": sha256(args.checkpoint),
                    "jpeg_settings": {
                        "quality": quality,
                        "subsampling": subsampling,
                        "optimize": False,
                    },
                    "source_order": "alphabetical source filename",
                    "images": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"exported {label}: {len(records)} images", flush=True)
    try:
        import tqdm  # noqa: F401  (imported by the supplied notebook)
    except ImportError:
        import types

        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda iterable, **kwargs: iterable
        sys.modules["tqdm"] = stub
    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    namespace = {
        "__name__": "supplied_evaluator",
        "device": torch.device("cuda" if device.startswith("cuda") else "cpu"),
    }
    for index in (2, 4, 5, 6, 7):
        exec(
            compile(
                "".join(notebook["cells"][index]["source"]),
                f"supplied_evaluator.ipynb:cell{index}",
                "exec",
            ),
            namespace,
        )
    for direction, label, target in (
        ("monet_to_photo", "pred_A2B", "photo_jpg"),
        ("photo_to_monet", "pred_B2A", "monet_jpg"),
    ):
        real = namespace["take_n"](
            namespace["list_images"](str(Path(args.data_root) / target)), 300
        )
        generated = namespace["take_n"](namespace["list_images"](str(output / label)), 300)
        fid, mifid = namespace["calculate_fid_mifid"](real, generated, batch_size=8)
        results[direction] = {"class_label": label, "FID": fid, "MiFID": mifid}
        print(f"{direction}: FID={fid:.6f} MiFID={mifid:.6f}", flush=True)
    means = {
        metric: sum(row[metric] for row in results.values()) / 2 for metric in ("FID", "MiFID")
    }
    with (output / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "FID", "MiFID"])
        writer.writeheader()
        writer.writerow({"ID": 1, **means})
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "notebook_sha256": EVALUATOR_SHA256,
        "n_eval": 300,
        "device": device,
        "directions": results,
        "directional_means": means,
        "local_composite": (means["FID"] + means["MiFID"]) / 2,
        "selection_note": saved.get("selection_note"),
        "class_scores_during_training": saved.get("class_scores"),
        "elapsed_seconds": time.time() - started,
        "kaggle_submitted": False,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print("COMPLETE " + json.dumps(means), flush=True)


def check(args: argparse.Namespace) -> None:
    """Report FID, KID, content similarity and nearest-training-image diagnostics.

    Photos 301-600 are outside the class scorer's subset, but may be training data.
    """
    import numpy as np
    import torch
    from lab1 import cyclegan as cg
    from lab1 import cyclegan_push as cp

    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, push = saved["config"], saved.get("push_config", {})
    quality, subsampling = int(push.get("jpeg_quality", 95)), int(push.get("jpeg_subsampling", 0))
    generators = {}
    for name in ("G_monet_to_photo", "G_photo_to_monet"):
        generator = cg.Generator(config.get("base_channels", 64), config.get("residual_blocks", 9))
        generator.load_state_dict(saved["models"][name], strict=True)
        generators[name] = generator.to(device).eval()
    features = cp.InceptionFeatures(device)
    monet = cp.image_paths(Path(args.data_root) / "monet_jpg")[:300]
    photos = cp.image_paths(Path(args.data_root) / "photo_jpg")
    first, fresh = photos[:300], photos[300:600]
    real_monet = features([cp.read_rgb(p) for p in monet])
    real_first = features([cp.read_rgb(p) for p in first])
    real_fresh = features([cp.read_rgb(p) for p in fresh])
    translate = lambda name, paths: features(
        cp.translate_to_jpegs(generators[name], paths, 256, device, quality, subsampling)
    )
    a2b, b2a_first, b2a_fresh = (
        translate("G_monet_to_photo", monet),
        translate("G_photo_to_monet", first),
        translate("G_photo_to_monet", fresh),
    )
    nn_generated = cp.nearest_cosine(b2a_first, real_monet)
    nn_real = cp.nearest_cosine(real_monet, real_monet, exclude_self=True)
    report = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": device,
        "class_protocol": {
            "fid_a2b": cp.fid_mifid(real_first, a2b)[0],
            "fid_b2a": cp.fid_mifid(real_monet, b2a_first)[0],
            "kid_a2b": cp.kid(real_first, a2b),
            "kid_b2a": cp.kid(real_monet, b2a_first),
        },
        "fresh_sample": {
            "fid_a2b_vs_photos_301_600": cp.fid_mifid(real_fresh, a2b)[0],
            "fid_b2a_from_photos_301_600": cp.fid_mifid(real_monet, b2a_fresh)[0],
            "kid_a2b_vs_photos_301_600": cp.kid(real_fresh, a2b),
            "kid_b2a_from_photos_301_600": cp.kid(real_monet, b2a_fresh),
        },
        "content_cosine_input_vs_output": {
            "monet_to_photo": float(cp.paired_cosine(real_monet, a2b).mean()),
            "photo_to_monet": float(cp.paired_cosine(real_first, b2a_first).mean()),
        },
        "memorisation": {
            "generated_monet_nearest_training_cosine_max": float(nn_generated.max()),
            "generated_monet_nearest_training_cosine_median": float(np.median(nn_generated)),
            "real_monet_nearest_other_real_cosine_max": float(nn_real.max()),
            "real_monet_nearest_other_real_cosine_median": float(np.median(nn_real)),
            "outputs_closer_to_a_training_monet_than_any_real_pair": int(
                (nn_generated > nn_real.max()).sum()
            ),
        },
    }
    report["fresh_sample"]["composite_estimate"] = (
        (
            report["fresh_sample"]["fid_a2b_vs_photos_301_600"]
            + report["fresh_sample"]["fid_b2a_from_photos_301_600"]
        )
        / 2
        + 0.41
    ) / 2
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CHECK " + json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("--init", type=Path, required=True)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--device", default="auto")
    trainer.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    trainer.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    trainer.add_argument("--resume", action="store_true")
    trainer.add_argument("--no-score", action="store_true")
    trainer.add_argument("--stop-after", type=int, default=None)
    trainer.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="If this file appears, score, checkpoint and exit cleanly",
    )
    add_recipe_arguments(trainer)
    exporter = commands.add_parser("export")
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--device", default="auto")
    exporter.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    exporter.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    exporter.add_argument(
        "--max-photos",
        type=int,
        default=None,
        help="Smoke checks only: export the first N (>=300) photos instead of all",
    )
    checker = commands.add_parser("check")
    checker.add_argument("--checkpoint", type=Path, required=True)
    checker.add_argument("--output", type=Path, required=True)
    checker.add_argument("--device", default="auto")
    checker.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    {"train": train, "export": export, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
