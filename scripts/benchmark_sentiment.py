"""Time full-size Yelp models on real cached reviews; never produce final results."""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1 import sentiment as s
from lab1.common import environment, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--encoded-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    args = parser.parse_args()
    if args.steps < 20 or not torch.cuda.is_available():
        parser.error("CUDA and at least 20 measured steps required")
    cfg = json.loads((ROOT / "task2_sentiment/srinidhi/config.json").read_text())["full"]
    cfg.update(data_dir=args.data_dir, encoded_cache=args.encoded_cache)
    torch.set_num_threads(cfg["cpu_threads"])
    records = s._load_records(cfg)
    vocabulary, datasets = s._load_features(cfg, records)
    results = []
    for workers in args.workers:
        for name in s.MODEL_NAMES:
            s._seed(cfg["seed"])
            model = s.build_model(name, len(vocabulary), cfg).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rates"][name], weight_decay=cfg["weight_decay"])
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
            loader = s._loader(datasets["train"], cfg["batch_size"], True,
                               torch.Generator().manual_seed(cfg["seed"]), device="cuda", num_workers=workers)
            torch.cuda.reset_peak_memory_stats()
            for index, (ids, labels, _) in enumerate(itertools.islice(loader, args.steps + 10)):
                if index == 10:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    logits = s._forward(model, ids, "cuda")
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.cuda(non_blocking=True))
                if not torch.isfinite(loss):
                    raise FloatingPointError(name)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                loss.item()  # Match the synchronization performed by the training logger.
            torch.cuda.synchronize()
            train_seconds_per_batch = (time.perf_counter() - started) / args.steps
            loader = s._loader(datasets["validation"], cfg["batch_size"], device="cuda", num_workers=workers)
            model.eval()
            with torch.inference_mode():
                for index, (ids, labels, _) in enumerate(itertools.islice(loader, 25)):
                    if index == 5:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                    logits = s._forward(model, ids, "cuda")
                    nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.cuda(non_blocking=True)).item()
                    torch.sigmoid(logits.float()).cpu().numpy()
            torch.cuda.synchronize()
            eval_seconds_per_batch = (time.perf_counter() - started) / 20
            estimate = (math.ceil(len(datasets["train"]) / cfg["batch_size"]) * cfg["epochs"] * train_seconds_per_batch
                        + (math.ceil(len(datasets["validation"]) / cfg["batch_size"]) * cfg["epochs"]
                           + math.ceil(len(datasets["test"]) / cfg["batch_size"])) * eval_seconds_per_batch)
            row = {"model": name, "num_workers": workers, "measured_training_steps": args.steps,
                   "train_seconds_per_batch": train_seconds_per_batch,
                   "eval_seconds_per_batch": eval_seconds_per_batch,
                   "maximum_six_epoch_train_eval_seconds": estimate,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                   "parameter_count": sum(p.numel() for p in model.parameters())}
            results.append(row)
            print(json.dumps(row), flush=True)
            write_json(args.output, {"purpose": "timing only, not final training", "utc": utc_now(),
                                    "environment": environment(), "results": results,
                                    "estimate_excludes": ["setup", "data transfer", "checkpoint serialization", "bootstrap metrics", "plots and export"],
                                    "recommended_headroom_fraction": 0.25})
            del optimizer, model, loader
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
