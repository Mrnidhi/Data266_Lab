"""Independently implemented, character-level causal language model for Lab 1.

No pretrained tokenizer/model or prebuilt attention/Transformer layer is used.
The public entry point is run(config, output_dir, device, resume=None).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import resource
except ImportError:  # Windows GPU workstations do not provide the resource module.
    resource = None


SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD, UNK, BOS, EOS = range(4)
PROMPTS = [
    "Once upon a time, a little bird",
    "Lily found a small red box. Inside",
    "Tom wanted to help his friend, but",
    "One rainy day, the dog and the cat",
    "The little girl learned that sharing",
]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def build_vocabulary(stories: list[str]) -> dict[str, Any]:
    chars = sorted(set("".join(stories)))
    return {"tokens": SPECIALS + chars, "special_ids": dict(zip(SPECIALS, range(4))),
            "unit": "Unicode character", "built_from": "training stories only",
            "unseen_character_policy": "map to <UNK>; report counts"}


class StoryWindows(Dataset):
    """Each next-character/EOS target appears once per story in one epoch.

    Windows do not cross stories. Adjacent windows share a single input/target
    boundary character, but no target is scored twice. Final windows are padded.
    """

    def __init__(self, stories: list[str], vocabulary: dict[str, Any], context_length: int):
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.context_length = context_length
        self.stoi = {char: i for i, char in enumerate(vocabulary["tokens"])}
        dtype = np.uint16 if len(self.stoi) <= 65535 else np.uint32
        self.encoded: list[np.ndarray] = []
        self.windows: list[tuple[int, int]] = []
        self.unknown_characters = 0
        self.character_count = 0
        self.target_count = 0
        for story_index, story in enumerate(stories):
            values = [self.stoi.get(char, UNK) for char in story]
            self.unknown_characters += values.count(UNK)
            self.character_count += len(values)
            encoded = np.asarray([BOS, *values, EOS], dtype=dtype)
            self.encoded.append(encoded)
            self.target_count += len(encoded) - 1
            self.windows.extend((story_index, start)
                                for start in range(0, len(encoded) - 1, context_length))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        story_index, start = self.windows[index]
        sequence = self.encoded[story_index]
        count = min(self.context_length, len(sequence) - start - 1)
        x = torch.full((self.context_length,), PAD, dtype=torch.long)
        y = torch.full((self.context_length,), PAD, dtype=torch.long)
        x[:count] = torch.as_tensor(sequence[start:start + count].astype(np.int64))
        y[:count] = torch.as_tensor(sequence[start + 1:start + count + 1].astype(np.int64))
        return x, y


class ManualCausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, context_length: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("Embedding width must be divisible by head count")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(
            context_length, context_length, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        # Explicit matrix products, scaling, triangular mask, softmax and AV.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.float().masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        probabilities = self.attention_dropout(F.softmax(scores, dim=-1)).to(v.dtype)
        attended = (probabilities @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        width = config["embedding_dim"]
        self.norm1 = nn.LayerNorm(width)
        self.attention = ManualCausalAttention(width, config["heads"],
                                               config["context_length"], config["dropout"])
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, config["feedforward_dim"]),
                                          nn.GELU(),
                                          nn.Linear(config["feedforward_dim"], width),
                                          nn.Dropout(config["dropout"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class CharacterGPT(nn.Module):
    def __init__(self, vocabulary_size: int, config: dict[str, Any]):
        super().__init__()
        self.context_length = config["context_length"]
        width = config["embedding_dim"]
        self.token_embedding = nn.Embedding(vocabulary_size, width, padding_idx=PAD)
        self.position_embedding = nn.Embedding(self.context_length, width)
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config["layers"]))
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.size(1) > self.context_length:
            raise ValueError("Expected [batch, length] IDs within configured context length")
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.embedding_dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))


def _smoke_stories(count: int, split: str) -> list[str]:
    subjects = ["bird", "cat", "dog", "child", "rabbit"]
    return [f"A {subjects[i % len(subjects)]} saw a red ball. It shared the ball. "
            f"The friends were happy. {split} story {i}." for i in range(count)]


def _cache_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    fields = {key: config[key] for key in ["dataset", "dataset_revision", "seed", "train_stories",
                                          "validation_stories", "selection", "shuffle_buffer"]}
    folder = Path(config["data_cache"]) / ("gpt_" + _digest(fields)[:16])
    return folder / "train.jsonl", folder / "validation.jsonl", folder / "manifest.json"


def prepare_data(config: dict[str, Any], previous: dict[str, Any] | None = None
                 ) -> tuple[list[str], list[str], dict[str, Any]]:
    """Prepare/verify cached JSONL story splits without starting training.

    Rehearsal uses seeded bounded-buffer streaming, not a uniform whole-corpus
    sample. Full mode uses a seeded permutation of indexed official splits.
    Cache hits are verified against saved hashes and need no network.
    Set offline=true to forbid uncached downloads. Return train, validation,
    manifest; cache paths are available through data_cache_paths(config).
    """
    if config["dataset"] == "synthetic":
        train = _smoke_stories(config["train_stories"], "train")
        valid = _smoke_stories(config["validation_stories"], "valid")
        manifest = {"dataset": "synthetic smoke fixture", "seed": config["seed"],
                    "train": [{"index": i, "sha256": _text_hash(s)} for i, s in enumerate(train)],
                    "validation": [{"index": i, "sha256": _text_hash(s)} for i, s in enumerate(valid)]}
    else:
        train_cache, valid_cache, manifest_cache = _cache_paths(config)
        if all(p.exists() for p in [train_cache, valid_cache, manifest_cache]):
            manifest = json.loads(manifest_cache.read_text())
            if _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
                raise ValueError("Cached data manifest checksum failed")
            cached = []
            for split, path in [("train", train_cache), ("validation", valid_cache)]:
                records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                expected_count = config["train_stories"] if split == "train" else config["validation_stories"]
                if len(records) != len(manifest[split]) or len(records) != expected_count:
                    raise ValueError("Cached data count differs from manifest")
                for record, expected in zip(records, manifest[split]):
                    if record["index"] != expected["index"] or _text_hash(record["text"]) != expected["sha256"]:
                        raise ValueError("Cached story checksum/index failed")
                cached.append([record["text"] for record in records])
            train_hashes = {item["sha256"] for item in manifest["train"]}
            valid_hashes = {item["sha256"] for item in manifest["validation"]}
            if (len(train_hashes) != len(manifest["train"]) or
                    len(valid_hashes) != len(manifest["validation"]) or train_hashes & valid_hashes):
                raise ValueError("Cached splits contain duplicate or overlapping stories")
            if previous and manifest["manifest_sha256"] != previous["manifest_sha256"]:
                raise ValueError("Cached data differs from checkpoint manifest")
            return cached[0], cached[1], manifest
        if config.get("offline", False):
            raise FileNotFoundError(f"Offline data cache is incomplete: {manifest_cache.parent}")
        from datasets import load_dataset
        from huggingface_hub import HfApi

        repo = config["dataset"]
        revision = (previous or {}).get("resolved_revision")
        if not revision:
            revision = HfApi().dataset_info(repo, revision=config["dataset_revision"]).sha
        streaming = config["selection"] == "streaming_buffer_shuffle"
        source = load_dataset(repo, revision=revision, streaming=streaming)
        manifest = {"dataset": repo, "requested_revision": config["dataset_revision"],
                    "resolved_revision": revision, "seed": config["seed"],
                    "source_url": "https://huggingface.co/datasets/" + repo,
                    "normalization": "CRLF/CR to LF; strip outer whitespace",
                    "selection": config["selection"], "shuffle_buffer": config["shuffle_buffer"],
                    "source_fingerprints": {name: getattr(source[name], "_fingerprint", None)
                                            for name in ["train", "validation"]}}
        selected: dict[str, list[str]] = {}
        occupied: set[str] = set()
        for split_number, (split, count) in enumerate([
                ("train", config["train_stories"]),
                ("validation", config["validation_stories"]) ]):
            selected[split], manifest[split] = [], []
            rows = source[split]
            if streaming:
                indexed = rows.map(lambda row, idx: {"source_index": idx}, with_indices=True)
                shuffled = indexed.shuffle(seed=config["seed"] + split_number,
                                             buffer_size=config["shuffle_buffer"])
                candidates = ({"index": int(row["source_index"]), "text": row["text"]} for row in shuffled)
                if previous:
                    # Re-create the deterministic selection; exact row/hash checks below.
                    candidates = iter(candidates)
            elif previous:
                candidates = previous[split]
            else:
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], split_number]))
                candidates = ({"index": int(i)} for i in rng.permutation(len(rows)))
            for record in candidates:
                index = record["index"]
                story = _normalize(record["text"] if "text" in record else rows[index]["text"])
                checksum = _text_hash(story)
                if previous and not streaming and checksum != record["sha256"]:
                    raise ValueError(f"Dataset content mismatch for {split} row {index}")
                if not story or checksum in occupied:
                    if previous and not streaming:
                        raise ValueError("Saved split contains empty or duplicate stories")
                    continue
                selected[split].append(story)
                manifest[split].append({"index": index, "sha256": checksum})
                occupied.add(checksum)
                if len(selected[split]) == count:
                    break
            if len(selected[split]) != count:
                raise ValueError(f"Not enough distinct {split} stories: need {count}")
        train, valid = selected["train"], selected["validation"]
    manifest["train_story_count"] = len(train)
    manifest["validation_story_count"] = len(valid)
    manifest["manifest_sha256"] = _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if previous and manifest["manifest_sha256"] != previous["manifest_sha256"]:
        raise ValueError("Reconstructed data manifest differs from the checkpoint")
    if config["dataset"] != "synthetic":
        for split, stories, path in [("train", train, train_cache), ("validation", valid, valid_cache)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("w") as handle:
                for metadata, story in zip(manifest[split], stories):
                    handle.write(json.dumps({**metadata, "text": story}, ensure_ascii=False) + "\n")
            temporary.replace(path)
        _json(manifest_cache, manifest)
    return train, valid, manifest


def data_cache_paths(config: dict[str, Any]) -> dict[str, str]:
    train, validation, manifest = _cache_paths(config)
    return {"train": str(train), "validation": str(validation), "manifest": str(manifest)}


def _loader(dataset: StoryWindows, batch_size: int, seed: int, epoch: int = 0,
            start_batch: int = 0, shuffle: bool = False, workers: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist() if shuffle else list(range(len(dataset)))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    # A private generator prevents iterator construction from advancing dropout RNG.
    return DataLoader(dataset, batch_sampler=batches[start_batch:], num_workers=workers,
                      generator=torch.Generator().manual_seed(seed + epoch + 100000))


def _precision(config: dict[str, Any], device: torch.device):
    enabled = bool(config["mixed_precision"] and device.type == "cuda")
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype


def _autocast(enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if enabled else contextlib.nullcontext()


def _metric(loss_sum: float, tokens: int, correct: int) -> dict[str, Any]:
    loss = loss_sum / tokens if tokens else None
    return {"cross_entropy": loss,
            "perplexity": math.exp(loss) if loss is not None and loss < 700 else None,
            "bits_per_character": loss / math.log(2) if loss is not None else None,
            "next_character_accuracy": correct / tokens if tokens else None,
            "scored_targets_including_eos": tokens}


@torch.no_grad()
def evaluate(model: CharacterGPT, loader: DataLoader, device: torch.device,
             amp: bool = False, dtype: torch.dtype = torch.float16) -> dict[str, Any]:
    model.eval()
    loss_sum, tokens, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with _autocast(amp, dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=PAD, reduction="sum")
        mask = y.ne(PAD)
        loss_sum += float(loss.item())
        tokens += int(mask.sum().item())
        correct += int(((logits.argmax(-1) == y) & mask).sum().item())
    if not math.isfinite(loss_sum):
        raise FloatingPointError("Nonfinite validation loss")
    return _metric(loss_sum, tokens, correct)


def diversity(texts: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"tokenization": "lowercase regex words; exclude prompt"}
    for n in [1, 2, 3, 4]:
        grams = []
        for text in texts:
            words = re.findall(r"\w+(?:'\w+)?", text.lower())
            grams.extend(tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1)))
        ratio = len(set(grams)) / len(grams) if grams else None
        if n < 4:
            result[f"distinct_{n}"] = ratio
        else:
            result["repeated_4gram_fraction"] = 1 - ratio if ratio is not None else None
        result[f"total_{n}grams"] = len(grams)
    return result


@torch.no_grad()
def generate(model: CharacterGPT, vocabulary: dict[str, Any], prompt: str,
             max_new_characters: int, device: torch.device, seed: int,
             temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    tokens = vocabulary["tokens"]
    stoi = {char: i for i, char in enumerate(tokens)}
    sequence = [BOS] + [stoi.get(c, UNK) for c in prompt]
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    for _ in range(max_new_characters):
        x = torch.tensor([sequence[-model.context_length:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        logits[[PAD, UNK, BOS]] = float("-inf")
        if temperature == 0:
            next_id = int(logits.argmax().item())
        else:
            logits = logits / temperature
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < cutoff] = float("-inf")
            next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).item())
        sequence.append(next_id)
        if next_id == EOS:
            break
        generated.append(tokens[next_id])
    return "".join(generated)


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        for index, rng in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng.cpu(), index)


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    # These checkpoint files contain our optimizer/RNG state, not only weights.
    return torch.load(path, map_location=device, weights_only=False)


def _restore_scaler(scaler: Any, saved_state: dict[str, Any]) -> str:
    if not scaler.is_enabled():
        return "disabled_for_current_device_precision"
    if saved_state:
        scaler.load_state_dict(saved_state)
        return "restored"
    # CPU/BF16 runs save no FP16 scaler. On a new FP16 device, retain the
    # freshly initialized scaler; cross-device bitwise equality is not claimed.
    return "fresh_scaler_no_saved_fp16_state"


def _empty_accumulator() -> dict[str, Any]:
    return {"loss_sum": 0.0, "tokens": 0, "correct": 0, "grad_norm_sum": 0.0,
            "batches": 0, "training_seconds": 0.0}


def _memory(device: torch.device) -> dict[str, Any]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    return {"host_peak_rss_mb": rss / (1024 ** 2 if sys.platform == "darwin" else 1024) if rss is not None else None,
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else None}


def _curves(history: list[dict[str, Any]], directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = [row["epoch"] + row["epoch_fraction"] - 1 for row in history]
    for ax, (key, title) in zip(axes.flat, [
            ("cross_entropy", "Cross-entropy"), ("perplexity", "Character perplexity"),
            ("bits_per_character", "Bits per character"),
            ("next_character_accuracy", "Next-character accuracy")]):
        for split in ["train", "validation"]:
            ax.plot(x, [row[split][key] for row in history], marker="o", label=split)
        ax.set(xlabel="Completed epoch equivalents", ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(x, [r["mean_gradient_norm_before_clipping"] for r in history], marker="o")
    axes[0].set(xlabel="Completed epoch equivalents", ylabel="Mean gradient norm")
    axes[1].plot(x, [r["training_targets_per_second"] for r in history], marker="o")
    axes[1].set(xlabel="Completed epoch equivalents", ylabel="Training targets / second")
    fig.tight_layout()
    fig.savefig(directory / "training_diagnostics.png", dpi=160)
    plt.close(fig)


def run(config: dict, output_dir: Path, device: str, resume: Path | None = None) -> dict:
    """Train/evaluate one explicitly labeled mode; no cloud resources are created."""
    config = dict(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if not resume and metrics_path.exists():
        raise FileExistsError("Output already contains training metrics; use a new directory or resume")
    started = time.perf_counter()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing a silent CPU fallback")
    if config["mode"] == "full":
        if config["epochs"] < 10 or config["train_stories"] < 100000 or config["validation_stories"] < 10000:
            raise ValueError("Full mode requires >=10 epochs, 100K train stories and 10K validation stories")
        if config.get("max_steps") is not None or config["dataset"] == "synthetic":
            raise ValueError("Full mode must use actual data with no step cap")
    seed_everything(config["seed"], config["deterministic"])
    if config.get("cpu_threads"):
        torch.set_num_threads(config["cpu_threads"])
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    checkpoint = load_checkpoint(Path(resume), "cpu") if resume else None
    if checkpoint and checkpoint["config"] != config:
        raise ValueError("Resume config must exactly match saved config (including schedule and data)")
    train_stories, valid_stories, manifest = prepare_data(config, checkpoint["data_manifest"] if checkpoint else None)
    vocabulary = build_vocabulary(train_stories)
    if checkpoint and checkpoint["vocabulary"] != vocabulary:
        raise ValueError("Rebuilt character vocabulary differs from saved vocabulary")
    _json(output_dir / "config.json", config)
    _json(output_dir / "data_manifest.json", manifest)
    _json(output_dir / "vocabulary.json", vocabulary)
    train_data = StoryWindows(train_stories, vocabulary, config["context_length"])
    valid_data = StoryWindows(valid_stories, vocabulary, config["context_length"])
    batches_per_epoch = math.ceil(len(train_data) / config["batch_size"])
    total_steps = batches_per_epoch * config["epochs"]
    planned_steps = min(total_steps, config["max_steps"]) if config.get("max_steps") else total_steps
    warmup_steps = max(1, int(planned_steps * config["warmup_fraction"]))
    model = CharacterGPT(len(vocabulary["tokens"]), config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  betas=tuple(config["adam_betas"]), weight_decay=config["weight_decay"])

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, planned_steps - warmup_steps))
        floor = config["minimum_lr_ratio"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp, amp_dtype = _precision(config, target)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    resume_scaler_status = "not_resumed"
    state = {"epoch": 0, "next_batch": 0, "global_step": 0, "best_validation_loss": float("inf"),
             "accumulator": _empty_accumulator(), "history": [], "total_training_seconds": 0.0,
             "total_training_targets": 0, "elapsed_seconds_before_resume": 0.0,
             "evaluation_pending": False}
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    optimizer_state[key] = value.to(target)
        scheduler.load_state_dict(checkpoint["scheduler"])
        resume_scaler_status = _restore_scaler(scaler, checkpoint["scaler"])
        state = checkpoint["progress"]
        _restore_rng(checkpoint["rng"])

    def save(path: Path) -> None:
        progress = dict(state)
        progress["elapsed_seconds_before_resume"] = (
            state["elapsed_seconds_before_resume"] + time.perf_counter() - started)
        _save_checkpoint(path, {"format_version": 1, "task": "character_gpt", "config": config,
                               "vocabulary": vocabulary, "data_manifest": manifest,
                               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                               "rng": _rng_state(), "progress": progress})

    # Preserve a source run's best checkpoint when resuming into a new directory.
    best_path = output_dir / "checkpoints" / "best.pt"
    if checkpoint and not best_path.exists():
        previous_best = Path(resume).parent / "best.pt"
        if previous_best.exists():
            import shutil
            best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous_best, best_path)
    validation_loader = _loader(valid_data, config["batch_size"], config["seed"], workers=config["num_workers"])
    while state["epoch"] < config["epochs"] and (state["global_step"] < planned_steps or state["evaluation_pending"]):
        epoch = state["epoch"]
        first_batch = state["next_batch"]
        loader = _loader(train_data, config["batch_size"], config["seed"], epoch,
                         first_batch, True, config["num_workers"])
        model.train()
        for batch_index, (x, y) in enumerate(loader, start=first_batch):
            if state["global_step"] >= planned_steps:
                break
            batch_started = time.perf_counter()
            x, y = x.to(target), y.to(target)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp, amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {state['global_step']}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"],
                                                       error_if_nonfinite=True)
            used_lr = optimizer.param_groups[0]["lr"]
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            mask = y.ne(PAD)
            tokens = int(mask.sum().item())
            correct = int(((logits.argmax(-1) == y) & mask).sum().item())
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            seconds = time.perf_counter() - batch_started
            accumulator = state["accumulator"]
            accumulator["loss_sum"] += float(loss.item()) * tokens
            accumulator["tokens"] += tokens
            accumulator["correct"] += correct
            accumulator["grad_norm_sum"] += float(grad_norm.item())
            accumulator["batches"] += 1
            accumulator["training_seconds"] += seconds
            state["global_step"] += 1
            state["evaluation_pending"] = True
            state["next_batch"] = batch_index + 1
            state["total_training_seconds"] += seconds
            state["total_training_targets"] += tokens
            if state["global_step"] % config["log_every_steps"] == 0:
                event = {"kind": "step", "mode": config["mode"], "epoch": epoch + 1,
                         "step": state["global_step"], "loss": float(loss.item()),
                         "gradient_norm_before_clipping": float(grad_norm.item()), "learning_rate": used_lr,
                         "targets_per_second": tokens / max(seconds, 1e-9), **_memory(target)}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
            if state["global_step"] % config["checkpoint_every_steps"] == 0:
                save(output_dir / "checkpoints" / "last.pt")
            if state["global_step"] >= planned_steps:
                break
        accumulator = state["accumulator"]
        validation = evaluate(model, validation_loader, target, amp, amp_dtype)
        training = _metric(accumulator["loss_sum"], accumulator["tokens"], accumulator["correct"])
        complete_epoch = state["next_batch"] == batches_per_epoch
        if complete_epoch and accumulator["tokens"] != train_data.target_count:
            raise AssertionError("Full epoch did not cover every training target exactly once")
        row = {"kind": "epoch", "mode": config["mode"], "epoch": epoch + 1,
               "epoch_complete": complete_epoch, "epoch_fraction": state["next_batch"] / batches_per_epoch,
               "step": state["global_step"], "train": training, "validation": validation,
               "generalization_gap_cross_entropy": validation["cross_entropy"] - training["cross_entropy"],
               "gap_definition": "validation eval-mode CE minus online training-mode CE",
               "mean_gradient_norm_before_clipping": accumulator["grad_norm_sum"] / accumulator["batches"],
               "training_targets_per_second": accumulator["tokens"] / max(accumulator["training_seconds"], 1e-9),
               "learning_rate_next_step": optimizer.param_groups[0]["lr"], **_memory(target)}
        state["history"].append(row)
        state["evaluation_pending"] = False
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        improved = validation["cross_entropy"] < state["best_validation_loss"]
        if improved:
            state["best_validation_loss"] = validation["cross_entropy"]
        if complete_epoch:
            state["epoch"] += 1
            state["next_batch"] = 0
            state["accumulator"] = _empty_accumulator()
        if improved:
            save(best_path)
        save(output_dir / "checkpoints" / "last.pt")
    if not best_path.exists():
        raise RuntimeError("No evaluated checkpoint is available")
    selected = load_checkpoint(best_path, "cpu")
    model.load_state_dict(selected["model"])
    model.eval()
    # Reload into a new instance and compare logits, not only state-dict keys.
    reloaded = CharacterGPT(len(vocabulary["tokens"]), config).to(target)
    reloaded.load_state_dict(selected["model"])
    reloaded.eval()
    probe = valid_data[0][0].unsqueeze(0).to(target)
    with torch.no_grad():
        reload_match = torch.equal(model(probe), reloaded(probe))
    if not reload_match:
        raise AssertionError("Checkpoint inference logits failed exact same-device reload check")
    del reloaded
    generations = []
    for prompt_index, prompt in enumerate(PROMPTS[:config["generation_prompts"]]):
        for decoding_index, (name, temperature, top_k) in enumerate([
                ("greedy", 0.0, 0), ("sampled", config["temperature"], config["top_k"]) ]):
            generation_seed = config["seed"] + 1000 + prompt_index * 10 + decoding_index
            continuation = generate(model, vocabulary, prompt, config["generation_characters"], target,
                                    generation_seed, temperature, top_k)
            generations.append({"prompt": prompt, "continuation": continuation,
                                "full_text": prompt + continuation, "decoding": name,
                                "temperature": temperature, "top_k": top_k, "seed": generation_seed,
                                "metrics": diversity([continuation])})
    _json(output_dir / "generations.json", generations)
    _json(output_dir / "history.json", state["history"])
    _curves(state["history"], output_dir / "figures")
    failure_path = output_dir / "failure_analysis.md"
    if not failure_path.exists():
        failure_path.write_text(
            "# Student review required: three real failure cases\n\n"
            "The program has not judged these outputs or claimed these are failures. "
            "Inspect generations.json, select three actual failures (generate more if needed), "
            "and replace every placeholder below with your own analysis.\n\n" +
            "\n\n".join(f"## Case {i}\n\n- Generation/prompt ID: [select a real output]\n"
                           "- Exact generated excerpt: [quote the observed failure]\n"
                           "- What failed: [your observation]\n"
                           "- Likely cause and supporting evidence: [your reasoning]\n"
                           "- Possible improvement: [your proposal]" for i in range(1, 4)) + "\n")
    completed_epochs = state["epoch"]
    summary = {"task": "character_gpt", "mode": config["mode"],
               "is_final_training_run": config["mode"] == "full" and completed_epochs == config["epochs"],
               "status": "training_complete_student_failure_analysis_pending" if config["mode"] == "full"
               else "limited_rehearsal_not_final_results",
               "dataset": manifest["dataset"], "manifest_sha256": manifest["manifest_sha256"],
               "train_stories": len(train_stories), "validation_stories": len(valid_stories),
               "train_windows_per_epoch": len(train_data), "train_targets_per_epoch": train_data.target_count,
               "validation_windows": len(valid_data), "vocabulary_size": len(vocabulary["tokens"]),
               "validation_unknown_characters": valid_data.unknown_characters,
               "validation_character_count": valid_data.character_count,
               "metric_note": "Character metrics include EOS targets and map unseen validation characters to UNK",
               "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(target),
               "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "cuda_version": torch.version.cuda, "mixed_precision": amp,
               "precision_dtype": str(amp_dtype) if amp else "torch.float32",
               "resume_scaler_status": resume_scaler_status,
               "completed_full_epochs": completed_epochs, "global_steps": state["global_step"],
               "planned_steps": planned_steps, "best_validation_cross_entropy": state["best_validation_loss"],
               "best_checkpoint_epoch": selected["progress"]["history"][-1]["epoch"],
               "best_validation": selected["progress"]["history"][-1]["validation"],
               "last_epoch_metrics": state["history"][-1],
               "training_targets_per_second": state["total_training_targets"] / max(state["total_training_seconds"], 1e-9),
               "total_training_seconds": state["total_training_seconds"],
               "throughput_scope": "training step including device transfer; excludes loading, validation and checkpoint I/O",
               "elapsed_seconds_including_prior_sessions": state["elapsed_seconds_before_resume"] + time.perf_counter() - started,
               "inference_reload_verified": reload_match,
               "generation_metrics": {name: diversity([g["continuation"] for g in generations if g["decoding"] == name])
                                      for name in ["greedy", "sampled"]},
               "failure_analysis_status": "requires_student_review",
               "artifacts": {"best_checkpoint": str(best_path),
                             "last_checkpoint": str(output_dir / "checkpoints" / "last.pt"),
                             "generations": str(output_dir / "generations.json"),
                             "failure_analysis": str(failure_path)}, **_memory(target)}
    _json(output_dir / "summary.json", summary)
    return summary
