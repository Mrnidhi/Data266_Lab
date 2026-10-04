"""Warm-start CycleGAN fine-tuning aimed at the supplied class FID/MiFID evaluator.

The assignment architecture is unchanged: two 9-block ResNet generators, one
discriminator per domain, least-squares adversarial loss and L1 cycle-consistency
(plus optional identity) loss. Options, each off by default:

* Multi-scale discriminators: scale 0 is the warm-started 70px PatchGAN; extra
  PatchGANs judge 2x/4x average-pooled images for global realism
  (pix2pixHD, Wang et al. 2018; SPatchGAN, Shao et al. 2021).
* DiffAugment on every discriminator input, including the generator's
  adversarial path (Zhao et al. 2020) for the 300-image Monet domain.
* Linearly ramped cycle/identity weights ("CycleGAN with Better Cycles").
* Generator EMA for scoring/export (Yazici et al. 2019).

No pretrained network enters training. Inception only scores exported JPEGs,
exactly as the supplied notebook does; checkpoint selection on that score is
recorded so the report can state it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from . import cyclegan as cg
from .cyclegan_ema import GeneratorEMA

TRAINER = "cyclegan_push_v1"
GENERATORS = ("G_photo_to_monet", "G_monet_to_photo")
DISCRIMINATORS = ("D_photo", "D_monet")
EXTENSIONS = (".jpg", ".jpeg", ".png")
N_EVAL = 300


@dataclass(frozen=True)
class PushConfig:
    steps: int = 56_240
    learning_rate: float = 5e-5
    beta1: float = 0.5
    beta2: float = 0.999
    constant_fraction: float = 0.5
    cycle_start: float = 10.0
    cycle_end: float = 10.0
    identity_start: float = 5.0
    identity_end: float = 5.0
    weight_ramp_steps: int = 0
    extra_scales: int = 0
    d_warmup_steps: int = 0
    aug_monet: str = ""
    aug_photo: str = ""
    flip_equivariance: float = 0.0
    ema_beta: float = 0.999
    ema_slow_beta: float = 0.0
    replay_size: int = 50
    eval_every: int = 2_812
    checkpoint_every: int = 1_000
    log_every: int = 50
    precision: str = "fp32"
    seed: int = 2342
    data: str = "all"
    jpeg_quality: int = 95
    jpeg_subsampling: int = 0
    eval_variants: str = "ema"
    train_view: str = "resize286"
    batch_size: int = 1
    cycle_lowpass: int = 0
    cycle_detail: float = 1.0
    cycle_photo_scale: float = 1.0
    cycle_monet_scale: float = 1.0
    nce_weight: float = 0.0
    nce_idt_weight: float = 0.0
    nce_layers: str = "0,4,8,12,16"
    nce_patches: int = 256
    nce_tau: float = 0.07
    r1_gamma: float = 0.0
    r1_every: int = 4
    d_noise_start: float = 0.0
    d_noise_end: float = 0.0
    prefetch: int = 3
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("steps", "eval_every", "checkpoint_every", "log_every", "batch_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("weight_ramp_steps", "extra_scales", "d_warmup_steps", "replay_size", "prefetch", "warmup_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("learning_rate", "cycle_start", "cycle_end", "identity_start", "identity_end", "flip_equivariance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0 or min(self.cycle_start, self.cycle_end) <= 0:
            raise ValueError("Learning rate and cycle-consistency weight must stay positive")
        for name in ("beta1", "beta2", "ema_beta", "constant_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.extra_scales > 3:
            raise ValueError("At most three extra discriminator scales are supported at 256px")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if self.data not in ("all", "train_split"):
            raise ValueError("data must be 'all' or 'train_split'")
        if not 1 <= self.jpeg_quality <= 100 or self.jpeg_subsampling not in (0, 1, 2):
            raise ValueError("Invalid JPEG export settings")
        variants = parse_variants(self.eval_variants)
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("eval_variants must list 'ema', 'ema_slow' and/or 'raw' once each")
        if not (self.ema_slow_beta == 0 or 0 < self.ema_slow_beta < 1):
            raise ValueError("ema_slow_beta must be 0 (off) or between 0 and 1")
        if "ema_slow" in variants and self.ema_slow_beta == 0:
            raise ValueError("Scoring 'ema_slow' needs ema_slow_beta > 0")
        if self.train_view not in ("resize286", "native"):
            raise ValueError("train_view must be 'resize286' or 'native'")
        if self.cycle_lowpass not in (0, 2, 4, 8, 16) or not 0 <= self.cycle_detail <= 1:
            raise ValueError("cycle_lowpass must be 0/2/4/8/16 and cycle_detail in [0, 1]")
        if self.cycle_lowpass == 0 and self.cycle_detail != 1:
            raise ValueError("cycle_detail < 1 needs cycle_lowpass > 0")
        if min(self.cycle_photo_scale, self.cycle_monet_scale) <= 0 or not all(
                math.isfinite(v) for v in (self.cycle_photo_scale, self.cycle_monet_scale)):
            raise ValueError("cycle scales must be finite and positive (cycle-consistency stays in both directions)")
        for name in ("nce_weight", "nce_idt_weight", "nce_tau", "r1_gamma", "d_noise_start", "d_noise_end"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.nce_tau == 0 or not isinstance(self.nce_patches, int) or self.nce_patches < 2:
            raise ValueError("nce_tau must be positive and nce_patches at least 2")
        if not isinstance(self.r1_every, int) or self.r1_every < 1:
            raise ValueError("r1_every must be a positive integer")
        if self.nce_idt_weight > 0 and self.nce_weight == 0:
            raise ValueError("nce_idt_weight needs nce_weight > 0")
        parse_layers(self.nce_layers)
        parse_policy(self.aug_monet)
        parse_policy(self.aug_photo)


# DiffAugment, adapted from mit-han-lab/data-efficient-gans DiffAugment_pytorch.py
# (Zhao et al., NeurIPS 2020). Images are in [-1, 1]; offsets use the torch RNG.
def _brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def _contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def _translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(x.size(0), device=x.device), torch.arange(x.size(2), device=x.device),
                                torch.arange(x.size(3), device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return padded.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(x.size(0), device=x.device), torch.arange(size[0], device=x.device),
                                torch.arange(size[1], device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - size[0] // 2, min=0, max=x.size(2) - 1)
    gy = torch.clamp(gy + oy - size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


AUGMENTATIONS = {"color": (_brightness, _saturation, _contrast), "translation": (_translation,), "cutout": (_cutout,)}


def parse_variants(variants: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in variants.split(",") if part.strip())
    if set(names) - {"ema", "ema_slow", "raw"}:
        raise ValueError(f"Unknown evaluation variant(s): {sorted(set(names) - {'ema', 'ema_slow', 'raw'})}")
    return names


def parse_policy(policy: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in policy.split(",") if part.strip())
    unknown = set(names) - set(AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unknown DiffAugment operation(s): {sorted(unknown)}")
    return names


def diff_augment(x: torch.Tensor, policy: str) -> torch.Tensor:
    for name in parse_policy(policy):
        for operation in AUGMENTATIONS[name]:
            x = operation(x)
    return x


class FastGeneratorEMA(GeneratorEMA):
    """GeneratorEMA with the same arithmetic (avg * beta + raw * (1 - beta)) in multi-tensor
    kernels: one launch per generator instead of two per parameter, which matters on Windows."""

    @torch.no_grad()
    def update(self, models: dict[str, nn.Module]) -> None:
        for name, averaged in self.models.items():
            targets = list(averaged.parameters())
            sources = [parameter.detach().float() for parameter in models[name].parameters()]
            torch._foreach_mul_(targets, self.beta)
            torch._foreach_add_(targets, sources, alpha=1 - self.beta)
            for buffer, raw_buffer in zip(averaged.buffers(), models[name].buffers()):
                buffer.copy_(raw_buffer)
        self.updates += 1


class MultiScaleDiscriminator(nn.Module):
    """Scale 0 is the original PatchGAN; scale k sees the input average-pooled k times."""

    def __init__(self, base: cg.PatchDiscriminator, extra_scales: int, base_channels: int = 64) -> None:
        super().__init__()
        extra = [cg.PatchDiscriminator(base_channels) for _ in range(extra_scales)]
        for module in extra:
            module.apply(cg._initialize)
        self.scales = nn.ModuleList([base, *extra])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index, discriminator in enumerate(self.scales):
            if index:
                x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
            outputs.append(discriminator(x))
        return outputs


def cycle_l1(reconstruction: torch.Tensor, original: torch.Tensor, lowpass: int = 0, detail: float = 1.0) -> torch.Tensor:
    """Cycle-consistency L1. With lowpass k, (1 - detail) of it compares k x k average-pooled images,
    so layout and colour must survive the cycle while fine texture is only weakly tied to the input."""
    full = F.l1_loss(reconstruction.float(), original.float())
    if lowpass == 0 or detail == 1:
        return full
    coarse = F.l1_loss(F.avg_pool2d(reconstruction.float(), lowpass), F.avg_pool2d(original.float(), lowpass))
    return detail * full + (1 - detail) * coarse


# PatchNCE (Park et al., ECCV 2020, "CUT"), used here as an extra content loss alongside the cycle
# loss. Features come from each generator's own encoder layers; no pretrained network is involved.
def parse_layers(layers: str) -> tuple[int, ...]:
    ids = tuple(sorted({int(part) for part in layers.split(",") if part.strip()}))
    if not ids or ids[0] < 0:
        raise ValueError("nce_layers must list non-negative generator layer indices")
    return ids


def encoder_features(generator: nn.Module, x: torch.Tensor, layer_ids: tuple[int, ...]) -> list[torch.Tensor]:
    features, h = [], x
    for index, layer in enumerate(generator.layers):
        h = layer(h)
        if index in layer_ids:
            features.append(h)
        if index >= layer_ids[-1]:
            break
    return features


class PatchHeads(nn.Module):
    """One two-layer MLP projection per encoder layer, as in CUT (256-d)."""

    def __init__(self, channels: list[int], width: int = 256) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(c, width), nn.ReLU(), nn.Linear(width, width))
                                   for c in channels)


def patch_nce(heads: PatchHeads, queries: list[torch.Tensor], keys: list[torch.Tensor],
              patches: int, tau: float) -> torch.Tensor:
    """InfoNCE between output (query) and input (key) patches at the same locations; other locations
    of the same image are negatives. Patch positions use the CPU RNG so they are checkpointed."""
    total = 0.0
    for head, query, key in zip(heads.heads, queries, keys):
        batch, channels, height, width = key.shape
        count = min(patches, height * width)
        index = torch.randperm(height * width)[:count].to(key.device)
        k = key.flatten(2)[:, :, index].permute(0, 2, 1).reshape(-1, channels).float()
        q = query.flatten(2)[:, :, index].permute(0, 2, 1).reshape(-1, channels).float()
        k = F.normalize(head(k), dim=1).detach().view(batch, count, -1)
        q = F.normalize(head(q), dim=1).view(batch, count, -1)
        logits = torch.bmm(q, k.transpose(1, 2)) / tau
        target = torch.arange(count, device=key.device).repeat(batch)
        total = total + F.cross_entropy(logits.reshape(batch * count, count), target)
    return total / len(heads.heads)


def lsgan(outputs: list[torch.Tensor], target: float) -> torch.Tensor:
    return sum(((output.float() - target) ** 2).mean() for output in outputs) / len(outputs)


def ramp(start: float, end: float, step: int, ramp_steps: int) -> float:
    if ramp_steps == 0 or step >= ramp_steps:
        return end
    return start + (end - start) * step / ramp_steps


def lr_factor(step: int, steps: int, constant_fraction: float, warmup: int = 0) -> float:
    """Optional linear warm-up (fresh Adam moments on a warm-started network otherwise jolt every
    weight at full rate), then constant, then linear decay that reaches zero after the final update."""
    scale = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    constant = int(round(steps * constant_fraction))
    if step < constant:
        return scale
    return scale * max(0.0, 1.0 - (step - constant) / max(1, steps - constant))


def image_paths(folder: Path) -> list[Path]:
    """Same set and order as the supplied evaluator's list_images for one folder."""
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: str(p))


def training_paths(data: str, data_root: Path, manifest_dir: Path) -> dict[str, list[Path]]:
    if data == "all":
        return {domain: image_paths(Path(data_root) / f"{domain}_jpg") for domain in ("photo", "monet")}
    result = {}
    for domain in ("photo", "monet"):
        lines = (Path(manifest_dir) / f"train_{domain}.txt").read_text().splitlines()
        result[domain] = [Path(data_root) / f"{domain}_jpg" / line.strip() for line in lines if line.strip()]
    return result


def fingerprint(paths: dict[str, list[Path]]) -> str:
    digest = hashlib.sha256()
    for domain in sorted(paths):
        for path in paths[domain]:
            digest.update(f"{domain}/{Path(path).name}\n".encode())
    return digest.hexdigest()


def load_image(path: Path, size: int, rng: random.Random | None, view: str = "resize286") -> torch.Tensor:
    """rng=None gives the export view. Training view "resize286" matches cyclegan.ImageDomain
    (286px resize, random 256 crop, flip); "native" keeps the evaluator's scale (flip only)."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    if rng is None or view == "native":
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        if rng is not None and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        side = max(size, round(size * 286 / 256))
        image = image.resize((side, side), Image.Resampling.BICUBIC)
        left, top = rng.randint(0, side - size), rng.randint(0, side - size)
        image = image.crop((left, top, left + size, top + size))
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def sample_batch(items: list[Path], domain: str, step: int, batch_size: int, size: int, seed: int,
                 view: str) -> torch.Tensor:
    """The training batch for one update, drawn from an RNG seeded by (seed, domain, step, index),
    so batches can be prepared ahead in background threads and resumed runs see identical data."""
    images = []
    for index in range(batch_size):
        rng = random.Random(f"{seed}:{domain}:{step}:{index}")
        images.append(load_image(items[rng.randrange(len(items))], size, rng, view))
    batch = torch.stack(images)
    return batch.pin_memory() if torch.cuda.is_available() else batch


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def to_jpeg(tensor: torch.Tensor, quality: int, subsampling: int) -> Image.Image:
    buffer = io.BytesIO()
    cg._pil(tensor).save(buffer, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    return image


@torch.no_grad()
def translate_to_jpegs(generator: nn.Module, paths: list[Path], size: int, device: str,
                       quality: int, subsampling: int, batch_size: int = 8) -> list[Image.Image]:
    generator.eval()
    images = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([load_image(p, size, None) for p in paths[start:start + batch_size]]).to(device)
        images += [to_jpeg(output, quality, subsampling) for output in generator(batch).float()]
    return images


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Identical arithmetic to the supplied notebook's frechet_distance."""
    import scipy.linalg
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid_mifid(real: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    from scipy.spatial.distance import cosine
    n = min(len(real), len(generated))
    real, generated = real[:n], generated[:n]
    fid = frechet_distance(real.mean(0), np.cov(real, rowvar=False), generated.mean(0), np.cov(generated, rowvar=False))
    return fid, float(np.mean([cosine(real[i], generated[i]) for i in range(n)]))


def _unit(features: np.ndarray) -> np.ndarray:
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def paired_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, e.g. input vs its own translation (content preservation)."""
    return (_unit(a) * _unit(b)).sum(1)


def nearest_cosine(queries: np.ndarray, references: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Each query's highest cosine similarity to any reference (memorisation check)."""
    similarity = _unit(queries) @ _unit(references).T
    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)
    return similarity.max(1)


def kid(real: np.ndarray, generated: np.ndarray) -> float:
    """Unbiased KID (Binkowski et al. 2018): squared MMD with the cubic polynomial kernel."""
    d = real.shape[1]
    k_rr = (real @ real.T / d + 1) ** 3
    k_gg = (generated @ generated.T / d + 1) ** 3
    k_rg = (real @ generated.T / d + 1) ** 3
    m, n = len(real), len(generated)
    return float((k_rr.sum() - np.trace(k_rr)) / (m * (m - 1)) + (k_gg.sum() - np.trace(k_gg)) / (n * (n - 1))
                 - 2 * k_rg.mean())


class InceptionFeatures:
    """The supplied evaluator's extractor: torchvision IMAGENET1K_V1, fc=Identity, no input transform."""

    def __init__(self, device: str) -> None:
        import torchvision.models as models
        import torchvision.transforms as T
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = nn.Identity()
        self.model, self.device = model.to(device).eval(), device
        self.transform = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        features = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(image.convert("RGB")) for image in images[start:start + batch_size]])
            features.append(self.model(batch.to(self.device)).float().cpu().numpy())
        return np.concatenate(features).astype(np.float64)


class ClassScorer:
    """In-memory version of the supplied notebook on the first 300 sorted files per folder.

    Exported names are zero-padded indices in alphabetical source order, so the first
    300 generated B2A files are translations of the first 300 sorted photos.
    """

    def __init__(self, data_root: Path, device: str, image_size: int, quality: int, subsampling: int,
                 features: Callable[[list[Image.Image]], np.ndarray] | None = None) -> None:
        self.device, self.size, self.quality, self.subsampling = device, image_size, quality, subsampling
        self.features = features or InceptionFeatures(device)
        self.monet = image_paths(Path(data_root) / "monet_jpg")[:N_EVAL]
        self.photo = image_paths(Path(data_root) / "photo_jpg")[:N_EVAL]
        self.real_monet = self.features([read_rgb(p) for p in self.monet])
        self.real_photo = self.features([read_rgb(p) for p in self.photo])

    def __call__(self, generators: dict[str, nn.Module]) -> dict[str, float]:
        a2b = self.features(translate_to_jpegs(generators["G_monet_to_photo"], self.monet, self.size,
                                               self.device, self.quality, self.subsampling))
        b2a = self.features(translate_to_jpegs(generators["G_photo_to_monet"], self.photo, self.size,
                                               self.device, self.quality, self.subsampling))
        fid_a2b, mifid_a2b = fid_mifid(self.real_photo, a2b)
        fid_b2a, mifid_b2a = fid_mifid(self.real_monet, b2a)
        fid, mifid = (fid_a2b + fid_b2a) / 2, (mifid_a2b + mifid_b2a) / 2
        return {"fid_a2b": fid_a2b, "fid_b2a": fid_b2a, "mifid_a2b": mifid_a2b, "mifid_b2a": mifid_b2a,
                "fid": fid, "mifid": mifid, "composite": (fid + mifid) / 2}


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
    temporary.replace(path)


def _cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _rng_state(sampler: random.Random) -> dict:
    # The replay pool draws from Python's global RNG, so it is checkpointed too.
    state = {"sampler": sampler.getstate(), "python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict, sampler: random.Random) -> None:
    sampler.setstate(state["sampler"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def _autocast(precision: str, device: str):
    if precision == "bf16":
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def _full_precision(precision: str, device: str):
    if precision == "bf16":
        return torch.autocast(device_type=torch.device(device).type, enabled=False)
    return torch.autocast(device_type="cpu", enabled=False)


def r1_penalty(discriminator: nn.Module, real: torch.Tensor) -> torch.Tensor:
    """Zero-centred gradient penalty on real inputs (Mescheder et al., ICML 2018): squared input-gradient
    norm of each image's mean patch score, averaged over the batch. Computed in FP32."""
    real = real.detach().float().requires_grad_(True)
    outputs = discriminator(real)
    score = sum(output.float().mean(dim=(1, 2, 3)).sum() for output in outputs) / len(outputs)
    (gradient,) = torch.autograd.grad(score, real, create_graph=True)
    return gradient.pow(2).flatten(1).sum(1).mean()


def build(base_config: dict, push: PushConfig, init_models: dict, device: str) -> tuple[dict, dict]:
    channels, blocks = base_config.get("base_channels", 64), base_config.get("residual_blocks", 9)
    generators = {name: cg.Generator(channels, blocks) for name in GENERATORS}
    for name, model in generators.items():
        model.load_state_dict(init_models[name])
        model.to(device).train()
    discriminators = {}
    for name in DISCRIMINATORS:
        state = init_models[name]
        if any(key.startswith("scales.") for key in state):
            # Saved by this trainer (possibly with a different number of scales): scale 0 is the
            # warm-started PatchGAN; saved extra scales carry over, missing ones start fresh.
            def scale_state(index: int) -> dict:
                prefix = f"scales.{index}."
                return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            base = cg.PatchDiscriminator(channels)
            base.load_state_dict(scale_state(0))
            discriminator = MultiScaleDiscriminator(base, push.extra_scales, channels)
            for index in range(1, push.extra_scales + 1):
                saved_scale = scale_state(index)
                if saved_scale:
                    discriminator.scales[index].load_state_dict(saved_scale)
        else:
            base = cg.PatchDiscriminator(channels)
            base.load_state_dict(state)
            discriminator = MultiScaleDiscriminator(base, push.extra_scales, channels)
        discriminators[name] = discriminator.to(device).train()
    return generators, discriminators


def run(push: PushConfig, init_checkpoint: Path, output_dir: Path, device: str, data_root: Path,
        manifest_dir: Path, scorer: Callable[[dict], dict] | None = None, resume: bool = False,
        stop_after: int | None = None, stop_file: Path | None = None) -> dict:
    """Train; score generators every eval_every updates; keep the best by class composite.

    If stop_file appears, the current weights are scored and checkpointed before a clean exit."""
    output_dir = Path(output_dir)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["push_config"] != asdict(push):
            raise ValueError("Resume must use the identical push configuration")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory, or resume=True")
    output_dir.mkdir(parents=True, exist_ok=True)
    if push.precision == "bf16" and torch.device(device).type != "cuda":
        raise ValueError("bf16 training requires CUDA")
    source = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    base_config = source["config"]
    size = int(base_config.get("image_size", 256))
    paths = training_paths(push.data, data_root, manifest_dir)
    random.seed(push.seed)
    np.random.seed(push.seed)
    torch.manual_seed(push.seed)
    sampler = random.Random(push.seed)
    generators, discriminators = build(base_config, push, (saved if resume else source)["models"], device)
    del source
    on_cuda = torch.device(device).type == "cuda"
    if on_cuda:
        # Fixed 256x256 batch-1 shapes: let cuDNN pick its fastest kernels once.
        torch.backends.cudnn.benchmark = True
    adam = {"betas": (push.beta1, push.beta2), **({"fused": True} if on_cuda else {})}
    nce_layers = parse_layers(push.nce_layers)
    heads = {}
    if push.nce_weight > 0:
        with torch.no_grad():
            probe = torch.zeros(1, 3, 64, 64, device=device)
            channels = [f.shape[1] for f in encoder_features(generators[GENERATORS[0]], probe, nce_layers)]
        if len(channels) != len(nce_layers):
            raise ValueError("nce_layers index beyond the generator's layers")
        heads = {name: PatchHeads(channels).to(device).train() for name in GENERATORS}
        if resume:
            for name, module in heads.items():
                module.load_state_dict(saved["nce_heads"][name])
    optimizers = {
        "generators": torch.optim.Adam([p for g in generators.values() for p in g.parameters()]
                                       + [p for h in heads.values() for p in h.parameters()],
                                       lr=push.learning_rate, **adam),
        "discriminators": torch.optim.Adam([p for d in discriminators.values() for p in d.parameters()],
                                           lr=push.learning_rate, **adam)}
    pool_device = device if on_cuda else "cpu"
    pools = {domain: cg.ReplayPool(push.replay_size, pool_device) for domain in ("photo", "monet")}
    ema = FastGeneratorEMA(generators, push.ema_beta)
    # Optional longer-horizon average (e.g. 0.9999 ~ 10k updates) for long runs.
    ema_slow = FastGeneratorEMA(generators, push.ema_slow_beta) if push.ema_slow_beta > 0 else None
    state = {"step": 0, "warmup_done": 0, "best": None, "history": [], "nan_events": 0, "training_seconds": 0.0,
             "init_checkpoint": str(init_checkpoint), "init_sha256": _sha256(init_checkpoint),
             "data_fingerprint": fingerprint(paths), "data_counts": {k: len(v) for k, v in paths.items()}}
    if resume:
        for name, opt in optimizers.items():
            opt.load_state_dict(saved["optimizers"][name])
        for name, pool in pools.items():
            pool.load_state_dict(saved["replay_pools"][name])
        ema.load_state_dict(saved["ema"])
        if ema_slow is not None:
            ema_slow.load_state_dict(saved["ema_slow"])
        state = saved["state"]
        _restore_rng(saved["rng"], sampler)
        del saved

    def checkpoint() -> None:
        _atomic_save({"trainer": TRAINER, "push_config": asdict(push), "base_config": base_config,
                      "models": {**{k: _cpu(v) for k, v in generators.items()}, **{k: _cpu(v) for k, v in discriminators.items()}},
                      "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                      "replay_pools": {k: v.state_dict() for k, v in pools.items()},
                      "ema": ema.state_dict(), "ema_slow": ema_slow.state_dict() if ema_slow else None,
                      "nce_heads": {k: _cpu(v) for k, v in heads.items()} or None,
                      "rng": _rng_state(sampler), "state": state}, last_path)

    def candidate(scores: dict, variant: str) -> dict:
        chosen = {"ema": ema.models, "raw": generators, "ema_slow": ema_slow.models if ema_slow else None}[variant]
        return {"format_version": 1, "trainer": TRAINER, "inference_only": True, "config": base_config,
                "push_config": asdict(push),
                "state": {"global_step": state["step"], "data_fingerprint": state["data_fingerprint"],
                          "data_counts": state["data_counts"], "training_data": push.data,
                          "init_checkpoint": state["init_checkpoint"], "init_sha256": state["init_sha256"],
                          "candidate_variant": variant,
                          "ema_beta": ema_slow.beta if variant == "ema_slow" else ema.beta, "ema_updates": ema.updates},
                "models": {**{k: _cpu(chosen[k]) for k in GENERATORS},
                           **{k: _cpu(v) for k, v in discriminators.items()}},
                "class_scores": scores,
                "selection_note": "Selected by the supplied class FID/MiFID evaluator on its fixed 300+300 images"}

    def log(record: dict) -> None:
        with (output_dir / "training_log.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def evaluate() -> None:
        if scorer is None:
            return
        for variant in parse_variants(push.eval_variants):
            models = {"ema": ema.models, "raw": generators, "ema_slow": ema_slow.models if ema_slow else None}[variant]
            scores = scorer(models)
            for model in models.values():
                model.train(variant == "raw")
            # Wall-clock time stays out of the checkpointed history so resumed runs match exactly.
            record = {"step": state["step"], "variant": variant, **scores}
            state["history"].append(record)
            log({"evaluation": record, "elapsed": time.perf_counter() - started})
            print("EVAL " + json.dumps(record), flush=True)
            if state["best"] is None or scores["composite"] < state["best"]["composite"]:
                state["best"] = record
                _atomic_save(candidate(scores, variant), best_path)

    def load_pair(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(sample_batch(paths[domain], domain, step, push.batch_size, size, push.seed, push.train_view)
                     for domain in ("photo", "monet"))

    # Background threads decode/augment the next batches while the GPU trains (PIL releases the GIL).
    loader = ThreadPoolExecutor(max_workers=2) if push.prefetch > 0 else None
    pending = {}

    def batch_for(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if loader is None or step < 0:
            pair = load_pair(step)
        else:
            for ahead in range(step, step + push.prefetch + 1):
                if ahead not in pending:
                    pending[ahead] = loader.submit(load_pair, ahead)
            pair = pending.pop(step).result()
        # Asynchronous copies only from pinned memory on CUDA; on MPS a non-blocking copy can read the
        # CPU tensor after Python has freed it.
        return tuple(x.to(device, non_blocking=on_cuda) for x in pair)

    def d_input(x: torch.Tensor, policy: str) -> torch.Tensor:
        # DiffAugment, then optional instance noise (Gaussian, linearly ramped over the schedule).
        x = diff_augment(x, policy)
        sigma = ramp(push.d_noise_start, push.d_noise_end, state["step"], push.steps)
        return x + sigma * torch.randn_like(x) if sigma > 0 else x

    def discriminator_step(photo, monet, fake_photo, fake_monet) -> dict[str, torch.Tensor]:
        losses = {}
        apply_r1 = push.r1_gamma > 0 and state["step"] % push.r1_every == 0
        with _autocast(push.precision, device):
            for domain, real, fake, policy in (("photo", photo, fake_photo, push.aug_photo),
                                               ("monet", monet, fake_monet, push.aug_monet)):
                d = discriminators[f"D_{domain}"]
                losses[f"discriminator_{domain}"] = 0.5 * (lsgan(d(d_input(real, policy)), 1.0)
                                                           + lsgan(d(d_input(pools[domain].query(fake), policy)), 0.0))
        if apply_r1:
            with _full_precision(push.precision, device):
                for domain, real, policy in (("photo", photo, push.aug_photo), ("monet", monet, push.aug_monet)):
                    penalty = r1_penalty(discriminators[f"D_{domain}"], d_input(real, policy))
                    losses[f"r1_{domain}"] = penalty
                    # Lazy regularisation (StyleGAN2): every r1_every steps, scaled by r1_every.
                    losses[f"discriminator_{domain}"] = (losses[f"discriminator_{domain}"]
                                                         + 0.5 * push.r1_gamma * push.r1_every * penalty)
        return losses

    started = time.perf_counter()
    if not resume and state["warmup_done"] == 0 and scorer is not None:
        evaluate()
    while state["warmup_done"] < push.d_warmup_steps:
        photo, monet = batch_for(-1 - state["warmup_done"])
        with torch.no_grad(), _autocast(push.precision, device):
            fake_monet, fake_photo = generators["G_photo_to_monet"](photo), generators["G_monet_to_photo"](monet)
        optimizers["discriminators"].zero_grad(set_to_none=True)
        losses = discriminator_step(photo, monet, fake_photo, fake_monet)
        (losses["discriminator_photo"] + losses["discriminator_monet"]).backward()
        optimizers["discriminators"].step()
        state["warmup_done"] += 1
        if state["warmup_done"] % push.log_every == 0:
            log({"warmup": state["warmup_done"], **{k: float(v.detach()) for k, v in losses.items()}})
    target = push.steps if stop_after is None else min(push.steps, stop_after)
    while state["step"] < target:
        step = state["step"]
        for opt in optimizers.values():
            for group in opt.param_groups:
                group["lr"] = push.learning_rate * lr_factor(step, push.steps, push.constant_fraction, push.warmup_steps)
        cycle_weight = ramp(push.cycle_start, push.cycle_end, step, push.weight_ramp_steps)
        identity_weight = ramp(push.identity_start, push.identity_end, step, push.weight_ramp_steps)
        photo, monet = batch_for(step)
        tick = time.perf_counter()
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        for model in discriminators.values():
            model.requires_grad_(False)
        with _autocast(push.precision, device):
            output = cg.cycle_forward(generators, photo, monet) if identity_weight > 0 else {
                "fake_monet": generators["G_photo_to_monet"](photo), "fake_photo": generators["G_monet_to_photo"](monet)}
            if identity_weight == 0:
                output["cycle_photo"] = generators["G_monet_to_photo"](output["fake_monet"])
                output["cycle_monet"] = generators["G_photo_to_monet"](output["fake_photo"])
            losses = {
                "gan_photo_to_monet": lsgan(discriminators["D_monet"](d_input(output["fake_monet"], push.aug_monet)), 1.0),
                "gan_monet_to_photo": lsgan(discriminators["D_photo"](d_input(output["fake_photo"], push.aug_photo)), 1.0),
                "cycle_photo_l1": cycle_l1(output["cycle_photo"], photo, push.cycle_lowpass, push.cycle_detail),
                "cycle_monet_l1": cycle_l1(output["cycle_monet"], monet, push.cycle_lowpass, push.cycle_detail)}
            # Photo cycle P->M->P limits how freely G_photo_to_monet can restyle; Monet cycle M->P->M
            # limits G_monet_to_photo. Separate scales let one direction translate more strongly.
            total = (losses["gan_photo_to_monet"] + losses["gan_monet_to_photo"]
                     + cycle_weight * (push.cycle_photo_scale * losses["cycle_photo_l1"]
                                       + push.cycle_monet_scale * losses["cycle_monet_l1"]))
            if identity_weight > 0:
                losses["identity_photo_l1"] = F.l1_loss(output["identity_photo"].float(), photo.float())
                losses["identity_monet_l1"] = F.l1_loss(output["identity_monet"].float(), monet.float())
                total = total + identity_weight * (losses["identity_photo_l1"] + losses["identity_monet_l1"])
            if push.nce_weight > 0:
                for name, source, translated in (("G_photo_to_monet", photo, output["fake_monet"]),
                                                 ("G_monet_to_photo", monet, output["fake_photo"])):
                    generator = generators[name]
                    losses[f"nce_{name}"] = patch_nce(heads[name], encoder_features(generator, translated, nce_layers),
                                                      encoder_features(generator, source, nce_layers),
                                                      push.nce_patches, push.nce_tau)
                total = total + push.nce_weight * (losses["nce_G_photo_to_monet"] + losses["nce_G_monet_to_photo"])
                if push.nce_idt_weight > 0:
                    # CUT's identity NCE: a generator should leave its target domain's structure alone.
                    for name, target_image in (("G_photo_to_monet", monet), ("G_monet_to_photo", photo)):
                        generator = generators[name]
                        same = output.get("identity_monet" if name == "G_photo_to_monet" else "identity_photo")
                        same = generator(target_image) if same is None else same
                        losses[f"nce_idt_{name}"] = patch_nce(heads[name], encoder_features(generator, same, nce_layers),
                                                              encoder_features(generator, target_image, nce_layers),
                                                              push.nce_patches, push.nce_tau)
                    total = total + push.nce_idt_weight * (losses["nce_idt_G_photo_to_monet"]
                                                           + losses["nce_idt_G_monet_to_photo"])
            if push.flip_equivariance > 0:
                losses["flip_equivariance_l1"] = (
                    F.l1_loss(torch.flip(generators["G_photo_to_monet"](torch.flip(photo, [3])), [3]).float(), output["fake_monet"].float())
                    + F.l1_loss(torch.flip(generators["G_monet_to_photo"](torch.flip(monet, [3])), [3]).float(), output["fake_photo"].float()))
                total = total + push.flip_equivariance * losses["flip_equivariance_l1"]
        losses["generator_total"] = total
        total.backward()
        for model in discriminators.values():
            model.requires_grad_(True)
        losses.update(discriminator_step(photo, monet, output["fake_photo"], output["fake_monet"]))
        (losses["discriminator_photo"] + losses["discriminator_monet"]).backward()
        loss_stack = torch.stack([v.detach().float().reshape(()) for v in losses.values()])
        if not bool(torch.isfinite(loss_stack).all()):  # one host sync per update instead of one per loss
            state["nan_events"] += 1
            checkpoint()
            raise FloatingPointError(f"Non-finite loss at update {step + 1}; last.pt holds the previous state")
        for opt in optimizers.values():
            opt.step()
        ema.update(generators)
        if ema_slow is not None:
            ema_slow.update(generators)
        state["step"] += 1
        state["training_seconds"] += time.perf_counter() - tick
        if state["step"] % push.log_every == 0:
            values = dict(zip(losses.keys(), loss_stack.tolist()))
            log({"step": state["step"], "elapsed": time.perf_counter() - started, "cycle_weight": cycle_weight,
                 "identity_weight": identity_weight, "lr": optimizers["generators"].param_groups[0]["lr"], **values})
            rate = state["step"] / max(time.perf_counter() - started, 1e-9)
            print(f"step {state['step']}/{push.steps} G={values['generator_total']:.3f} "
                  f"Dp={values['discriminator_photo']:.3f} Dm={values['discriminator_monet']:.3f} "
                  f"cyc_w={cycle_weight:.2f} id_w={identity_weight:.2f} {rate:.2f} it/s", flush=True)
        evaluated = state["step"] % push.eval_every == 0 or state["step"] == push.steps
        if evaluated:
            evaluate()
        if stop_file is not None and state["step"] % push.log_every == 0 and Path(stop_file).exists():
            if not evaluated:
                evaluate()
            checkpoint()
            print(f"STOP file seen at update {state['step']}; scored and checkpointed", flush=True)
            break
        if state["step"] % push.checkpoint_every == 0 or state["step"] == target:
            checkpoint()
    if loader is not None:
        loader.shutdown(wait=False, cancel_futures=True)
    summary = {"trainer": TRAINER, "status": "completed" if state["step"] == push.steps else "partial",
               "completed_updates": state["step"], "best": state["best"], "nan_events": state["nan_events"],
               "training_seconds": state["training_seconds"], "push_config": asdict(push),
               "data_counts": state["data_counts"], "init_sha256": state["init_sha256"],
               "best_checkpoint": str(best_path) if best_path.exists() else None}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
