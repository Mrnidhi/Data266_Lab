"""Part 3: independently trained CycleGAN, explicit data provenance, honest metrics.

Architecture/loss references: Zhu et al. https://arxiv.org/abs/1703.10593
Evaluation: https://github.com/GaParmar/clean-fid,
https://github.com/richzhang/PerceptualSimilarity,
https://github.com/clovaai/generative-evaluation-prdc.
No pretrained weights are used by the four trainable networks.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F


from .common import project_root, task_config_path


DIRECTIONS = ("photo_to_monet", "monet_to_photo")
METRICS = ("fid", "kid", "precision", "recall", "density", "coverage", "lpips_cycle", "content_cosine")
RESUME_FIELDS = ("seed", "image_size", "base_channels", "residual_blocks", "learning_rate", "betas",
                 "cycle_weight", "identity_weight", "epochs", "constant_epochs", "batch_size", "replay_size", "precision")


def _recipe_value(config, field):
    # Checkpoints predating mixed precision were always FP32.
    return config.get(field, "fp32" if field == "precision" else None)


def _training_options(config, device):
    precision = config.get("precision", "fp32")
    replay_device = config.get("replay_device", "cpu")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be 'fp32' or 'bf16'; fp16 scaling is not implemented")
    if replay_device not in ("cpu", "device"):
        raise ValueError("replay_device must be 'cpu' or 'device'")
    if precision == "bf16":
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("bf16 training requires a CUDA device with native BF16 support")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("bf16 training requires a CUDA device with native BF16 support")
    return precision, replay_device


def _training_autocast(precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def resolve_config(config):
    config = dict(config)
    if "smoke" in config:
        mode = config.get("mode", "smoke")
        return {**config[mode], "mode": mode}
    return config


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels))

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    def __init__(self, base_channels=64, residual_blocks=9):
        super().__init__()
        c = base_channels
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, c, 7),
                  nn.InstanceNorm2d(c), nn.ReLU(True)]
        for _ in range(2):
            layers += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        layers += [ResidualBlock(c) for _ in range(residual_blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PatchDiscriminator(nn.Module):
    """Five kernel-4 convolutions with strides 2,2,2,1,1: 70px receptive field."""
    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        layers = [nn.Conv2d(3, c, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        for multiplier, stride in ((2, 2), (4, 2), (8, 1)):
            out = base_channels * multiplier
            layers += [nn.Conv2d(c, out, 4, stride, 1),
                       nn.InstanceNorm2d(out), nn.LeakyReLU(0.2, True)]
            c = out
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def _initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_models(config, device):
    c, blocks = config.get("base_channels", 64), config.get("residual_blocks", 9)
    models = {"G_photo_to_monet": Generator(c, blocks), "G_monet_to_photo": Generator(c, blocks),
              "D_photo": PatchDiscriminator(c), "D_monet": PatchDiscriminator(c)}
    for model in models.values():
        model.apply(_initialize)
        model.to(device)
    return models


class ReplayPool:
    def __init__(self, capacity=50, device="cpu"):
        self.capacity, self.images = int(capacity), []
        self.device = torch.device(device)

    def query(self, batch):
        result = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if self.capacity == 0:
                result.append(image)
            elif len(self.images) < self.capacity:
                self.images.append(image.to(self.device).clone())
                result.append(image)
            elif random.random() < 0.5:
                index = random.randrange(self.capacity)
                result.append(self.images[index].to(image.device).clone())
                self.images[index] = image.to(self.device).clone()
            else:
                result.append(image)
        return torch.cat(result, 0).detach()

    def state_dict(self):
        return {"capacity": self.capacity, "images": [x.detach().cpu().clone() for x in self.images]}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.images = [x.detach().to(self.device).clone() for x in state["images"]]


def cycle_forward(models, photo, monet):
    fake_monet = models["G_photo_to_monet"](photo)
    fake_photo = models["G_monet_to_photo"](monet)
    return {"fake_monet": fake_monet, "fake_photo": fake_photo,
            "cycle_photo": models["G_monet_to_photo"](fake_monet),
            "cycle_monet": models["G_photo_to_monet"](fake_photo),
            "identity_photo": models["G_monet_to_photo"](photo),
            "identity_monet": models["G_photo_to_monet"](monet)}


def generator_losses(models, photo, monet, output, cycle_weight=10., identity_weight=5.):
    """identity_weight is absolute, unlike the original repo's relative flag."""
    raw = {
        "gan_photo_to_monet": ((models["D_monet"](output["fake_monet"]).float() - 1) ** 2).mean(),
        "gan_monet_to_photo": ((models["D_photo"](output["fake_photo"]).float() - 1) ** 2).mean(),
        "cycle_photo_l1": F.l1_loss(output["cycle_photo"].float(), photo.float()),
        "cycle_monet_l1": F.l1_loss(output["cycle_monet"].float(), monet.float()),
        "identity_photo_l1": F.l1_loss(output["identity_photo"].float(), photo.float()),
        "identity_monet_l1": F.l1_loss(output["identity_monet"].float(), monet.float())}
    raw["generator_total"] = (raw["gan_photo_to_monet"] + raw["gan_monet_to_photo"]
                              + cycle_weight * (raw["cycle_photo_l1"] + raw["cycle_monet_l1"])
                              + identity_weight * (raw["identity_photo_l1"] + raw["identity_monet_l1"]))
    return raw


def discriminator_loss(discriminator, real, fake):
    return 0.5 * (((discriminator(real).float() - 1) ** 2).mean() + (discriminator(fake.detach()).float() ** 2).mean())


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        available = torch.cuda.device_count()
        for index, saved in enumerate(state["cuda"][:available]):
            torch.cuda.set_rng_state(saved.cpu(), device=index)
        if len(state["cuda"]) != available:
            print("CUDA device count changed on resume; restored matching device RNG states. "
                  "Additional devices keep their initialized seed. Cross-device continuation "
                  "is not guaranteed to be bitwise identical.", flush=True)
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path, models, optimizers, schedulers, pools, config, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "config": config, "state": state,
                "models": {k: v.state_dict() for k, v in models.items()},
                "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                "replay_pools": {k: v.state_dict() for k, v in pools.items()},
                "rng": rng_state()}, temporary)
    temporary.replace(path)


def load_checkpoint(path, models, optimizers=None, schedulers=None, pools=None, restore_random=True):
    # Only load this project's own trusted checkpoints; optimizer/RNG state requires pickle.
    checkpoint = path if isinstance(path, dict) else torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported CycleGAN checkpoint format")
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
    for key, objects in (("optimizers", optimizers), ("schedulers", schedulers), ("replay_pools", pools)):
        if objects is not None:
            for name, obj in objects.items():
                obj.load_state_dict(checkpoint[key][name])
    if restore_random:
        restore_rng(checkpoint["rng"])
    return checkpoint


def _preserve_best_checkpoint(resume, output_dir, restored, provenance):
    source, destination = Path(resume).parent / "best.pt", Path(output_dir) / "best.pt"
    selection = restored["state"].get("best_selection")
    if selection is None:
        if source.exists() or destination.exists() or restored["state"].get("best_validation_score") is not None:
            raise ValueError("Untracked best checkpoint/score; use a consistent checkpoint bundle")
        return
    if not provenance["class_results"]:
        raise ValueError("Synthetic rehearsal checkpoints cannot have a selected class best checkpoint")
    if not source.is_file():
        raise ValueError("Selected best.pt is missing; transfer it alongside the resume checkpoint")
    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    expected = digest(source)
    candidate = torch.load(source, map_location="cpu", weights_only=False)
    candidate_state = candidate.get("state", {})
    valid = (candidate.get("format_version") == 1
             and all(_recipe_value(candidate.get("config", {}), key) == _recipe_value(restored["config"], key) for key in RESUME_FIELDS)
             and candidate_state.get("manifest_fingerprint") == provenance["manifest_fingerprint"]
             and candidate_state.get("best_selection") == selection
             and candidate_state.get("best_validation_score") == restored["state"].get("best_validation_score")
             and candidate_state.get("global_step") == selection.get("step")
             and isinstance(selection.get("step"), int)
             and 0 <= selection["step"] <= restored["state"]["global_step"]
             and selection.get("validation_only") is True
             and selection.get("metric") == "mean_validation_KID_both_directions"
             and selection.get("value") == restored["state"].get("best_validation_score")
             and isinstance(selection.get("value"), (int, float)) and math.isfinite(selection["value"]))
    del candidate
    if not valid:
        raise ValueError("Best checkpoint is stale, future, or inconsistent with the resumed data/recipe/selection")
    if digest(source) != expected:
        raise ValueError("Best checkpoint changed while validating; pause training and retry")
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        if digest(destination) != expected:
            raise FileExistsError("Refusing to overwrite a different best.pt in the destination run")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".best-transfer-", suffix=".tmp", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise ValueError("Best checkpoint changed while copying; pause training and retry")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class ImageDomain:
    def __init__(self, records, size, augment=False, synthetic_seed=None):
        self.records, self.size, self.augment, self.synthetic_seed = records, size, augment, synthetic_seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.synthetic_seed is not None:
            rng = np.random.default_rng(self.synthetic_seed + index)
            array = rng.integers(0, 256, (self.size, self.size, 3), dtype=np.uint8)
            image = Image.fromarray(array)
        else:
            with Image.open(self.records[index]) as source:
                image = source.convert("RGB")
        if self.augment:
            side = max(self.size, round(self.size * 286 / 256))
            image = image.resize((side, side), Image.Resampling.BICUBIC)
            left, top = random.randint(0, side - self.size), random.randint(0, side - self.size)
            image = image.crop((left, top, left + self.size, top + self.size))
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.
        return torch.from_numpy(array.transpose(2, 0, 1))


def prepare_data(config):
    """Explicit six-file manifests are required for actual class data."""
    mode, spec = config.get("mode", "smoke"), config.get("data", {})
    size = int(config.get("image_size", 256))
    if size < 32 or size % 4:
        raise ValueError("image_size must be at least 32 and divisible by 4")
    supplied = any(spec.get(k) for k in ("monet_dir", "photo_dir", "manifest_dir"))
    if not supplied:
        if mode == "full" or not config.get("allow_synthetic", False):
            raise ValueError("Actual class data required: set data.monet_dir, photo_dir, manifest_dir; no dataset is assumed")
        data, counts = {}, {}
        for split in ("train", "val", "test"):
            for domain in ("photo", "monet"):
                n = int(config.get("synthetic_images", 8))
                seed = config.get("seed", 2342) + (10000 if domain == "monet" else 0) + (20000 * ("train", "val", "test").index(split))
                key = f"{split}_{domain}"
                data[key] = ImageDomain(list(range(n)), size, split == "train", seed)
                counts[key] = n
        return data, {"kind": "synthetic_pipeline_check", "class_results": False, "counts": counts,
                      "manifest_fingerprint": f"synthetic:{config.get('seed', 2342)}:{size}:{n}"}
    if not all(spec.get(k) for k in ("monet_dir", "photo_dir", "manifest_dir")):
        raise ValueError("Provide all three data paths; partial/invalid data never silently falls back to synthetic")
    manifest_dir = Path(spec["manifest_dir"]).expanduser().resolve()
    data, entries, content_seen = {}, {}, {}
    for domain in ("photo", "monet"):
        root = Path(spec[f"{domain}_dir"]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for split in ("train", "val", "test"):
            key = f"{split}_{domain}"
            manifest = manifest_dir / f"{key}.txt"
            lines = [x.strip() for x in manifest.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
            if not lines or len(set(lines)) != len(lines):
                raise ValueError(f"Empty/duplicate entries in {manifest}")
            paths, hashes = [], []
            for entry in lines:
                path = (root / entry).resolve()
                if Path(entry).is_absolute() or not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Manifest entry must be an existing relative image under {root}: {entry}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = content_seen.get((domain, digest))
                if previous is not None:
                    raise ValueError(f"Duplicate image content within/across splits: {previous} and {key}/{entry}")
                content_seen[(domain, digest)] = f"{key}/{entry}"
                with Image.open(path) as image:
                    image.verify()
                paths.append(path)
                hashes.append({"relative_path": entry, "sha256": digest})
            data[key] = ImageDomain(paths, size, split == "train")
            entries[key] = hashes
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return data, {"kind": "explicit_class_manifests", "class_results": True,
                  "counts": {k: len(v) for k, v in data.items()}, "entries": entries,
                  "manifest_fingerprint": fingerprint, "source_note": spec.get("source_note", "User-provided class data; course rules verification required")}


def _pil(tensor):
    array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _grad_norm(model):
    values = [p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.


@torch.no_grad()
def save_grid(models, data, path, device):
    size = data["val_photo"].size
    canvas = Image.new("RGB", (size * 3, (size + 22) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for row, domain in enumerate(("photo", "monet")):
        target = "monet" if domain == "photo" else "photo"
        x = data[f"val_{domain}"][0].unsqueeze(0).to(device)
        fake = models[f"G_{domain}_to_{target}"](x)
        cycle = models[f"G_{target}_to_{domain}"](fake)
        for col, (tensor, label) in enumerate(((x, domain), (fake, f"generated {target}"), (cycle, f"cycle {domain}"))):
            draw.text((col * size + 2, row * (size + 22) + 2), label, fill="black")
            canvas.paste(_pil(tensor[0]), (col * size, row * (size + 22) + 22))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _unavailable(reason):
    return {"status": "unavailable", "value": None, "reason": str(reason)}


def _measurement(value, **extra):
    value = float(value)
    if not math.isfinite(value):
        return _unavailable("Metric computation returned a non-finite value")
    return {"status": "computed", "value": value, **extra}


def configure_metric_cache(cache_dir=None):
    """Keep transferable evaluation weights together, independent of /tmp."""
    default = project_root() / ".cache" / "metric_weights"
    directory = Path(cache_dir or os.environ.get("LAB1_METRIC_CACHE", default)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(directory / "torch")
    torch.hub.set_dir(str(directory / "torch" / "hub"))
    return directory


def build_metric_feature_extractor(device, cache_dir=None, download=True):
    """Clean-FID's clean-mode model, with an explicit portable cache path."""
    from cleanfid.inception_torchscript import InceptionV3W
    directory = configure_metric_cache(cache_dir) / "cleanfid"
    directory.mkdir(parents=True, exist_ok=True)
    model = InceptionV3W(str(directory), download=download, resize_inside=False).to(device).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate_models(models, data, provenance, output_dir, device, split="test", options=None):
    """Both directions. Optional metric failures are explicit, never fabricated zeros."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation must use val or test manifests")
    options, output_dir = options or {}, Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_modes = {name: model.training for name, model in models.items()}
    for model in models.values():
        model.eval()
    result = {"split": split, "data_kind": provenance["kind"], "class_results": provenance["class_results"],
              "manifest_fingerprint": provenance["manifest_fingerprint"], "directions": {},
              "lpips_pairing": "original versus cycle reconstruction (not arbitrary unpaired target)",
              "content_encoder": "torchvision ResNet18 IMAGENET1K_V1, frozen, pooled512; input versus translation",
              "distribution_encoder": "Clean-FID clean-mode InceptionV3 2048 features",
              "caveat": "Small reference sets make FID and neighborhood metrics uncertain; fixed counts do not remove FID bias."}
    external = bool(options.get("allow_metric_downloads", False)) and provenance["class_results"]
    reason = "Synthetic data are pipeline checks, not class quality results" if not provenance["class_results"] else "Pretrained metric loading disabled (allow_metric_downloads=false)"
    lpips_net, content_net, content_transform, fid_module, feature_model = None, None, None, None, None
    issues = {name: reason for name in METRICS}
    if external:
        cache_directory = configure_metric_cache(options.get("cache_dir"))
        try:
            import lpips
            lpips_net = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
            lpips_net.requires_grad_(False)
        except Exception as exc:
            issues["lpips_cycle"] = f"LPIPS dependency/weights unavailable: {type(exc).__name__}: {exc}"
        try:
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
            content_net = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
            content_net.requires_grad_(False)
            content_transform = weights.transforms()
        except Exception as exc:
            issues["content_cosine"] = f"Frozen content encoder unavailable: {type(exc).__name__}: {exc}"
        try:
            from cleanfid import fid as fid_module
            feature_model = build_metric_feature_extractor(torch.device(device), cache_directory)
        except Exception as exc:
            fid_module, feature_model = None, None
            for name in ("fid", "kid", "precision", "recall", "density", "coverage"):
                issues[name] = f"Clean-FID feature dependency/weights unavailable: {type(exc).__name__}: {exc}"
    try:
        for direction in DIRECTIONS:
            source, target = direction.split("_to_")
            source_set, real_set = data[f"{split}_{source}"], data[f"{split}_{target}"]
            cap = options.get("max_images")
            n_source = min(len(source_set), int(cap)) if cap else len(source_set)
            n_real = min(len(real_set), int(cap)) if cap else len(real_set)
            folder = output_dir / direction
            for name in ("source", "generated", "cycle", "real_target"):
                target_folder = folder / name
                if target_folder.exists() and any(target_folder.iterdir()):
                    raise FileExistsError(f"Use a fresh evaluation directory to prevent stale image contamination: {target_folder}")
                target_folder.mkdir(parents=True, exist_ok=True)
            entry = {"source_count": n_source, "generated_count": n_source, "real_reference_count": n_real,
                     "metrics": {name: _unavailable(issues[name]) for name in METRICS}}
            cycles, perceptual, content = [], [], []
            source_ids = []
            _sync(device)
            start = time.perf_counter()
            inference_seconds = 0.
            for i in range(n_source):
                x = source_set[i].unsqueeze(0).to(device)
                _sync(device)
                infer_start = time.perf_counter()
                translated = models[f"G_{direction}"](x)
                _sync(device)
                inference_seconds += time.perf_counter() - infer_start
                reconstruction = models[f"G_{target}_to_{source}"](translated)
                cycles.append(float(F.l1_loss(reconstruction, x)))
                if lpips_net is not None:
                    perceptual.append(float(lpips_net(x, reconstruction).mean()))
                if content_net is not None:
                    a = content_net(content_transform((x + 1) / 2)).flatten(1)
                    b = content_net(content_transform((translated + 1) / 2)).flatten(1)
                    content.append(float(F.cosine_similarity(a, b).mean()))
                for label, tensor in (("source", x), ("generated", translated), ("cycle", reconstruction)):
                    _pil(tensor[0]).save(folder / label / f"{i:06d}.png")
                source_ids.append(str(source_set.records[i]))
            for i in range(n_real):
                _pil(real_set[i]).save(folder / "real_target" / f"{i:06d}.png")
            entry.update({"inference_seconds": inference_seconds,
                          "inference_images_per_second": n_source / max(inference_seconds, 1e-9),
                          "generation_and_pair_metrics_seconds": time.perf_counter() - start})
            entry["metrics"]["cycle_l1"] = _measurement(np.mean(cycles), count=n_source, pixel_range="[-1,1]")
            if perceptual:
                entry["metrics"]["lpips_cycle"] = _measurement(np.mean(perceptual), count=n_source)
            if content:
                entry["metrics"]["content_cosine"] = _measurement(np.mean(content), count=n_source)
            if feature_model is not None:
                try:
                    feature_args = {"model": feature_model, "num_workers": 0, "batch_size": options.get("batch_size", 16),
                                    "device": torch.device(device), "mode": "clean", "verbose": False}
                    real_features = fid_module.get_folder_features(str(folder / "real_target"), **feature_args)
                    fake_features = fid_module.get_folder_features(str(folder / "generated"), **feature_args)
                    if min(n_real, n_source) < 2:
                        raise ValueError("FID/KID need at least two images per distribution")
                    entry["metrics"]["fid"] = _measurement(fid_module.fid_from_feats(real_features, fake_features))
                    subset_size = min(options.get("kid_subset_size", 100), n_real, n_source)
                    repetitions = int(options.get("kid_subsets", 50))
                    old_numpy = np.random.get_state()
                    np.random.seed(int(options.get("seed", 2342)))
                    try:
                        values = [fid_module.kernel_distance(real_features, fake_features, num_subsets=1,
                                                            max_subset_size=subset_size) for _ in range(repetitions)]
                    finally:
                        np.random.set_state(old_numpy)
                    entry["metrics"]["kid"] = _measurement(np.mean(values), subset_std=float(np.std(values)),
                                                            subsets=repetitions, subset_size=subset_size,
                                                            note="Subset SD is not training-seed uncertainty; unbiased KID estimates may be negative")
                    try:
                        from prdc import compute_prdc
                        nearest_k = int(options.get("prdc_k", 5))
                        if min(n_real, n_source) <= nearest_k:
                            raise ValueError(f"PRDC requires more than k={nearest_k} samples per distribution")
                        values = compute_prdc(real_features, fake_features, nearest_k=nearest_k)
                        for name, value in values.items():
                            entry["metrics"][name] = _measurement(value, nearest_k=nearest_k)
                    except Exception as exc:
                        for name in ("precision", "recall", "density", "coverage"):
                            entry["metrics"][name] = _unavailable(f"PRDC: {type(exc).__name__}: {exc}")
                except Exception as exc:
                    for name in ("fid", "kid", "precision", "recall", "density", "coverage"):
                        if entry["metrics"][name]["status"] != "computed":
                            entry["metrics"][name] = _unavailable(f"Distribution metrics: {type(exc).__name__}: {exc}")
            _json(folder / "source_ids.json", source_ids)
            result["directions"][direction] = entry
    finally:
        for name, model in models.items():
            model.train(prior_modes[name])
    _json(output_dir / "metrics.json", result)
    return result


def make_human_audit(evaluation_dir, output_dir, seed=2342, samples_per_direction=30):
    """Blank independent rating sheets; 30 per direction is a conservative interpretation."""
    evaluation_dir, output_dir = Path(evaluation_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng, pairs, key = random.Random(seed), [], []
    for direction in DIRECTIONS:
        files = sorted((evaluation_dir / direction / "generated").glob("*.png"))
        rng.shuffle(files)
        for generated in files[:samples_per_direction]:
            pairs.append((direction, generated))
    rng.shuffle(pairs)
    rows = []
    for i, (direction, generated) in enumerate(pairs):
        sample_id = f"sample_{i + 1:03d}"
        source = generated.parent.parent / "source" / generated.name
        with Image.open(source) as original, Image.open(generated) as output:
            image = Image.new("RGB", (original.width * 2, original.height + 22), "white")
            ImageDraw.Draw(image).text((3, 3), "Original (left) | Anonymous translation (right)", fill="black")
            image.paste(original, (0, 22))
            image.paste(output, (original.width, 22))
            image.save(output_dir / f"{sample_id}.png")
        rows.append({"sample_id": sample_id, "direction": direction, "style": "", "content": "", "artifacts": ""})
        key.append({"sample_id": sample_id, "generated_path": str(generated), "direction": direction})
    for rater in (1, 2):
        with (output_dir / f"rater_{rater}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "direction", "style", "content", "artifacts"])
            writer.writeheader()
            writer.writerows(rows)
    _json(output_dir / "private_key_do_not_share_with_raters.json", key)
    manifest = {"requested_per_direction": samples_per_direction, "actual_total": len(rows),
                "counts": {d: sum(r["direction"] == d for r in rows) for d in DIRECTIONS},
                "complete_sample_count": len(rows) == samples_per_direction * 2,
                "rubric": {"style": "1=no target style, 5=strong target style", "content": "1=scene changed, 5=scene preserved",
                           "artifacts": "1=severe artifacts, 5=no visible artifacts"},
                "instructions": "Two humans rate independently. Hide model/student labels and private key. Use the same fixed inputs for every team model. Do not fill missing ratings automatically.",
                "requirement_interpretation": "Lab says 30 fixed samples and metrics in both directions; this prepares up to 30 per direction."}
    _json(output_dir / "audit_protocol.json", manifest)
    return manifest


def score_human_audit(rater_one, rater_two):
    def read(path):
        with Path(path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        result = {}
        for row in rows:
            identity = (row["sample_id"], row["direction"])
            if identity in result:
                raise ValueError("Duplicate rating sample")
            values = []
            for field in ("style", "content", "artifacts"):
                if row.get(field, "") not in ("1", "2", "3", "4", "5"):
                    raise ValueError(f"Actual human ratings 1–5 required for {identity}/{field}; blank sheets are not results")
                values.append(int(row[field]))
            result[identity] = values
        return result
    first, second = read(rater_one), read(rater_two)
    if not first or first.keys() != second.keys():
        raise ValueError("Both raters must evaluate the same nonempty sample list")
    result = {"raters": 2, "scoring": "1–5 ordinal; quadratic weighted Cohen kappa", "groups": {}}
    for group in ("all",) + DIRECTIONS:
        identities = sorted(k for k in first if group == "all" or k[1] == group)
        if not identities:
            continue
        a, b = np.array([first[k] for k in identities]), np.array([second[k] for k in identities])
        output = {"count": len(identities), "criteria": {}}
        for column, criterion in enumerate(("style", "content", "artifacts")):
            matrix = np.zeros((5, 5))
            for x, y in zip(a[:, column], b[:, column]):
                matrix[x - 1, y - 1] += 1
            matrix /= len(identities)
            expected = np.outer(matrix.sum(1), matrix.sum(0))
            weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16
            denominator = float((weights * expected).sum())
            kappa = 1 - float((weights * matrix).sum()) / denominator if denominator else None
            output["criteria"][criterion] = {"mean_score": float(np.mean([a[:, column], b[:, column]])),
                                                "percent_agreement": float(100 * np.mean(a[:, column] == b[:, column])),
                                                "quadratic_weighted_kappa": kappa,
                                                "kappa_note": "undefined when expected disagreement is zero" if kappa is None else None}
        result["groups"][group] = output
    return result


def run(config: dict, output_dir: Path, device: str, resume: Path | None = None) -> dict:
    config, output_dir = resolve_config(config), Path(output_dir)
    precision, replay_device = _training_options(config, device)
    config["precision"], config["replay_device"] = precision, replay_device
    mode, seed = config.get("mode", "smoke"), int(config.get("seed", 2342))
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and ((output_dir / "last.pt").exists() or (output_dir / "training_log.jsonl").exists()):
        raise FileExistsError("Existing CycleGAN run: resume explicitly or choose a fresh output directory")
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("This independently implemented training loop fixes batch_size=1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.get("cpu_threads") and str(device) == "cpu":
        torch.set_num_threads(int(config["cpu_threads"]))
    data, provenance = prepare_data(config)
    restored = None
    if resume is not None:
        restored = torch.load(resume, map_location="cpu", weights_only=False)
        for field in RESUME_FIELDS:
            if _recipe_value(restored["config"], field) != _recipe_value(config, field):
                raise ValueError(f"Resume must preserve {field}; create a new run for a changed training recipe")
        if restored["state"]["manifest_fingerprint"] != provenance["manifest_fingerprint"]:
            raise ValueError("Resume data manifest mismatch")
        existing_log = output_dir / "training_log.jsonl"
        if existing_log.exists():
            last_line = None
            with existing_log.open() as handle:
                for line in handle:
                    if line.strip():
                        last_line = line
            if last_line and json.loads(last_line)["step"] > restored["state"]["global_step"]:
                raise ValueError("Log extends beyond the requested checkpoint; resume into a new directory to preserve run history")
        _preserve_best_checkpoint(resume, output_dir, restored, provenance)
    _json(output_dir / "resolved_config.json", config)
    _json(output_dir / "data_manifest.json", provenance)
    models = build_models(config, device)
    optimizers = {
        "generators": torch.optim.Adam(list(models["G_photo_to_monet"].parameters()) + list(models["G_monet_to_photo"].parameters()),
                                       lr=config.get("learning_rate", 1e-4), betas=tuple(config.get("betas", [.5, .999]))),
        "discriminators": torch.optim.Adam(list(models["D_photo"].parameters()) + list(models["D_monet"].parameters()),
                                          lr=config.get("learning_rate", 1e-4), betas=tuple(config.get("betas", [.5, .999])))}
    steps_per_epoch = max(len(data["train_photo"]), len(data["train_monet"]))
    epochs = int(config.get("epochs", 30))
    constant = int(config.get("constant_epochs", epochs // 2))
    if epochs < 1 or not 0 <= constant <= epochs:
        raise ValueError("Need epochs>=1 and 0<=constant_epochs<=epochs")
    def decay(step):
        epoch_fraction = step / steps_per_epoch
        return max(0., 1. - max(0., epoch_fraction - constant) / max(1, epochs - constant))
    schedulers = {name: torch.optim.lr_scheduler.LambdaLR(opt, decay) for name, opt in optimizers.items()}
    pool_device = device if replay_device == "device" else "cpu"
    pools = {domain: ReplayPool(config.get("replay_size", 50), device=pool_device) for domain in ("photo", "monet")}
    state = {"global_step": 0, "training_seconds": 0., "nan_events": 0, "peak_gpu_memory_bytes": 0,
             "best_validation_score": None, "manifest_fingerprint": provenance["manifest_fingerprint"]}
    if restored is not None:
        load_checkpoint(restored, models, optimizers, schedulers, pools)
        state = restored["state"]
        del restored
    target_steps = epochs * steps_per_epoch
    if config.get("max_steps") is not None:
        target_steps = min(target_steps, int(config["max_steps"]))
    if target_steps <= state["global_step"]:
        raise ValueError("No new updates requested; increase max_steps within the existing epoch schedule")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    log_path = output_dir / "training_log.jsonl"
    session_start = time.perf_counter()
    session_initial_step = state["global_step"]
    print(f"cyclegan/{mode} training device={device} step={session_initial_step}/{target_steps}; "
          "ETA estimates cover remaining training steps; evaluation/export may add time", flush=True)
    interval = max(1, int(config.get("checkpoint_every", 500)))
    validation_epochs = max(1, int(config.get("validation_every_epochs", 5)))
    cached_epoch, order = None, None
    while state["global_step"] < target_steps:
        step = state["global_step"]
        epoch, offset = divmod(step, steps_per_epoch)
        if epoch != cached_epoch:
            order = list(range(steps_per_epoch))
            random.Random(seed + epoch).shuffle(order)
            cached_epoch = epoch
        larger = "photo" if len(data["train_photo"]) >= len(data["train_monet"]) else "monet"
        ids = {larger: order[offset], "monet" if larger == "photo" else "photo": random.randrange(len(data["train_monet" if larger == "photo" else "train_photo"]))}
        photo = data["train_photo"][ids["photo"]].unsqueeze(0).to(device)
        monet = data["train_monet"][ids["monet"]].unsqueeze(0).to(device)
        _sync(device)
        start = time.perf_counter()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for name in ("D_photo", "D_monet"):
            models[name].requires_grad_(False)
        with _training_autocast(precision):
            output = cycle_forward(models, photo, monet)
            losses = generator_losses(models, photo, monet, output, config.get("cycle_weight", 10.), config.get("identity_weight", 5.))
        losses["generator_total"].backward()
        for name in ("D_photo", "D_monet"):
            models[name].requires_grad_(True)
        with _training_autocast(precision):
            for domain, real in (("photo", photo), ("monet", monet)):
                losses[f"discriminator_{domain}"] = discriminator_loss(models[f"D_{domain}"], real, pools[domain].query(output[f"fake_{domain}"]))
        (losses["discriminator_photo"] + losses["discriminator_monet"]).backward()
        norms = {name: _grad_norm(model) for name, model in models.items()}
        scalar_losses = {name: float(value.detach()) for name, value in losses.items()}
        if not all(math.isfinite(x) for x in list(norms.values()) + list(scalar_losses.values())):
            state["nan_events"] += 1
            _json(output_dir / "training_failure.json", {"step": step, "reason": "Non-finite loss or gradients; update was not applied", "nan_events": state["nan_events"]})
            raise FloatingPointError("CycleGAN produced non-finite loss/gradients; see training_failure.json and last completed checkpoint")
        for optimizer in optimizers.values():
            optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()
        _sync(device)
        duration = time.perf_counter() - start
        state["global_step"] += 1
        state["training_seconds"] += duration
        if str(device).startswith("cuda"):
            state["peak_gpu_memory_bytes"] = max(state["peak_gpu_memory_bytes"], torch.cuda.max_memory_allocated())
        record = {"step": state["global_step"], "epoch_fraction": state["global_step"] / steps_per_epoch,
                  "losses": scalar_losses, "gradient_norms": norms, "nan_events": state["nan_events"],
                  "step_seconds": duration, "source_images_per_second": 2 / max(duration, 1e-9),
                  "learning_rate_next_step": optimizers["generators"].param_groups[0]["lr"]}
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        current = state["global_step"]
        if current % max(1, int(config.get("log_every", 10))) == 0 or current == target_steps:
            elapsed = time.perf_counter() - session_start
            completed = current - session_initial_step
            eta = elapsed / completed * max(0, target_steps - current)
            print(f"cyclegan/{mode} epoch={epoch + 1}/{epochs} batch={offset + 1}/{steps_per_epoch} "
                  f"step={current}/{target_steps} G_loss={scalar_losses['generator_total']:.4f} "
                  f"D_photo_loss={scalar_losses['discriminator_photo']:.4f} "
                  f"D_monet_loss={scalar_losses['discriminator_monet']:.4f} "
                  f"step_seconds={duration:.3f} elapsed={elapsed:.1f}s eta_steps_est={eta:.1f}s", flush=True)
        epoch_done = current % steps_per_epoch == 0
        if current % interval == 0 or current == target_steps or epoch_done:
            save_grid(models, data, output_dir / "grids" / f"step_{current:07d}.png", device)
            save_checkpoint(output_dir / "last.pt", models, optimizers, schedulers, pools, config, state)
        if provenance["class_results"] and config.get("select_best", True) and epoch_done and (current // steps_per_epoch) % validation_epochs == 0:
            print(f"cyclegan/{mode} step={current} validating", flush=True)
            preserved = rng_state()
            validation = evaluate_models(models, data, provenance, output_dir / "validation" / f"step_{current:07d}", device, "val", config.get("evaluation", {}))
            restore_rng(preserved)
            kids = [validation["directions"][d]["metrics"]["kid"] for d in DIRECTIONS]
            if all(x["status"] == "computed" for x in kids):
                value = sum(x["value"] for x in kids) / 2
                if state["best_validation_score"] is None or value < state["best_validation_score"]:
                    state["best_validation_score"] = value
                    state["best_selection"] = {"metric": "mean_validation_KID_both_directions", "step": current,
                                               "value": value, "validation_only": True}
                    save_checkpoint(output_dir / "best.pt", models, optimizers, schedulers, pools, config, state)
            save_checkpoint(output_dir / "last.pt", models, optimizers, schedulers, pools, config, state)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
    report = {"part": 3, "mode": mode, "device": str(device), "precision": precision, "replay_device": replay_device,
              "seed": seed, "data_kind": provenance["kind"],
              "class_results": provenance["class_results"], "completed_updates": state["global_step"],
              "completed_epoch_fraction": state["global_step"] / steps_per_epoch, "steps_per_epoch": steps_per_epoch,
              "epoch_definition": "one shuffled pass through larger training domain; smaller domain sampled independently with replacement",
              "parameter_counts": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
              "training_seconds": state["training_seconds"], "session_wall_seconds": time.perf_counter() - session_start,
              "training_source_images_per_second": 2 * state["global_step"] / max(state["training_seconds"], 1e-9),
              "throughput_definition": "two source images per complete paired-domain G/D update; excludes data loading and evaluation",
              "peak_gpu_memory_bytes": state["peak_gpu_memory_bytes"] if str(device).startswith("cuda") else None,
              "memory_note": "CUDA peak allocated training memory; CPU/MPS metric unavailable" if not str(device).startswith("cuda") else "CUDA peak allocated memory; not reserved memory",
              "nan_events": state["nan_events"], "checkpoint": str(output_dir / "last.pt"),
              "best_checkpoint": str(output_dir / "best.pt") if (output_dir / "best.pt").exists() else None,
              "best_selection": state.get("best_selection"), "metrics_status": "Run evaluate_checkpoint for held-out evaluation; no scores inferred from loss",
              "human_audit_status": "Requires independent real ratings by two humans", "kaggle_submission_status": "Not submitted; actual class format unverified"}
    if config.get("evaluate_after_run", False):
        evaluation = evaluate_models(models, data, provenance, output_dir / f"evaluation_step_{state['global_step']:07d}", device, "val", config.get("evaluation", {}))
        report["evaluation"] = evaluation
    _json(output_dir / "run_summary.json", report)
    return report


def evaluate_checkpoint(checkpoint, config, output_dir, device, split="test"):
    config = resolve_config(config)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data, provenance = prepare_data(config)
    if saved["state"]["manifest_fingerprint"] != provenance["manifest_fingerprint"]:
        raise ValueError("Evaluation manifests differ from the checkpoint manifests")
    models = build_models(saved["config"], device)
    load_checkpoint(saved, models, restore_random=False)
    checkpoint_step = saved["state"]["global_step"]
    del saved
    result = evaluate_models(models, data, provenance, output_dir, device, split, config.get("evaluation", {}))
    result["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    result["checkpoint_step"] = checkpoint_step
    _json(Path(output_dir) / "metrics.json", result)
    make_human_audit(output_dir, Path(output_dir) / "human_audit", config.get("seed", 2342))
    return result


@torch.no_grad()
def export_checkpoint(checkpoint, input_dir, manifest, output_dir, device, direction="photo_to_monet"):
    """Direct inference PNGs only. No assumptions about class Kaggle submission schema."""
    if direction not in DIRECTIONS:
        raise ValueError(direction)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["state"]["manifest_fingerprint"].startswith("synthetic:"):
        raise ValueError("Synthetic rehearsal checkpoints cannot produce class submission exports")
    root, output_dir = Path(input_dir).resolve(), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Export requires an empty directory")
    entries = [x.strip() for x in Path(manifest).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if not entries or len(set(entries)) != len(entries):
        raise ValueError("Export manifest must contain unique, nonempty relative image paths")
    paths = [(root / x).resolve() for x in entries]
    if any(Path(x).is_absolute() for x in entries) or any(not x.is_relative_to(root) or not x.is_file() for x in paths):
        raise ValueError("Invalid export input manifest")
    cfg = saved["config"]
    model = Generator(cfg.get("base_channels", 64), cfg.get("residual_blocks", 9)).to(device).eval()
    model.load_state_dict(saved["models"][f"G_{direction}"])
    dataset = ImageDomain(paths, cfg.get("image_size", 256))
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        _pil(model(dataset[i].unsqueeze(0).to(device))[0]).save(output_dir / f"{i:06d}.png")
    result = {"count": len(paths), "direction": direction, "image_size": cfg.get("image_size", 256),
              "format": "RGB PNG", "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "inputs": entries, "submission_ready": False,
              "note": "Neutral direct-inference export only. Verify class filename/count/resolution/format rules before packaging or submitting."}
    _json(output_dir / "export_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "export", "audit-score"))
    parser.add_argument("--config", type=Path, default=task_config_path("cyclegan"))
    parser.add_argument("--mode", choices=("smoke", "rehearsal", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--direction", choices=DIRECTIONS, default="photo_to_monet")
    parser.add_argument("--rater-one", type=Path)
    parser.add_argument("--rater-two", type=Path)
    args = parser.parse_args()
    if args.action == "audit-score":
        if not args.rater_one or not args.rater_two:
            parser.error("audit-score requires --rater-one and --rater-two")
        result = score_human_audit(args.rater_one, args.rater_two)
        _json(args.output_dir / "human_agreement.json", result)
    else:
        config = json.loads(Path(args.config).read_text())
        config = {**config[args.mode], "mode": args.mode} if "smoke" in config else {**config, "mode": args.mode}
        if args.action == "train":
            result = run(config, args.output_dir, args.device, args.resume)
        elif args.action == "evaluate":
            if not args.checkpoint:
                parser.error("evaluate requires --checkpoint")
            result = evaluate_checkpoint(args.checkpoint, config, args.output_dir, args.device, args.split)
        else:
            if not all((args.checkpoint, args.input_dir, args.manifest)):
                parser.error("export requires --checkpoint, --input-dir and --manifest")
            result = export_checkpoint(args.checkpoint, args.input_dir, args.manifest, args.output_dir, args.device, args.direction)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
