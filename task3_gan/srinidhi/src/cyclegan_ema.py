"""Isolated eight-epoch CycleGAN continuation with paired generator EMA.

EMA is an inference-only observer: it never supplies training activations, losses,
gradients, replay images or randomness. The existing cyclegan module is unchanged.
Reference: https://arxiv.org/abs/1806.04498 (our beta/schedule are chosen settings).
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from . import cyclegan as cg


TRAINER = "cyclegan_ema_v1"
GENERATORS = ("G_photo_to_monet", "G_monet_to_photo")
METRIC = "mean_validation_KID_both_directions"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_save(payload, path, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite immutable candidate: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
    if replace:
        temporary.replace(path)
    else:
        _require(not path.exists(), f"Candidate appeared while saving: {path}")
        temporary.rename(path)


def _cpu_states(models):
    return {name: {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            for name, model in models.items()}


class GeneratorEMA:
    """FP32 generator copies with a checkpointed update counter; consumes no RNG."""
    def __init__(self, models, beta=.999):
        _require(isinstance(beta, (float, int)) and not isinstance(beta, bool)
                 and math.isfinite(beta) and 0 < beta < 1, "EMA beta must be finite and between zero and one")
        self.beta, self.updates = float(beta), 0
        self.models = {name: copy.deepcopy(models[name]).float().eval().requires_grad_(False) for name in GENERATORS}

    @torch.no_grad()
    def update(self, models):
        for name, averaged in self.models.items():
            raw_parameters = dict(models[name].named_parameters())
            for key, parameter in averaged.named_parameters():
                parameter.mul_(self.beta).add_(raw_parameters[key].detach().float(), alpha=1 - self.beta)
            # Match the official generator-EMA convention: copy buffers rather
            # than averaging integer counters or normalization state.
            raw_buffers = dict(models[name].named_buffers())
            for key, buffer in averaged.named_buffers():
                buffer.copy_(raw_buffers[key])
        self.updates += 1

    def state_dict(self):
        return {"beta": self.beta, "updates": self.updates, "models": _cpu_states(self.models)}

    def load_state_dict(self, saved):
        _require(saved.get("beta") == self.beta and type(saved.get("updates")) is int and saved["updates"] >= 0,
                 "EMA beta/update counter mismatch")
        _require(set(saved["models"]) == set(GENERATORS), "EMA must contain both generators")
        for name, model in self.models.items():
            for tensor in saved["models"][name].values():
                _require(not tensor.is_floating_point() or (tensor.dtype == torch.float32 and torch.isfinite(tensor).all()),
                         "EMA checkpoint must have finite FP32 floating tensors")
            model.load_state_dict(saved["models"][name], strict=True)
        self.updates = saved["updates"]


def evaluation_models(models, ema, variant):
    _require(variant in ("raw", "ema"), "Unknown candidate variant")
    # No swapping or mutation of online training weights, even on evaluation failure.
    return dict(models) if variant == "raw" else {**models, **ema.models}


def candidate_payload(models, ema, config, state, variant):
    chosen = evaluation_models(models, ema, variant)
    states = _cpu_states(chosen)
    _require(all(not tensor.is_floating_point() or torch.isfinite(tensor).all()
                 for values in states.values() for tensor in values.values()), "Non-finite candidate weights")
    candidate_state = {"global_step": state["global_step"], "manifest_fingerprint": state["manifest_fingerprint"],
                       "initialization": copy.deepcopy(state["initialization"]), "candidate_variant": variant,
                       "ema_beta": ema.beta, "ema_updates": ema.updates}
    return {"format_version": 1, "trainer": TRAINER, "inference_only": True,
            "config": copy.deepcopy(config), "state": candidate_state, "models": states,
            "weights_note": "Paired EMA generators and raw discriminators" if variant == "ema" else "All four raw networks",
            "resume_note": "Inference candidate only; exact continuation requires this run's own last.pt"}


def _export_candidate(path, models, ema, config, state, variant):
    expected = candidate_payload(models, ema, config, state, variant)
    if path.exists():
        # Recovery after a crash between candidate export and validation receipt.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        _require(saved.get("trainer") == TRAINER and saved.get("inference_only") is True
                 and saved.get("state") == expected["state"]
                 and _recipe(saved["config"]) == _recipe(config), "Existing candidate has incompatible provenance")
        _require(set(saved["models"]) == set(expected["models"]), "Existing candidate network inventory differs")
        for name, values in expected["models"].items():
            _require(saved["models"][name].keys() == values.keys()
                     and all(torch.equal(saved["models"][name][key], value) for key, value in values.items()),
                     "Existing candidate weights differ from the resumed online/EMA state")
    else:
        _atomic_save(expected, path)
    return _sha256(path)


def _recipe(config):
    return {key: value for key, value in config.items() if key != "max_steps"}


def _config(config, device):
    config = copy.deepcopy(cg.resolve_config(config))
    config.setdefault("ema_beta", .999)
    config.setdefault("validation_epochs", [4, 8])
    config.setdefault("monet_translation_ratio", 0.)
    config.setdefault("max_steps", None)
    precision, replay_device = cg._training_options(config, device)
    config.update(precision=precision, replay_device=replay_device)
    _require(config.get("epochs") == 8 and config.get("constant_epochs") == 4,
             "EMA continuation requires the frozen eight-epoch/four-constant schedule")
    _require(config["validation_epochs"] == [4, 8], "EMA validation epochs must remain [4, 8]")
    _require(config["ema_beta"] == .999, "This experiment fixes ema_beta=.999")
    _require(config["monet_translation_ratio"] == 0, "EMA experiment disables translation augmentation")
    _require(config.get("batch_size", 1) == 1, "EMA trainer requires batch_size=1")
    lr = config.get("learning_rate")
    _require(isinstance(lr, (float, int)) and not isinstance(lr, bool) and math.isfinite(lr) and lr > 0,
             "learning_rate must be positive and finite")
    _require(config["max_steps"] is None or (type(config["max_steps"]) is int and config["max_steps"] > 0),
             "max_steps must be a positive cumulative update cap")
    _require(not config.get("evaluate_after_run", False), "No implicit evaluation after this frozen candidate schedule")
    return config


def _save_last(output, models, optimizers, schedulers, pools, ema, config, state):
    payload = {"format_version": 1, "trainer": TRAINER, "inference_only": False, "config": config,
               "state": state, "models": _cpu_states(models),
               "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
               "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
               "replay_pools": {k: v.state_dict() for k, v in pools.items()},
               "rng": cg.rng_state(), "ema": ema.state_dict()}
    _atomic_save(payload, output / "last.pt", replace=True)


def _scores(metrics, provenance):
    _require(metrics.get("split") == "val" and metrics.get("manifest_fingerprint") == provenance["manifest_fingerprint"],
             "Validation uses a different split or manifest")
    scores = {}
    for direction in cg.DIRECTIONS:
        source, target = direction.split("_to_")
        entry = metrics["directions"][direction]
        _require(entry["source_count"] == entry["generated_count"] == provenance["counts"][f"val_{source}"]
                 and entry["real_reference_count"] == provenance["counts"][f"val_{target}"], "Incomplete validation images")
        metric = entry["metrics"]["kid"]
        _require(metric.get("status") == "computed" and isinstance(metric.get("value"), (float, int))
                 and not isinstance(metric["value"], bool) and math.isfinite(metric["value"]),
                 "Both validation KID values must be computed and finite")
        scores[direction] = metric["value"]
    return scores


def _check_receipt(directory, checkpoint_hash, state, variant, provenance):
    receipt = _read(directory / "receipt.json")
    _require(receipt.get("status") == "complete" and receipt.get("checkpoint_sha256") == checkpoint_hash
             and receipt.get("step") == state["global_step"] and receipt.get("candidate_variant") == variant
             and receipt.get("source_checkpoint_sha256") == state["initialization"]["checkpoint_sha256"]
             and receipt.get("manifest_fingerprint") == provenance["manifest_fingerprint"], "Validation receipt identity mismatch")
    _require(receipt["metrics_sha256"] == _sha256(directory / "metrics.json"), "Validation metric receipt changed")
    metrics = _read(directory / "metrics.json")
    _require(metrics.get("checkpoint_sha256") == checkpoint_hash
             and metrics.get("checkpoint_step") == state["global_step"]
             and metrics.get("candidate_variant") == variant, "Validation metric checkpoint binding differs")
    scores = _scores(metrics, provenance)
    _require(receipt.get("directional_kid") == scores and receipt.get("mean_validation_kid") == sum(scores.values()) / 2,
             "Validation receipt scores differ")
    return receipt


def _validate_variant(models, ema, data, provenance, output, device, config, state, variant):
    step = state["global_step"]
    destination = output / "validation" / f"step_{step:07d}" / variant
    if step == 0:
        _require(variant == "raw", "Baseline has only the unchanged raw source")
        checkpoint = Path(state["initialization"]["checkpoint"])
        checkpoint_hash = _sha256(checkpoint)
        _require(checkpoint_hash == state["initialization"]["checkpoint_sha256"], "Source changed before baseline validation")
    else:
        checkpoint = output / "candidates" / f"step_{step:07d}" / f"{variant}.pt"
        checkpoint_hash = _export_candidate(checkpoint, models, ema, config, state, variant)
    if destination.exists():
        return _check_receipt(destination, checkpoint_hash, state, variant, provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{variant}-attempt-", dir=destination.parent))
    chosen = evaluation_models(models, ema, variant)
    preserved_rng, preserved_modes = cg.rng_state(), {name: model.training for name, model in chosen.items()}
    try:
        metrics = cg.evaluate_models(chosen, data, provenance, staging, device, "val", config.get("evaluation", {}))
        scores = _scores(metrics, provenance)
        metrics.update(checkpoint_sha256=checkpoint_hash, checkpoint_step=step, candidate_variant=variant,
                       ema_beta=ema.beta, ema_updates=ema.updates)
        cg._json(staging / "metrics.json", metrics)
        receipt = {"status": "complete", "trainer": TRAINER, "step": step, "candidate_variant": variant,
                   "checkpoint_sha256": checkpoint_hash, "metrics_sha256": _sha256(staging / "metrics.json"),
                   "source_checkpoint_sha256": state["initialization"]["checkpoint_sha256"],
                   "manifest_fingerprint": provenance["manifest_fingerprint"], "directional_kid": scores,
                   "mean_validation_kid": sum(scores.values()) / 2,
                   "checkpoint": str(checkpoint), "metrics_path": str(destination / "metrics.json"),
                   "weights_note": "Unchanged starting checkpoint" if step == 0 else
                       ("Paired EMA generators and raw discriminators" if variant == "ema" else "All four raw networks")}
        cg._json(staging / "receipt.json", receipt)
        _require(not destination.exists(), "Validation destination appeared during evaluation")
        staging.rename(destination)
    finally:
        cg.restore_rng(preserved_rng)
        for name, model in chosen.items():
            model.train(preserved_modes[name])
    return _check_receipt(destination, checkpoint_hash, state, variant, provenance)


def _deadline(value):
    if value is None:
        return None
    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(moment.tzinfo is not None, "deadline_utc must include a timezone")
    return moment.timestamp()


def run(config, output_dir, device, resume=None, warm_start=None, *, deadline_utc=None, wall_budget_seconds=None):
    """Train a fresh matched continuation or resume only its own online last.pt.

    deadline_utc is cooperative: checked before every update/evaluation. A metric
    call already underway completes before control returns; it is never killed.
    """
    invoked_at = time.monotonic()
    _require(wall_budget_seconds is None or (isinstance(wall_budget_seconds, (float, int))
             and not isinstance(wall_budget_seconds, bool) and math.isfinite(wall_budget_seconds)
             and wall_budget_seconds >= 0), "wall_budget_seconds must be finite and nonnegative")
    _require((resume is None) != (warm_start is None), "Supply exactly one of warm_start or resume")
    config, output = _config(config, device), Path(output_dir).resolve()
    deadline = _deadline(deadline_utc)
    expired = lambda: ((deadline is not None and time.time() >= deadline)
                       or (wall_budget_seconds is not None and time.monotonic() - invoked_at >= wall_budget_seconds))
    if (output / "training_failure.json").exists():
        raise ValueError("A failed non-finite training run cannot be silently resumed")
    if resume is None:
        # Wrapper logs/provenance may exist; trainer artifacts must all be fresh.
        for name in ("last.pt", "training_log.jsonl", "resolved_config.json", "data_manifest.json", "trainer_summary.json",
                     "validation", "candidates", "initialization.json"):
            if (output / name).exists():
                raise FileExistsError(f"Fresh training cannot overwrite {output / name}")
        _require(not Path(warm_start).resolve().is_relative_to(output), "Warm-start source must be outside the new run")
    else:
        _require(Path(resume).resolve() == output / "last.pt", "Resume requires this run's own last.pt")
    seed = int(config.get("seed", 2342))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.get("cpu_threads") and str(device) == "cpu":
        torch.set_num_threads(int(config["cpu_threads"]))
    data, provenance = cg.prepare_data(config)
    steps_per_epoch = max(len(data["train_photo"]), len(data["train_monet"]))
    full_steps = 8 * steps_per_epoch
    target = min(full_steps, config["max_steps"]) if config["max_steps"] is not None else full_steps
    restored = None
    if resume is not None:
        restored = torch.load(resume, map_location="cpu", weights_only=False)
        _require(restored.get("trainer") == TRAINER and restored.get("inference_only") is False
                 and restored.get("format_version") == 1, "Only this trainer's resumable online checkpoint is accepted")
        _require(_recipe(restored["config"]) == _recipe(config), "Resume must preserve the complete immutable recipe")
        _require(_recipe(_read(output / "resolved_config.json")) == _recipe(config), "Saved immutable recipe changed")
        state = restored["state"]
        _require(state["run_directory"] == str(output), "Resume checkpoint belongs to a different run directory")
        _require(state["manifest_fingerprint"] == provenance["manifest_fingerprint"], "Resume data manifest mismatch")
        _require(_read(output / "data_manifest.json") == provenance, "Saved data-manifest evidence changed")
        log_steps = [json.loads(line)["step"] for line in (output / "training_log.jsonl").read_text().splitlines() if line.strip()] if (output / "training_log.jsonl").exists() else []
        _require(log_steps == list(range(1, state["global_step"] + 1)), "Training log must match checkpoint exactly; log-ahead or missing updates refused")
        source = Path(state["initialization"]["checkpoint"])
        _require(_sha256(source) == state["initialization"]["checkpoint_sha256"], "Frozen source changed on resume")
        for receipt in state["validation_history"]:
            directory = output / "validation" / f"step_{receipt['step']:07d}" / receipt["candidate_variant"]
            _require(_read(directory / "receipt.json") == receipt and _sha256(directory / "metrics.json") == receipt["metrics_sha256"],
                     "Completed validation evidence changed on resume")
    else:
        initialized, initialization = cg._warm_start_checkpoint(warm_start, config, provenance)
        expected = config.get("expected_source_sha256")
        _require(expected is None or initialization["checkpoint_sha256"] == expected, "Warm-start checkpoint does not match expected source SHA256")
        state = {"global_step": 0, "training_seconds": 0., "nan_events": 0, "peak_gpu_memory_bytes": 0,
                 "manifest_fingerprint": provenance["manifest_fingerprint"], "initialization": initialization,
                 "run_directory": str(output), "validation_history": [],
                 "pending_validation": {"step": 0, "variants": ["raw"]}}
    _require(target >= state["global_step"] and (target > state["global_step"] or state["pending_validation"] is not None),
             "No new updates or pending validation requested")
    models = cg.build_models(config, device)
    optimizers = {"generators": torch.optim.Adam([p for name in GENERATORS for p in models[name].parameters()], lr=config["learning_rate"], betas=tuple(config["betas"])),
                  "discriminators": torch.optim.Adam([p for name in ("D_photo", "D_monet") for p in models[name].parameters()], lr=config["learning_rate"], betas=tuple(config["betas"]))}
    def decay(step):
        return max(0., 1. - max(0., step / steps_per_epoch - 4) / 4)
    schedulers = {name: torch.optim.lr_scheduler.LambdaLR(opt, decay) for name, opt in optimizers.items()}
    pools = {name: cg.ReplayPool(config.get("replay_size", 50), device=device if config["replay_device"] == "device" else "cpu") for name in ("photo", "monet")}
    if restored is None:
        cg.load_checkpoint(initialized, models, restore_random=False)
        del initialized
    else:
        cg.load_checkpoint(restored, models, optimizers, schedulers, pools)
    ema = GeneratorEMA(models, config["ema_beta"])
    if restored is not None:
        ema.load_state_dict(restored["ema"])
        _require(ema.updates == state["global_step"], "EMA update count must match successful online updates")
        del restored
    output.mkdir(parents=True, exist_ok=True)
    cg._json(output / "resolved_config.json", config)
    cg._json(output / "data_manifest.json", provenance)
    cg._json(output / "initialization.json", state["initialization"])
    save = lambda: _save_last(output, models, optimizers, schedulers, pools, ema, config, state)
    session_start, initial_step = time.perf_counter(), state["global_step"]
    save()

    def finish_pending():
        pending = state["pending_validation"]
        if pending is None:
            return True
        _require(pending["step"] == state["global_step"]
                 and pending["variants"] in (["raw"], ["raw", "ema"], ["ema"]), "Invalid pending validation state")
        while pending["variants"]:
            if expired():
                save(); return False
            variant = pending["variants"][0]
            print(f"{TRAINER} step={state['global_step']} validation={variant}", flush=True)
            receipt = _validate_variant(models, ema, data, provenance, output, device, config, state, variant)
            state["validation_history"].append(receipt)
            if state["global_step"] == 0:
                state["initialization"]["baseline_validation"] = {"status": "computed", "step": 0,
                    "metric": METRIC, "value": receipt["mean_validation_kid"], "directions": receipt["directional_kid"]}
                cg._json(output / "initialization.json", state["initialization"])
            pending["variants"].pop(0)
            if not pending["variants"]:
                state["pending_validation"] = None
            save()
        return True

    stopped = not finish_pending()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    cached_epoch, order = None, None
    while not stopped and state["global_step"] < target:
        if expired():
            stopped = True; break
        step = state["global_step"]
        epoch, offset = divmod(step, steps_per_epoch)
        if epoch != cached_epoch:
            order = list(range(steps_per_epoch)); random.Random(seed + epoch).shuffle(order); cached_epoch = epoch
        larger = "photo" if len(data["train_photo"]) >= len(data["train_monet"]) else "monet"
        smaller = "monet" if larger == "photo" else "photo"
        ids = {larger: order[offset], smaller: random.randrange(len(data[f"train_{smaller}"]))}
        photo = data["train_photo"][ids["photo"]].unsqueeze(0).to(device)
        monet = data["train_monet"][ids["monet"]].unsqueeze(0).to(device)
        cg._sync(device); started = time.perf_counter()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for name in ("D_photo", "D_monet"):
            models[name].requires_grad_(False)
        with cg._training_autocast(config["precision"]):
            forward = cg.cycle_forward(models, photo, monet)
            losses = cg.generator_losses(models, photo, monet, forward, config["cycle_weight"], config["identity_weight"], 0.)
        losses["generator_total"].backward()
        for name in ("D_photo", "D_monet"):
            models[name].requires_grad_(True)
        with cg._training_autocast(config["precision"]):
            for domain, real in (("photo", photo), ("monet", monet)):
                losses[f"discriminator_{domain}"] = cg.discriminator_loss(models[f"D_{domain}"], real, pools[domain].query(forward[f"fake_{domain}"]))
        (losses["discriminator_photo"] + losses["discriminator_monet"]).backward()
        norms = {name: cg._grad_norm(model) for name, model in models.items()}
        scalars = {name: float(value.detach()) for name, value in losses.items()}
        if not all(math.isfinite(value) for value in [*norms.values(), *scalars.values()]):
            state["nan_events"] += 1
            cg._json(output / "training_failure.json", {"step": step, "reason": "Non-finite loss/gradient; update was not applied", "nan_events": state["nan_events"]})
            raise FloatingPointError("Non-finite online loss/gradient; last successful checkpoint retained")
        for optimizer in optimizers.values():
            optimizer.step()
        ema.update(models)
        for scheduler in schedulers.values():
            scheduler.step()
        cg._sync(device); duration = time.perf_counter() - started
        state["global_step"] += 1; state["training_seconds"] += duration
        if str(device).startswith("cuda"):
            state["peak_gpu_memory_bytes"] = max(state["peak_gpu_memory_bytes"], torch.cuda.max_memory_allocated())
        current = state["global_step"]
        record = {"step": current, "epoch_fraction": current / steps_per_epoch, "losses": scalars, "gradient_norms": norms,
                  "nan_events": state["nan_events"], "step_seconds": duration, "source_images_per_second": 2 / max(duration, 1e-9),
                  "learning_rate_next_step": optimizers["generators"].param_groups[0]["lr"], "ema_updates": ema.updates}
        with (output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if current % max(1, int(config.get("log_every", 50))) == 0 or current == target:
            print(f"{TRAINER} step={current}/{target} epoch={current / steps_per_epoch:.4f} G={scalars['generator_total']:.5f} seconds={duration:.3f}", flush=True)
        if current in (4 * steps_per_epoch, 8 * steps_per_epoch):
            state["pending_validation"] = {"step": current, "variants": ["raw", "ema"]}
            save()
            stopped = not finish_pending()
        elif current % max(1, int(config.get("checkpoint_every", 500))) == 0 or current == target:
            save()
        if expired():
            stopped = True
    save()
    expected_validations = {(0, "raw"), *( (epoch * steps_per_epoch, variant) for epoch in (4, 8) for variant in ("raw", "ema") )}
    actual_validations = {(receipt["step"], receipt["candidate_variant"]) for receipt in state["validation_history"]}
    complete = state["global_step"] == full_steps and state["pending_validation"] is None and actual_validations == expected_validations
    summary = {"part": 3, "trainer": TRAINER, "status": "completed" if complete else "partial",
        "stop_reason": "completed" if complete else ("budget_exhausted" if stopped else "max_steps"),
        "completed_updates": state["global_step"], "completed_epoch_fraction": state["global_step"] / steps_per_epoch,
        "steps_per_epoch": steps_per_epoch, "manifest_fingerprint": provenance["manifest_fingerprint"],
        "class_results": provenance["class_results"], "mode": config.get("mode"), "device": str(device),
        "precision": config["precision"], "training_seconds": state["training_seconds"],
        "session_wall_seconds": time.perf_counter() - session_start, "session_initial_step": initial_step,
        "training_source_images_per_second": 2 * state["global_step"] / max(state["training_seconds"], 1e-9),
        "throughput_definition": "Two source images per paired G/D update; includes EMA updates, excludes data loading/evaluation/checkpoint I/O",
        "peak_gpu_memory_bytes": state["peak_gpu_memory_bytes"] if str(device).startswith("cuda") else None,
        "parameter_counts": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        "nan_events": state["nan_events"], "initialization": state["initialization"], "ema_beta": ema.beta, "ema_updates": ema.updates,
        "checkpoint": str(output / "last.pt"), "pending_validation": state["pending_validation"],
        "validation_history": state["validation_history"], "selection_note": "Cross-arm selection is performed by the frozen wrapper; no test/class feedback here"}
    cg._json(output / "trainer_summary.json", summary)
    return summary
