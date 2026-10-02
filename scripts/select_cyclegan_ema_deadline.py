"""Select the five prospective deadline-amendment slots using validation only.

This does not complete the original nine-candidate study. Epoch-eight results
are checked only as evidence that the already-running first arm ended safely.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("frozen_ema_study", ROOT / "scripts/study_cyclegan_ema.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_amendment(output):
    output, protocol, _ = base.load_protocol(output)
    path = output / "deadline_amendment.json"
    require(base.digest(path) == (output / "deadline_amendment.sha256").read_text().strip(), "Frozen deadline amendment changed")
    amendment = base.read_json(path)
    require(amendment.get("format_version") == 1 and amendment.get("kind") == "prospective_access_deadline_amendment"
            and amendment.get("original_protocol_sha256") == base.digest(output / "protocol.json")
            and amendment.get("source_checkpoint_sha256") == protocol["source_checkpoint_sha256"]
            and amendment.get("source_manifest_fingerprint") == protocol["source_manifest_fingerprint"]
            and amendment.get("candidate_epochs") == [4] and amendment.get("excluded_candidate_epochs") == [8]
            and amendment.get("max_candidates") == 5 and amendment.get("arms") == list(base.ARMS)
            and amendment.get("variants") == list(base.VARIANTS)
            and amendment.get("no_expansion_after_results") is True
            and amendment.get("test_or_class_feedback_used_for_amendment") is False
            and amendment.get("original_eight_epoch_study_completion_claimed") is False,
            "Unsupported or inconsistent prospective amendment")
    steps = amendment.get("candidate_steps")
    require(isinstance(steps, list) and len(steps) == 1 and type(steps[0]) is int and steps[0] > 0 and steps[0] % 4 == 0,
            "Exactly one valid epoch-four step is required")
    selection = dict(protocol["selection"], max_candidates=5, require_both_arms_complete=False,
                     require_both_epoch_four_endpoints_complete=True)
    require(amendment.get("selection") == selection, "Amendment changed non-budget selection rules")
    required = amendment.get("required_arm_results", {})
    for arm, status, count, updates in (("lr5e5", "completed", 5, 2 * steps[0]), ("lr2e5", "partial", 3, steps[0])):
        entry = required.get(arm, {})
        require(entry.get("status") == status and entry.get("validation_receipts") == count
                and entry.get("completed_updates") == updates
                and (arm != "lr2e5" or entry.get("stop_reason") == "max_steps"), "Amended completion rule differs")
    try:
        times = [datetime.fromisoformat(amendment[key]) for key in ("created_utc", "training_deadline_utc",
            "class_evaluation_deadline_utc", "finalization_deadline_utc", "lab_access_deadline_utc")]
        valid_times = all(value.tzinfo is not None for value in times) and all(a < b for a, b in zip(times, times[1:]))
        valid_times = valid_times and datetime.fromisoformat(protocol["created_utc"]) <= times[0]
    except (KeyError, ValueError, TypeError):
        valid_times = False
    require(valid_times, "Amendment needs ordered timezone-aware access deadlines")
    evidence = amendment.get("freeze_evidence", {})
    require(evidence.get("nonzero_validation_results_exist") is False
            and evidence.get("nonzero_candidate_checkpoints_exist") is False,
            "Amendment was not frozen before candidate evidence")
    progress = evidence.get("training_progress", {})
    first, second = progress.get("lr5e5", {}), progress.get("lr2e5", {})
    require(type(first.get("step")) is int and 0 <= first["step"] < steps[0] and first.get("nan_events") == 0
            and math.isclose(base.finite(first.get("epoch_fraction"), "freeze epoch"), first["step"] / (steps[0] / 4), rel_tol=1e-12, abs_tol=1e-12)
            and second.get("step") == 0 and second.get("not_started") is True,
            "Freeze progress already passed a candidate endpoint")
    inventory = evidence.get("completed_validation_receipts", [])
    baseline_path = "arms/lr5e5/validation/step_0000000/raw/receipt.json"
    require(len(inventory) == 1 and inventory[0].get("path") == baseline_path
            and inventory[0].get("step") == 0 and inventory[0].get("variant") == "raw",
            "Freeze inventory must contain only the unchanged first-arm baseline")
    baseline = base.read_json(output / baseline_path)
    require(inventory[0].get("sha256") == base.digest(output / baseline_path)
            and inventory[0].get("metrics_sha256") == base.digest((output / baseline_path).with_name("metrics.json"))
            and baseline.get("status") == "complete" and baseline.get("step") == 0 and baseline.get("candidate_variant") == "raw"
            and baseline.get("checkpoint_sha256") == protocol["source_checkpoint_sha256"],
            "Frozen unchanged baseline inventory differs")
    return amendment


def verify_operational_sources(output):
    output = Path(output).resolve()
    path = output / "deadline_operational_sources.json"
    require(base.digest(path) == (output / "deadline_operational_sources.sha256").read_text().strip(), "Operational source receipt changed")
    sources = base.read_json(path)
    require(sources.get("scripts/select_cyclegan_ema_deadline.py") == base.digest(Path(__file__)), "Deadline selector not frozen")
    for relative, expected in sources.items():
        candidate = (base.ROOT / relative).resolve()
        require((Path(relative).is_absolute() or candidate.is_relative_to(base.ROOT))
                and base.digest(candidate) == expected, f"Operational source changed: {relative}")
    return base.digest(path)


def verify_ended(wrapper, final, expected_status):
    require(wrapper == final and wrapper.get("status") == expected_status,
            "Matching ended wrapper/final provenance required")
    try:
        start, end = [datetime.fromisoformat(wrapper[key]) for key in ("started_utc", "ended_utc")]
        valid = start.tzinfo is not None and end.tzinfo is not None and end >= start
    except (KeyError, ValueError, TypeError):
        valid = False
    require(valid and base.finite(wrapper.get("elapsed_seconds"), "elapsed seconds") >= 0,
            "Ended invocation timestamp and elapsed seconds required")


def verify_candidate(path, config, protocol, step, variant):
    expected = base.digest(path)
    saved = base.load_checkpoint(path)
    state = saved.get("state", {})
    require(saved.get("format_version") == 1 and saved.get("trainer") == "cyclegan_ema_v1"
            and saved.get("inference_only") is True and saved.get("config") == config
            and not any(key in saved for key in ("optimizers", "schedulers", "replay_pools", "rng"))
            and state.get("global_step") == step and state.get("candidate_variant") == variant
            and state.get("ema_beta") == .999 and state.get("ema_updates") == step
            and state.get("manifest_fingerprint") == protocol["source_manifest_fingerprint"]
            and state.get("initialization", {}).get("checkpoint_sha256") == protocol["source_checkpoint_sha256"],
            "Candidate checkpoint recipe/identity differs")
    base.verify_models(saved)
    del saved
    require(base.digest(path) == expected, "Candidate changed during verification")
    return expected


def verify_arm(output, protocol, amendment, arm, source):
    output = Path(output).resolve()
    run = output / protocol["arms"][arm]["run_dir"]
    require(not (run / "training_failure.json").exists(), "Recorded training failure blocks selection")
    wrapper = base.read_json(run / "run_summary.json")
    provenance_paths = sorted((run / "provenance").glob("*.json"))
    require(bool(provenance_paths), "Ended provenance required")
    first = arm == "lr5e5"
    status, epochs = ("completed", 8) if first else ("partial", 4)
    verify_ended(wrapper, base.read_json(provenance_paths[-1]), status)
    config = base.read_json(run / "resolved_config.json")
    planned = base.read_json(output / protocol["arms"][arm]["config"])
    manifest = base.read_json(run / "data_manifest.json")
    summary = wrapper.get("summary", {})
    per_epoch = max(manifest["counts"]["train_photo"], manifest["counts"]["train_monet"])
    candidate_step, expected_updates = 4 * per_epoch, epochs * per_epoch
    require(candidate_step == amendment["candidate_steps"][0], "Candidate step differs from frozen amendment")
    require(wrapper.get("task") == "cyclegan_ema" and wrapper.get("arm") == arm
            and wrapper.get("config") == config and base.same_recipe(config, planned)
            and config.get("epochs") == 8 and config.get("constant_epochs") == 4
            and config.get("max_steps") == (None if first else candidate_step)
            and wrapper.get("source_sha256") == protocol["source_code_hashes"]
            and wrapper.get("protocol_sha256") == base.digest(output / "protocol.json")
            and wrapper.get("budget_sha256") == base.digest(output / "budget.json")
            and wrapper.get("warm_start_source_sha256") == protocol["source_checkpoint_sha256"]
            and bool(wrapper.get("environment")) and summary.get("completed_updates") == expected_updates
            and summary.get("completed_epoch_fraction") == epochs and summary.get("nan_events") == 0
            and summary.get("status") == status and summary.get("stop_reason") == ("completed" if first else "max_steps")
            and not summary.get("pending_validation") and summary.get("trainer") == "cyclegan_ema_v1"
            and summary.get("steps_per_epoch") == per_epoch and summary.get("ema_beta") == .999
            and summary.get("ema_updates") == expected_updates
            and manifest.get("manifest_fingerprint") == protocol["source_manifest_fingerprint"]
            and summary.get("initialization", {}).get("checkpoint_sha256") == protocol["source_checkpoint_sha256"],
            "Arm does not satisfy the exact amended completion requirements")
    require(base.read_json(run / "trainer_summary.json") == summary, "Trainer/wrapper summaries differ")
    log_hash = base.verify_training_log(run / "training_log.jsonl", expected_updates)
    with (run / "training_log.jsonl").open(encoding="utf-8") as handle:
        require(all(json.loads(line).get("ema_updates") == index for index, line in enumerate(handle, 1)),
                "Training log EMA count differs from update count")
    baseline = base.validation_kid(run / "validation/step_0000000/raw/metrics.json", manifest, output)
    receipts = [base.validation_receipt(run / "validation/step_0000000/raw", source, 0, "raw", baseline, protocol, output)]
    rows, excluded = [], []
    for epoch in ([4, 8] if first else [4]):
        step = epoch * per_epoch
        for variant in base.VARIANTS:
            directory = run / "validation" / f"step_{step:07d}" / variant
            candidate = run / "candidates" / f"step_{step:07d}" / f"{variant}.pt"
            row = base.validation_kid(directory / "metrics.json", manifest, output)
            receipts.append(base.validation_receipt(directory, candidate, step, variant, row, protocol, output))
            if epoch == 4:
                checkpoint_hash = verify_candidate(candidate, config, protocol, step, variant)
                rows.append(dict(row, kind="arm", arm=arm, step=step, epoch=epoch, variant=variant,
                                 checkpoint=base.relative_path(candidate, output), checkpoint_sha256=checkpoint_hash))
            else:
                # No epoch-eight numerical result enters any candidate/ranking structure.
                excluded.append({"arm": arm, "epoch": 8, "step": step, "variant": variant,
                                 "receipt_path": row["receipt_path"], "receipt_sha256": row["receipt_sha256"],
                                 "reason": "Prospectively excluded by deadline amendment; completion evidence only"})
    require(summary.get("validation_history") == receipts, "Summary does not bind the exact required validation history")
    expected_slots = {(receipt["step"], receipt["candidate_variant"]) for receipt in receipts}
    actual_slots = set()
    for path in (run / "validation").glob("step_*/*/receipt.json"):
        receipt = base.read_json(path)
        actual_slots.add((receipt.get("step"), receipt.get("candidate_variant")))
    require(actual_slots == expected_slots, "Unexpected validation receipt slots")
    evidence = {"status": status, "completed_updates": expected_updates,
        "run_summary_sha256": base.digest(run / "run_summary.json"), "final_provenance_sha256": base.digest(provenance_paths[-1]),
        "trainer_summary_sha256": base.digest(run / "trainer_summary.json"), "training_log_sha256": log_hash,
        "provenance": [{"path": base.relative_path(path, output), "sha256": base.digest(path)} for path in provenance_paths],
        "validation_receipt_count": len(receipts)}
    return {"baseline": dict(baseline, arm=arm), "candidates": rows, "excluded": excluded, "completion": evidence}


def select(output, write=True):
    output, protocol, source = base.load_protocol(output)
    with base.runner_lock(output / ".study_runner.lock"):
        base.read_budget(output, protocol)
        amendment = load_amendment(output)
        operational_hash = verify_operational_sources(output)
        verified = {arm: verify_arm(output, protocol, amendment, arm, source) for arm in base.ARMS}
        baselines = [verified[arm]["baseline"] for arm in base.ARMS]
        baseline = baselines[0]
        require(all(math.isclose(other["directions"][direction], baseline["directions"][direction], abs_tol=1e-6, rel_tol=1e-6)
                    for other in baselines[1:] for direction in base.DIRECTIONS), "Source baseline disagreement")
        source_row = dict(baseline, kind="unchanged_source", arm=None, variant="source", epoch=None,
                          step=protocol["source_global_step"], checkpoint=base.relative_path(source, output),
                          checkpoint_sha256=protocol["source_checkpoint_sha256"])
        candidates = [source_row, *(row for arm in base.ARMS for row in verified[arm]["candidates"])]
        require(len(candidates) == 5 and all(row["epoch"] in (None, 4) for row in candidates), "Exactly five predeclared slots required")
        for row in candidates:
            row["eligible"] = row["kind"] == "unchanged_source" or all(row["directions"][d] <= baseline["directions"][d] + 1e-8 for d in base.DIRECTIONS)
            row["guardrail_deltas"] = {d: row["directions"][d] - baseline["directions"][d] for d in base.DIRECTIONS}
            row["exclusion_reason"] = None if row["eligible"] else "Directional validation KID exceeds incumbent + 1e-8"
        ranking = sorted([row for row in candidates if row["eligible"]],
                         key=lambda r: (r["value"], r["kind"] != "unchanged_source", r["arm"] or "", r["variant"] == "ema"))
        base.load_protocol(output)
        load_amendment(output)
        require(verify_operational_sources(output) == operational_hash, "Operational source receipt changed during selection")
        result = {"format_version": 1, "created_utc": base.utc_now(), "status": "completed_amended_study",
            "selection_status": "validation_selected_pending_visual_review", "original_study_completed": False,
            "split": "val", "metric": base.METRIC, "protocol_sha256": base.digest(output / "protocol.json"),
            "amendment_path": "deadline_amendment.json", "amendment_sha256": base.digest(output / "deadline_amendment.json"),
            "source_checkpoint_sha256": protocol["source_checkpoint_sha256"], "source_code_hashes": protocol["source_code_hashes"],
            "selector_sha256": base.digest(Path(__file__)), "budget_sha256": base.digest(output / "budget.json"),
            "operational_sources_sha256": operational_hash,
            "path_base": "selection_json_directory", "baseline_rechecks": baselines,
            "arm_completion_receipts": {arm: verified[arm]["completion"] for arm in base.ARMS},
            "all_candidates": candidates, "ranking": ranking, "winner": ranking[0],
            "excluded_candidate_epochs": [8], "excluded_completion_evidence": [row for arm in base.ARMS for row in verified[arm]["excluded"]],
            "guardrail": amendment["selection"],
            "mean_kid_improvement": source_row["value"] - ranking[0]["value"],
            "test_evaluated": False, "class_evaluated": False, "published": False}
        if write:
            base.write_json(output / "selection_deadline.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.study), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
