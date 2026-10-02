"""CPU receipt fixtures for prospective deadline selection, not model training."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


deadline = module("deadline_selector", ROOT / "scripts/select_cyclegan_ema_deadline.py")
helpers = module("original_study_tests", ROOT / "tests/test_study_cyclegan_ema.py")
experiment = helpers.experiment


def freeze_json(path, value):
    deadline.base.write_json(path, value)
    path.with_suffix(".sha256").write_text(deadline.base.digest(path))


@pytest.fixture
def amended(experiment, monkeypatch):
    output, source, protocol = experiment
    monkeypatch.setattr(deadline.base, "ROOT", helpers.study.ROOT)
    copied = helpers.study.ROOT / "scripts/select_cyclegan_ema_deadline.py"
    copied.write_bytes((ROOT / "scripts/select_cyclegan_ema_deadline.py").read_bytes())
    runs = {arm: helpers.completed_arm(output, protocol, arm) for arm in helpers.study.ARMS}
    first = runs["lr5e5"]
    baseline_receipt = first / "validation/step_0000000/raw/receipt.json"
    now = datetime.now(timezone.utc)
    amendment = {"format_version": 1, "kind": "prospective_access_deadline_amendment", "created_utc": now.isoformat(),
        "original_protocol_sha256": deadline.base.digest(output / "protocol.json"),
        "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        "source_manifest_fingerprint": protocol["source_manifest_fingerprint"],
        "candidate_epochs": [4], "candidate_steps": [8], "excluded_candidate_epochs": [8], "max_candidates": 5,
        "arms": ["lr5e5", "lr2e5"], "variants": ["raw", "ema"],
        "selection": dict(protocol["selection"], max_candidates=5, require_both_arms_complete=False,
                          require_both_epoch_four_endpoints_complete=True),
        "required_arm_results": {"lr5e5": {"status": "completed", "completed_updates": 16, "validation_receipts": 5},
            "lr2e5": {"status": "partial", "stop_reason": "max_steps", "completed_updates": 8, "validation_receipts": 3}},
        "training_deadline_utc": (now+timedelta(hours=2)).isoformat(),
        "class_evaluation_deadline_utc": (now+timedelta(hours=3)).isoformat(),
        "finalization_deadline_utc": (now+timedelta(hours=4)).isoformat(),
        "lab_access_deadline_utc": (now+timedelta(hours=5)).isoformat(),
        "freeze_evidence": {"completed_validation_receipts": [{"path": deadline.base.relative_path(baseline_receipt, output),
            "sha256": deadline.base.digest(baseline_receipt), "step": 0, "variant": "raw",
            "metrics_sha256": deadline.base.digest(baseline_receipt.with_name("metrics.json"))}],
            "training_progress": {"lr5e5": {"step": 1, "epoch_fraction": .5, "nan_events": 0},
                                  "lr2e5": {"step": 0, "not_started": True}},
            "nonzero_validation_results_exist": False, "nonzero_candidate_checkpoints_exist": False},
        "no_expansion_after_results": True, "test_or_class_feedback_used_for_amendment": False,
        "original_eight_epoch_study_completion_claimed": False}
    freeze_json(output / "deadline_amendment.json", amendment)
    freeze_json(output / "deadline_operational_sources.json", {"scripts/select_cyclegan_ema_deadline.py": deadline.base.digest(copied)})
    for arm, run in runs.items():
        wrapper = deadline.base.read_json(run / "run_summary.json")
        summary = wrapper["summary"]
        wrapper.update(started_utc=(now-timedelta(hours=1) if arm == "lr5e5" else now+timedelta(seconds=1)).isoformat(),
                       ended_utc=(now+timedelta(minutes=1)).isoformat())
        summary["stop_reason"] = "completed" if arm == "lr5e5" else "max_steps"
        if arm == "lr2e5":
            wrapper["status"] = "partial"
            wrapper["config"]["max_steps"] = 8
            deadline.base.write_json(run / "resolved_config.json", wrapper["config"])
            summary.update(status="partial", completed_updates=8, completed_epoch_fraction=4, ema_updates=8)
            summary["validation_history"] = summary["validation_history"][:3]
            # Keep unused fixture bytes outside the audited validation tree.
            (run / "validation/step_0000016").rename(run / "unused_fixture_validation")
            lines = (run / "training_log.jsonl").read_text().splitlines()[:8]
            (run / "training_log.jsonl").write_text("\n".join(lines)+"\n")
            for variant in ("raw", "ema"):
                path = run / f"candidates/step_0000008/{variant}.pt"
                saved = deadline.base.load_checkpoint(path)
                saved["config"]["max_steps"] = 8
                helpers.torch.save(saved, path)
                directory = run / f"validation/step_0000008/{variant}"
                index = 1 if variant == "raw" else 2
                summary["validation_history"][index] = helpers.validation_evidence(directory, path, protocol, 8, variant, (.018, .018))
        rows = [json.loads(line) for line in (run / "training_log.jsonl").read_text().splitlines()]
        for row in rows:
            row["ema_updates"] = row["step"]
        (run / "training_log.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        deadline.base.write_json(run / "run_summary.json", wrapper)
        deadline.base.write_json(run / "provenance/final.json", wrapper)
        deadline.base.write_json(run / "trainer_summary.json", summary)
    return output, protocol, amendment, runs


def update_summary(run, transform):
    wrapper = deadline.base.read_json(run / "run_summary.json")
    transform(wrapper)
    deadline.base.write_json(run / "run_summary.json", wrapper)
    deadline.base.write_json(run / "provenance/final.json", wrapper)
    deadline.base.write_json(run / "trainer_summary.json", wrapper["summary"])


def set_scores(run, protocol, step, variant, values):
    directory = run / f"validation/step_{step:07d}/{variant}"
    receipt = helpers.validation_evidence(directory, run / f"candidates/step_{step:07d}/{variant}.pt", protocol, step, variant, values)
    def transform(wrapper):
        wrapper["summary"]["validation_history"] = [receipt if (row["step"], row["candidate_variant"]) == (step, variant) else row
                                                   for row in wrapper["summary"]["validation_history"]]
    update_summary(run, transform)


def test_selects_exactly_five_slots_and_unconditionally_ignores_better_epoch_eight(amended):
    output, protocol, _, runs = amended
    set_scores(runs["lr5e5"], protocol, 16, "ema", (-1, -1))
    result = deadline.select(output)
    assert result["status"] == "completed_amended_study" and not result["original_study_completed"]
    assert len(result["all_candidates"]) == 5 and all(row["epoch"] in (None, 4) for row in result["all_candidates"])
    assert result["winner"]["epoch"] == 4 and result["winner"]["value"] == .018
    assert len(result["excluded_completion_evidence"]) == 2
    assert all("value" not in row and "directions" not in row for row in result["excluded_completion_evidence"])


def test_keeps_guardrail_and_incumbent_exact_tie(amended):
    output, protocol, _, runs = amended
    for arm, run in runs.items():
        for variant in ("raw", "ema"):
            set_scores(run, protocol, 8, variant, (.001, .021) if arm == "lr5e5" else (.02, .02))
    result = deadline.select(output, write=False)
    assert result["winner"]["kind"] == "unchanged_source"
    assert all(not row["eligible"] for row in result["all_candidates"] if row["arm"] == "lr5e5")
    assert not (output / "selection_deadline.json").exists()


@pytest.mark.parametrize("failure", ["nonzero_at_freeze", "progress_endpoint", "expanded_slots", "guardrail_changed", "amendment_hash", "source_code", "source_pin"])
def test_bad_amendment_or_implementation_freeze_refused(amended, failure):
    output, _, amendment, _ = amended
    if failure == "nonzero_at_freeze": amendment["freeze_evidence"]["nonzero_validation_results_exist"] = True
    if failure == "progress_endpoint": amendment["freeze_evidence"]["training_progress"]["lr5e5"]["step"] = 8
    if failure == "expanded_slots": amendment["candidate_epochs"] = [4, 8]
    if failure == "guardrail_changed": amendment["selection"]["guardrail_tolerance"] = 1
    freeze_json(output / "deadline_amendment.json", amendment)
    if failure == "amendment_hash": (output / "deadline_amendment.json").write_text(json.dumps(amendment)+" ")
    if failure == "source_code": (deadline.base.ROOT / "scripts/select_cyclegan_ema_deadline.py").write_text("# changed")
    if failure == "source_pin": freeze_json(output / "deadline_operational_sources.json", {})
    with pytest.raises(ValueError): deadline.select(output)
    assert not (output / "selection_deadline.json").exists()


@pytest.mark.parametrize("failure", ["first_partial", "second_budget_stop", "second_pending", "second_extra_step", "missing_end", "stale_provenance", "history", "bad_log", "wrong_variant", "nonfinite_weights"])
def test_incomplete_or_inconsistent_arm_refused(amended, failure):
    output, protocol, _, runs = amended
    run = runs["lr2e5"]
    if failure == "first_partial": update_summary(runs["lr5e5"], lambda wrapper: wrapper.update(status="partial"))
    elif failure == "second_budget_stop": update_summary(run, lambda wrapper: wrapper["summary"].update(stop_reason="budget_exhausted"))
    elif failure == "second_pending": update_summary(run, lambda wrapper: wrapper["summary"].update(pending_validation={"step": 8}))
    elif failure == "second_extra_step": update_summary(run, lambda wrapper: wrapper["summary"].update(completed_updates=9))
    elif failure == "missing_end": update_summary(run, lambda wrapper: wrapper.pop("ended_utc"))
    elif failure == "history": update_summary(run, lambda wrapper: wrapper["summary"]["validation_history"].pop())
    elif failure == "stale_provenance": deadline.base.write_json(run / "provenance/final.json", {})
    elif failure == "bad_log":
        with (run / "training_log.jsonl").open("a") as stream: stream.write('{}\n')
    else:
        path = run / "candidates/step_0000008/ema.pt"
        saved = deadline.base.load_checkpoint(path)
        if failure == "wrong_variant": saved["state"]["candidate_variant"] = "raw"
        else: saved["models"]["G_photo_to_monet"]["weight"].fill_(float("nan"))
        helpers.torch.save(saved, path)
        set_scores(run, protocol, 8, "ema", (.018, .018))
    with pytest.raises(ValueError): deadline.select(output)
    assert not (output / "selection_deadline.json").exists()
