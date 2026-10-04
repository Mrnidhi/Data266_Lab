"""Schedule persistence checks that do not start a training process."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/part3_push_package/run_all.py"
if not RUNNER.exists():
    RUNNER = ROOT / "run_all.py"
spec = importlib.util.spec_from_file_location("part3_push_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def plan():
    return {
        "defaults": {"steps": "auto", "eval_every": "auto"},
        "arms": [{"name": "b1_c5"}, {"name": "b8_c5"}, {"name": "b8_c5_r1"}],
        "eval_rounds": 25,
        "eval_minutes_per_arm": 45,
        "min_steps": 8000,
        "max_steps": 600000,
    }


def test_full_and_smoke_schedules_have_separate_files(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(runner, "WORK", tmp_path)
    rates = {"b1_c5": 11.0, "b8_c5": 2.6, "b8_c5_r1": 2.4}
    resolved = runner.resolve_schedule(plan, rates, 10.0)
    full_path = tmp_path / "schedule.json"
    full_bytes = full_path.read_bytes()
    assert resolved["b1_c5"] == {"steps": 323000, "eval_every": 12750}
    assert resolved["b8_c5"] == {"steps": 76000, "eval_every": 3000}
    assert resolved["b8_c5_r1"] == {"steps": 70000, "eval_every": 2750}
    assert runner.load_schedule(full_path, plan) == resolved

    runner.resolve_schedule(plan, rates, 1.0, "schedule_smoke.json")
    assert full_path.read_bytes() == full_bytes
    assert (tmp_path / "schedule_smoke.json").is_file()
    assert not any((tmp_path / arm["name"]).exists() for arm in plan["arms"])


def test_missing_saved_schedule_cannot_be_recomputed_on_resume(tmp_path, plan):
    with pytest.raises(SystemExit, match="Restore the original verified schedule"):
        runner.load_schedule(tmp_path / "schedule.json", plan)


def test_saved_schedule_must_match_the_requested_experiments(tmp_path, plan):
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps({"arms": {"old_experiment": {"steps": 1000}}}))
    with pytest.raises(SystemExit, match="does not match"):
        runner.load_schedule(schedule, plan)


@pytest.mark.parametrize("invalid", ["auto", 0, -1, 12.5, True, None])
def test_saved_schedule_requires_concrete_positive_steps(tmp_path, plan, invalid):
    schedule = tmp_path / "schedule.json"
    arms = {arm["name"]: {"steps": invalid, "eval_every": 500} for arm in plan["arms"]}
    schedule.write_text(json.dumps({"arms": arms}))
    with pytest.raises(SystemExit, match="positive integer"):
        runner.load_schedule(schedule, plan)
