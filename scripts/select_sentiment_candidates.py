"""Freeze the bounded Part B selection using validation scores only; never train or test."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import finalize_sentiment_selection as finalizer
from lab1 import sentiment as s
from lab1.common import utc_now
import torch


RECIPES = {
    "maxpool_mlp_long_schedule_control": "maxpool_mlp",
    "bilstm_long_schedule_control": "bilstm",
    "dilated_cnn_long_schedule_control": "dilated_cnn",
    "mlp_wider": "maxpool_mlp",
    "bilstm_wider": "bilstm",
    "cnn_long_context": "dilated_cnn",
}
MARGIN = Decimal("0.001")
VALIDATION_FIELDS = ("selected_epoch", "selected_validation_macro_f1", "parameter_count", "train_seconds")
RULE = (
    "For each model family find the highest validation macro-F1. Among candidates within "
    "0.001 inclusive of that maximum, prefer fewer parameters, then higher validation macro-F1. "
    "Break an exact remaining tie by source path and candidate ID. This is a practical "
    "preference, not a statistical significance test; test metrics never enter selection."
)


def score(candidate):
    return Decimal(str(candidate["validation_macro_f1"]))


def choose_family(candidates):
    """Apply the prespecified tolerance to the maximum, not successive pairwise ties."""
    if not candidates:
        raise ValueError("Cannot select from an empty family")
    highest = max(map(score, candidates))
    eligible = [row for row in candidates if highest - score(row) <= MARGIN]
    chosen = min(eligible, key=lambda row: (
        row["parameter_count"], -score(row), row["source_run"], row["candidate_id"]))
    return chosen, highest, eligible


def _invocations(root):
    """Read completion status for every expected invocation before opening checkpoints."""
    recipe_dir = root / "task2_sentiment/srinidhi/experiments"
    sources = [{"run": root / "runs/part-b-full", "names": s.MODEL_NAMES,
                "candidate_id": "reference", "recipe": None, "provenance_file": "run_summary.json"}]
    for recipe_name, model_name in RECIPES.items():
        path = recipe_dir / f"{recipe_name}.json"
        recipe = finalizer.read_json(path)
        if recipe.get("name") != recipe_name or recipe.get("model") != model_name:
            raise ValueError(f"Unexpected recipe identity: {recipe_name}")
        sources.append({"run": root / "runs/part-b-tuning" / recipe_name,
                        "names": (model_name,), "candidate_id": recipe_name,
                        "recipe": recipe, "recipe_path": path, "provenance_file": "provenance.json"})
    incomplete = []
    for source in sources:
        path = source["run"] / source["provenance_file"]
        if not path.is_file():
            incomplete.append(f"{source['run'].relative_to(root)}: missing provenance")
            continue
        source["provenance"] = finalizer.read_json(path)
        if source["provenance"].get("status") != "completed":
            incomplete.append(f"{source['run'].relative_to(root)}: {source['provenance'].get('status', 'unknown')}")
    if incomplete:
        raise ValueError("All seven invocations must be completed before selection: " + "; ".join(incomplete))
    return sources


def _candidate(root, base, source, name):
    run, recipe = source["run"], source["recipe"]
    metadata_name = "metrics.json" if recipe is None else "validation_selection.json"
    # Reference metrics can contain test fields. Extract only this explicit whitelist;
    # the original provenance is preserved opaquely for the required integrity check.
    raw_metadata = finalizer.read_json(run / name / metadata_name)
    metadata = {key: raw_metadata[key] for key in VALIDATION_FIELDS}
    del raw_metadata
    f1, count, seconds = (metadata["selected_validation_macro_f1"],
                          metadata["parameter_count"], metadata["train_seconds"])
    if (isinstance(f1, bool) or not isinstance(f1, (int, float)) or not math.isfinite(f1) or not 0 <= f1 <= 1
            or isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError(f"Invalid validation/training metadata: {name} in {run}")
    checkpoint = run / name / "checkpoints/best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if recipe is not None:
        if (finalizer.read_json(run / "candidate.json") != recipe
                or source["provenance"].get("candidate") != recipe):
            raise ValueError(f"Run does not match its explicit experiment recipe: {source['candidate_id']}")
        expected = {**base, **recipe["overrides"], "validation_only": True}
        if s._resume_contract(cfg) != s._resume_contract(expected):
            raise ValueError(f"Checkpoint configuration does not match recipe: {source['candidate_id']}")
    elif cfg.get("validation_only"):
        raise ValueError("Reference suite unexpectedly marked validation-only")
    entry = {"candidate_id": f"reference:{name}" if recipe is None else source["candidate_id"],
             "model_name": name, "source_run": run.relative_to(root).as_posix(),
             "checkpoint": checkpoint.relative_to(root).as_posix(), "sha256": finalizer.digest(checkpoint),
             "config": cfg, "validation_macro_f1": f1, "selected_epoch": metadata["selected_epoch"],
             "parameter_count": count, "train_seconds": seconds, "provenance": source["provenance"],
             "selection_metadata_file": (run / name / metadata_name).relative_to(root).as_posix(),
             "selection_fields_used": list(VALIDATION_FIELDS), "recipe": recipe}
    if recipe is not None:
        entry.update(recipe_path=source["recipe_path"].relative_to(root).as_posix(),
                     recipe_sha256=finalizer.digest(source["recipe_path"]))
    # The verifier checks opaque provenance against original artifacts, never ranks
    # test scores. Run it for every candidate, including candidates that will lose.
    verified = finalizer.verify_source(root, name, entry, base)
    model = s.build_model(name, len(state["vocabulary"]), cfg)
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    del model, state
    if actual_count != count:
        raise ValueError(f"Parameter count disagrees with checkpoint configuration: {entry['candidate_id']}")
    entry["source_metadata_sha256"] = verified["metadata_sha256"]
    return entry


def select_candidates(output, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Frozen selection destination already exists; choose a fresh file")
    base = finalizer.read_json(root / "task2_sentiment/srinidhi/config.json")["full"]
    sources = _invocations(root)
    candidates = [_candidate(root, base, source, name) for source in sources for name in source["names"]]
    selected, rationale, confirmations = {}, {}, []
    for name in s.MODEL_NAMES:
        family = [row for row in candidates if row["model_name"] == name]
        chosen, highest, eligible = choose_family(family)
        eligible_ids = {row["candidate_id"] for row in eligible}
        for row in family:
            gap = highest - score(row)
            reason = ("Selected by parameter count, then validation score within the practical band."
                      if row is chosen else "Outside the 0.001 band from the highest validation score."
                      if row["candidate_id"] not in eligible_ids else
                      "Within the practical band; another candidate wins the size/validation/stable-path ordering.")
            row["decision"] = {"selected": row is chosen, "within_practical_band": gap <= MARGIN,
                               "gap_to_highest_validation_macro_f1": float(gap), "reason": reason}
        selected[name] = chosen
        rationale[name] = {"highest_validation_macro_f1": float(highest),
                           "eligible_candidate_ids": sorted(eligible_ids),
                           "selected_candidate_id": chosen["candidate_id"],
                           "selected_parameter_count": chosen["parameter_count"],
                           "selected_validation_macro_f1": chosen["validation_macro_f1"],
                           "practical_margin": float(MARGIN), "statistical_significance_claimed": False}
        control = next(row for row in family if row["candidate_id"] == f"{name}_long_schedule_control")
        gain = score(chosen) - score(control)
        if chosen["parameter_count"] > control["parameter_count"] and MARGIN < gain <= 2 * MARGIN:
            confirmations.append({"model_name": name, "selected_candidate_id": chosen["candidate_id"],
                "gain_over_same_schedule_control": float(gain), "optional": True, "automatic_run_started": False,
                "reason": "Architectural gain lies just above the practical margin (0.001 to 0.002 inclusive at the upper bound). A further initialization on the same frozen split can assess stability; this is not a significance claim."})
    # Recheck every source and recipe after collection, before publishing the frozen file.
    for row in candidates:
        checksums = dict(row["source_metadata_sha256"])
        if row["recipe"] is not None:
            checksums[row["recipe_path"]] = row["recipe_sha256"]
        for relative, checksum in checksums.items():
            if finalizer.digest(root / relative) != checksum:
                raise ValueError(f"Selection source changed during verification: {relative}")
    manifest = {"format_version": 1, "created_utc": utc_now(), "selection_rule": RULE,
                "practical_margin": float(MARGIN), "required_completed_invocations": 7,
                "candidate_count": len(candidates), "candidates": candidates, "selected": selected,
                "rationale": rationale, "optional_seed_confirmations": confirmations,
                "test_metrics_used_for_selection": False, "automatic_training_started": False,
                "reference_test_policy": "Original reference provenance is preserved unchanged; reference test metrics were not used for ranking or selection."}
    serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(serialized)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh frozen selection JSON file")
    args = parser.parse_args()
    manifest = select_candidates(args.output)
    print(json.dumps({"output": str(args.output), "candidate_count": manifest["candidate_count"],
                      "selected": {name: {key: row[key] for key in ("candidate_id", "validation_macro_f1", "parameter_count")}
                                   for name, row in manifest["selected"].items()},
                      "optional_seed_confirmations": manifest["optional_seed_confirmations"]}, indent=2))


if __name__ == "__main__":
    main()
