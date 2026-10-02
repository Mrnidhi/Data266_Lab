"""Derive portable invocation receipts without editing original training evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = "reproducibility/raw_logs/srinidhi/desktop-quality-20261001/part2-search"
DESTINATION = "task2_sentiment/srinidhi/outputs/quality_search/desktop-quality-20261001"
TRANSFORMATION = "Only command[0] replaced with portable python executable; all other recorded invocation values preserved."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def external_path(value):
    if isinstance(value, dict):
        return any(external_path(item) for item in value.values())
    if isinstance(value, list):
        return any(external_path(item) for item in value)
    return isinstance(value, str) and bool(re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/Users/", "/home/")))


def save_immutable(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Derived evidence already exists with different bytes: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def serialize(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def export(run, output, root=ROOT):
    root, run, output = root.resolve(), run.resolve(), output.resolve()
    if not run.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("Source and derived evidence must remain within the repository")
    plan_path, selection_path = run / "quality_plan.json", run / "selection_manifest.json"
    plan, selection = read_json(plan_path), read_json(selection_path)
    if selection.get("test_metrics_used_for_selection") is not False:
        raise ValueError("Expected a completed validation-only selection")
    if selection.get("quality_plan_sha256") != digest(plan_path):
        raise ValueError("Selection and original frozen plan disagree")
    expected = set(plan["recipes"])
    sources = []
    for path in sorted((run / "invocations").glob("*.json")):
        invocation = read_json(path)
        if invocation.get("returncode") != 0 or not invocation.get("ended_utc"):
            raise ValueError("Every original invocation must have completed successfully before this export")
        if invocation.get("candidate_id") not in expected:
            raise ValueError("Unexpected candidate invocation")
        command = invocation.get("command")
        if not isinstance(command, list) or len(command) < 2 or command[1] != "scripts/tune_sentiment.py":
            raise ValueError("Unexpected invocation command")
        sources.append((path, invocation, digest(path)))
    if len(sources) != len(expected) or {row[1]["candidate_id"] for row in sources} != expected:
        raise ValueError("Need exactly one successful original invocation for every planned candidate")
    receipts, pending = [], []
    for source, invocation, checksum in sources:
        relative = source.relative_to(root).as_posix()
        derived = dict(invocation)
        derived["command"] = ["python", *invocation["command"][1:]]
        derived["derived_evidence"] = {"source_receipt": relative, "source_sha256": checksum,
                                       "transformation": TRANSFORMATION, "original_receipt_preserved": True}
        if external_path(derived):
            raise ValueError("An external absolute path remains after the declared single-field transformation")
        published = output / "invocations" / source.name
        payload = serialize(derived)
        pending.append((published, payload))
        receipts.append({"candidate_id": invocation["candidate_id"], "source_receipt": relative,
                         "source_sha256": checksum, "published_receipt": published.relative_to(root).as_posix(),
                         "published_sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "quality_plan": {"path": plan_path.relative_to(root).as_posix(), "sha256": digest(plan_path)},
                "selection_manifest": {"path": selection_path.relative_to(root).as_posix(), "sha256": digest(selection_path)},
                "raw_receipts_included": False, "receipts": receipts}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        manifest["created_utc"] = existing["created_utc"]
        if existing != manifest:
            raise ValueError("Existing portable manifest disagrees with original source evidence")
    for source, _, checksum in sources:
        if digest(source) != checksum:
            raise ValueError("An original receipt changed during export")
    for path, payload in pending:
        save_immutable(path, payload)
    save_immutable(manifest_path, serialize(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(RUN))
    parser.add_argument("--output", type=Path, default=Path(DESTINATION))
    args = parser.parse_args()
    manifest = export(args.run_dir, args.output)
    print(json.dumps({"portable_receipts": len(manifest["receipts"]), "raw_originals_modified": False,
                      "manifest": (args.output / "manifest.json").as_posix()}, indent=2))


if __name__ == "__main__":
    main()
