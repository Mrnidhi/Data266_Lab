"""Render current Part 2 AI error drafts from verified, already-authored CSVs.

No annotations are invented, no models run, and no human-review fields change.
All three current AI draft CSVs must exist and match their current packets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_part2 import verify_ai_error_draft

MODELS = ("maxpool_mlp", "bilstm", "dilated_cnn")
GROUPS = {"confident_false_positive": 5, "confident_false_negative": 5,
          "near_threshold": 5, "long_review_slice": 5}
GROUP_LABELS = {"confident_false_positive": "Confident false positives",
                "confident_false_negative": "Confident false negatives",
                "near_threshold": "Near-threshold errors", "long_review_slice": "Long-review slice errors"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def recorded_review(value):
    normalized = str(value).strip().lower()
    require(normalized in ("true", "1", "false", "0", ""), "Unknown student_reviewed value")
    return normalized in ("true", "1")


def local_path(root, value):
    require(not Path(value).is_absolute(), "Published paths must be repository-relative")
    path = (root / value).resolve()
    require(path.is_relative_to(root), "Published path escapes repository")
    return path


def cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def quote_block(quote):
    longest = max((len(m.group()) for m in re.finditer(r"`+", quote)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + "text", quote, fence, ""]


def inspect_inputs(root):
    root = Path(root).resolve()
    member = root / "task2_sentiment/srinidhi"
    full = member / "outputs/full"
    manifest_path = member / "checkpoints/manifest.json"
    manifest = read_json(manifest_path)
    summary = read_json(full / "summary.json")
    selection = read_json(full / "selection_manifest.json")
    publication = read_json(full / "publication_metadata.json")
    provenance = read_json(full / "run_summary.json")
    sources = read_json(full / "training_sources.json")
    audit = read_json(full / "data_audit.json")
    paired = read_json(full / "paired_mcnemar.json")
    notebook = member / "src/sentiment.ipynb"
    require(summary.get("mode") == "full" and summary.get("synthetic") is False
            and set(summary.get("models", {})) == set(MODELS), "Need a completed real three-model publication")
    require(set(selection.get("selected", {})) == set(MODELS)
            and provenance.get("status") == "completed" and provenance.get("summary") == summary,
            "Published selection/provenance/summary disagree")
    selection_sha = digest(full / "selection_manifest.json")
    require(summary.get("selection_manifest_sha256") == selection_sha
            and provenance.get("selection_manifest_sha256") == selection_sha,
            "Selection hash differs from evaluated publication")
    source_run = publication.get("source_run")
    require(isinstance(source_run, str) and source_run.strip(), "Missing publication source run")
    local_path(root, source_run)
    require(paired == summary.get("paired_mcnemar"), "Published paired tests disagree with summary")
    protected = [manifest_path, notebook, *(full / name for name in (
        "summary.json", "selection_manifest.json", "publication_metadata.json", "run_summary.json",
        "training_sources.json", "data_audit.json", "paired_mcnemar.json"))]
    for item in publication.get("files", []):
        path = local_path(root, item["published_path"])
        require(digest(path) == item["published_sha256"], "Derived publication metadata hash changed")
        protected.append(path)
    model_data, review_total = {}, 0
    for name in MODELS:
        folder = full / name
        paths = {key: folder / filename for key,filename in (
            ("packet", "required_20_errors_for_review.csv"), ("draft", "ai_error_review_draft.csv"),
            ("metrics", "metrics.json"), ("history", "history.json"),
            ("config", "training_config.json"), ("predictions", "test_predictions.csv"))}
        require(paths["draft"].is_file(), f"Current authored AI draft is missing: {name}")
        best = manifest[name]["best.pt"]
        checkpoint = local_path(root, best["path"])
        require(digest(checkpoint) == best["sha256"], f"Actual checkpoint hash differs: {name}")
        if "bytes" in best:
            require(checkpoint.stat().st_size == best["bytes"], f"Checkpoint size differs: {name}")
        verified = verify_ai_error_draft(paths["packet"], paths["draft"], name, best["sha256"])
        require(verified is not None, f"AI linkage verification missing: {name}")
        packet, drafts = read_csv(paths["packet"]), read_csv(paths["draft"])
        require(Counter(r["review_group"] for r in packet) == GROUPS, f"Required review groups differ: {name}")
        metrics, history, config = (read_json(paths[key]) for key in ("metrics", "history", "config"))
        chosen = selection["selected"][name]
        require(metrics == summary["models"][name] and metrics["count"] == audit["split_counts"]["test"],
                f"Actual model metrics/count disagree: {name}")
        require(history and [r["epoch"] for r in history] == list(range(1, len(history) + 1)),
                f"Incomplete epoch history: {name}")
        epoch = metrics["selected_epoch"]
        require(type(epoch) is int and 1 <= epoch <= len(history)
                and chosen["selected_epoch"] == epoch and chosen["sha256"] == best["sha256"]
                and chosen["config"] == config
                and chosen["validation_macro_f1"] == metrics["selected_validation_macro_f1"]
                and history[epoch - 1]["validation_macro_f1"] == chosen["validation_macro_f1"],
                f"Checkpoint/epoch/config/validation selection differs: {name}")
        require(sources[name]["source_run"] == chosen["source_run"], f"Training source differs: {name}")
        predictions = read_csv(paths["predictions"])
        by_id = {r["example_id"]: r for r in predictions}
        require(len(predictions) == len(by_id) == metrics["count"], f"Prediction ID/count mismatch: {name}")
        matrix = [[0, 0], [0, 0]]
        for row in predictions:
            y, pred, p = int(row["true_label"]), int(row["predicted_label"]), float(row["positive_probability"])
            require(y in (0, 1) and pred in (0, 1) and math.isfinite(p) and 0 <= p <= 1
                    and pred == int(p >= metrics["threshold"]), f"Malformed prediction: {name}")
            matrix[y][pred] += 1
        require(matrix == metrics["confusion_matrix"], f"Predictions disagree with confusion matrix: {name}")
        require(math.isclose((matrix[0][0] + matrix[1][1]) / len(predictions), metrics["accuracy"], abs_tol=1e-12),
                f"Prediction accuracy differs: {name}")
        for row in packet:
            measured = by_id.get(row["example_id"])
            require(measured is not None and row["true_label"] == measured["true_label"]
                    and row["predicted_label"] == measured["predicted_label"]
                    and row["true_label"] != row["predicted_label"]
                    and math.isclose(float(row["positive_probability"]), float(measured["positive_probability"]),
                                     rel_tol=1e-12, abs_tol=0), f"Packet is not the actual selected model error: {name}")
        reviewed = sum(recorded_review(r.get("student_reviewed", "")) for r in packet)
        complete = sum(recorded_review(r.get("student_reviewed", "")) and bool(r.get("error_type", "").strip())
                       and bool(r.get("testable_fix", "").strip()) for r in packet)
        review_total += reviewed
        by_draft = {r["example_id"]: r for r in drafts}
        model_data[name] = {"paths": paths, "packet": packet, "drafts": [by_draft[r["example_id"]] for r in packet],
                            "metrics": metrics, "history": history, "config": config, "checkpoint_sha256": best["sha256"],
                            "reviewed": reviewed, "complete_review_fields": complete,
                            "hardware": sources[name]["hardware"], "source_run": chosen["source_run"],
                            "linkage": verified, "errors": matrix[0][1] + matrix[1][0]}
        protected.extend([checkpoint, *paths.values()])
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in sorted(set(protected))}
    return {"root": root, "member": member, "models": model_data, "publication": publication,
            "selection": selection, "selection_sha256": selection_sha, "audit": audit, "paired": paired,
            "notebook": notebook, "protected": hashes, "reviewed": review_total}


def document(evidence):
    models = evidence["models"]
    entries = sum(len(data["drafts"]) for data in models.values())
    complete = sum(data["complete_review_fields"] for data in models.values())
    lines = ["# Part 2 - AI-assisted error-analysis draft", "",
             f"This document renders {entries} already-authored AI-assisted hypotheses: twenty actual test errors "
             "for each required model. Exact case IDs, quotes, text hashes, packet hashes and checkpoint identities "
             "were checked before rendering. It does not certify independent student analysis or a causal explanation.", "",
             "Each packet contains five confident false positives, five confident false negatives, five near-threshold "
             "errors and five prespecified long-review slice errors. IDs are distinct within a model; texts can overlap "
             "across models. These selected cases cannot estimate the prevalence of error types.", "",
             f"Current evaluation publication: `{evidence['publication']['source_run']}`. "
             f"Frozen selection SHA-256: `{evidence['selection_sha256']}`.", "",
             "Selection rule recorded by the completed evaluation:", "",
             *quote_block(evidence["selection"]["selection_rule"]),
             "Historical official-test scores and texts have already been observed; this repeated evaluation is not "
             "a newly sealed holdout. Checkpoint selection uses validation. The proposals below are future training/validation "
             "studies with appropriate new evaluation, not permission to tune on these test errors or alter test labels.", "",
             f"Observed packet metadata: **{evidence['reviewed']} / {entries} `student_reviewed` flags are set**, "
             f"and {complete} flagged rows also have both required human interpretation fields filled. "
             "The renderer preserves those fields and the notebook byte-for-byte. Recorded flags are not independent "
             "verification that a human performed the review.", "",
             "## Current measured comparison", "",
             "| Model | Completed / selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Brier | ECE (15 bins) | Errors / test rows |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,data in models.items():
        m = data["metrics"]
        lines.append(f"| {name} | {len(data['history'])} / {m['selected_epoch']} | {m['selected_validation_macro_f1']:.6f} | "
                     f"{m['accuracy'] * 100:.4f}% | {m['macro']['f1']:.6f} | {m['brier']:.6f} | {m['ece_15']:.6f} | "
                     f"{data['errors']:,} / {m['count']:,} |")
    lines += ["", "These are the actual selected checkpoints' measurements. The table does not establish a global optimum, "
              "multi-seed ranking or production readiness. Bootstrap intervals in the complete metric files concern "
              "test-row sampling rather than training-seed uncertainty.", "",
              "| Experimental model vs MLP | Accuracy difference (percentage points) | Exact paired McNemar p-value |",
              "|---|---:|---:|"]
    base_accuracy = models["maxpool_mlp"]["metrics"]["accuracy"]
    for name in MODELS[1:]:
        lines.append(f"| {name} | {(models[name]['metrics']['accuracy'] - base_accuracy) * 100:+.4f} | "
                     f"{evidence['paired'][name]['pvalue_exact_two_sided']:.6g} |")
    overlap = evidence["audit"].get("cross_split_shared_text_hashes", {}).get("train__test")
    lines += ["", "The two stored baseline comparisons are unadjusted paired tests on the same examples; they do not "
              "compare the two experimental models directly or prove an architectural cause.", "",
              f"The duplicate audit records {overlap if overlap is not None else 'an unreported number of'} train/test shared "
              "raw-text hashes. Preserve and disclose the official-row protocol rather than treating it as a duplicate-clean benchmark.", "",
              "| Model | Prefix token cap | Recorded training CPU | Recorded training GPU |",
              "|---|---:|---|---|"]
    for name,data in models.items():
        lines.append(f"| {name} | {data['config']['max_length']} | {cell(data['hardware']['cpu'])} | {cell(data['hardware']['gpu'])} |")
    lines += ["", "Predicted probabilities and review text alone do not identify causal mechanisms. Mixed sentiment, "
              "target attribution, temporal changes, irony, truncation and label/text mismatch are hypotheses to investigate. "
              "The prefix cap can omit context, but a long review alone does not establish why a prediction failed.", ""]
    for name,data in models.items():
        lines += [f"## {name}", "", f"Training source: `{data['source_run']}`. "
                  f"Selected checkpoint SHA-256: `{data['checkpoint_sha256']}`.", "",
                  f"Recorded reviewed flags: {data['reviewed']} / {len(data['packet'])}. "
                  "Every explanation below remains labeled as an AI-assisted draft.", ""]
        group = None
        for number,row in enumerate(data["drafts"], 1):
            if row["review_group"] != group:
                group = row["review_group"]
                lines += [f"### {GROUP_LABELS[group]}", ""]
            lines += [f"#### {number}. {cell(row['example_id'])} - {cell(row['ai_draft_error_type'])}", "",
                      f"Official label {row['true_label']}; prediction {row['predicted_label']}; "
                      f"P(positive) {float(row['positive_probability']):.6f}; "
                      f"{row['original_token_length']} processed tokens before truncation.", "", "Exact evidence:", ""]
            for quote in json.loads(row["ai_draft_evidence_quotes"]):
                lines += quote_block(quote)
            lines += ["**AI-assisted hypothesis:** " + row["ai_draft_reason"], "",
                      "**Proposed future training/validation study:** " + row["ai_draft_testable_fix"], ""]
    lines += ["## Student and team review", "",
              "Review each full packet, retained input, official label and selected-model prediction. Verify or revise "
              "the hypotheses in personal words and propose a testable study. Set a human-reviewed flag only after "
              "that review occurs. This rendering does not add, remove or approve human annotations.", "",
              "Independent teammate evidence, the joint comparison/report, individual viva understanding and the "
              "remaining whole-lab obligations require their own completion evidence. The root AI_USE.md disclosure "
              "describes assistance; these drafts do not establish independent authorship.", ""]
    return "\n".join(lines)


def render(root=ROOT, *, check_only=False):
    evidence = inspect_inputs(root)
    root = evidence["root"]
    failure = evidence["member"] / "failure_analysis.md"
    body = document(evidence).encode("utf-8")
    notebook_hash = digest(evidence["notebook"])
    entries = sum(len(data["drafts"]) for data in evidence["models"].values())
    complete_fields = sum(data["complete_review_fields"] for data in evidence["models"].values())
    receipt = {"status": "verified_AI_assisted_not_student_review", "renderer": "scripts/render_sentiment_error_drafts.py",
               "renderer_sha256": digest(Path(__file__)), "total_entries": entries,
               "human_reviewed_rows": evidence["reviewed"], "human_review_complete": complete_fields == entries,
               "human_review_status_scope": "Observed packet flags and field completeness only; no independent human-review certification",
               "source_text_hash_definition": "SHA-256 of UTF-8 decoded human CSV text field, preserving literal display escapes",
               "evidence_definition": "Exact case-sensitive substrings of the full human CSV text field",
               "publication_source_run": evidence["publication"]["source_run"],
               "selection_manifest_sha256": evidence["selection_sha256"], "models": {},
               "protected_input_sha256": evidence["protected"],
               "failure_analysis_path": failure.relative_to(root).as_posix(),
               "failure_analysis_sha256": hashlib.sha256(body).hexdigest(),
               "failure_analysis_normalized_lf_sha256": hashlib.sha256(body).hexdigest(),
               "notebook_sha256_before": notebook_hash, "notebook_sha256_after": notebook_hash,
               "notebook_unchanged": True, "all_human_packets_unchanged": True}
    for name,data in evidence["models"].items():
        packet, draft = data["paths"]["packet"], data["paths"]["draft"]
        receipt["models"][name] = {
            "entries": len(data["drafts"]), "group_counts": dict(Counter(r["review_group"] for r in data["packet"])),
            "checkpoint_sha256": data["checkpoint_sha256"], "completed_epochs": len(data["history"]),
            "selected_epoch": data["metrics"]["selected_epoch"], "exact_quotes_verified": True,
            "all_ids_and_source_text_hashes_verified": True, "human_packet": packet.relative_to(root).as_posix(),
            "human_packet_sha256_before": digest(packet), "human_packet_sha256_after": digest(packet),
            "human_reviewed_rows": data["reviewed"], "human_review_fields_complete_rows": data["complete_review_fields"],
            "human_fields_blank": all(not r.get("error_type", "").strip() and not r.get("testable_fix", "").strip() for r in data["packet"]),
            "draft_path": draft.relative_to(root).as_posix(), "draft_sha256": digest(draft)}
    require(all(digest(root / path) == sha for path,sha in evidence["protected"].items()),
            "Inputs changed while rendering; no document was written")
    if not check_only:
        failure.write_bytes(body)
        require(all(digest(root / path) == sha for path,sha in evidence["protected"].items()),
                "Protected inputs changed; verification receipt not written")
        destination = root / "verification/part2_ai_error_drafts.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and rendering without writing any artifact")
    args = parser.parse_args()
    receipt = render(check_only=args.check_only)
    print(json.dumps({"entries": receipt["total_entries"], "observed_human_reviewed_rows": receipt["human_reviewed_rows"],
                      "human_packets_unchanged": receipt["all_human_packets_unchanged"],
                      "notebook_unchanged": receipt["notebook_unchanged"], "check_only": args.check_only}))


if __name__ == "__main__":
    main()
