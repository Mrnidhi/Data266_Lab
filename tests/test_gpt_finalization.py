"""Evidence gates reject altered results; archives preserve portable Part 1 data."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import zipfile

import nbformat
import pytest
import torch

from lab1 import gpt

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import publish_gpt_results as publisher
import finalize_gpt as finalizer


def write(path, value):
    publisher.write_json(path, value)


@pytest.fixture(scope="module")
def full_evidence(tmp_path_factory):
    """A tiny state with complete synthetic metadata, never a submitted result."""
    root = tmp_path_factory.mktemp("gpt-evidence-fixture")
    run = root / "reproducibility/raw_logs/srinidhi/unit-fixture/part1-full"
    run.mkdir(parents=True)
    cfg = {"mode": "full", "dataset": "roneneldan/TinyStories", "epochs": 10, "max_steps": None,
           "train_stories": 100000, "validation_stories": 10000, "batch_size": 100000,
           "context_length": 8, "embedding_dim": 8, "heads": 2, "feedforward_dim": 16,
           "layers": 1, "dropout": 0.0, "generation_prompts": 1}
    vocabulary = gpt.build_vocabulary(["abc"])
    # Cache is tested separately with real per-text checks; this fixture isolates
    # result/metadata gates without encoding or training 110K stories.
    manifest = {"dataset": cfg["dataset"], "train": [{"index": i} for i in range(100000)],
                "validation": [{"index": i} for i in range(10000)]}
    manifest["manifest_sha256"] = gpt._digest(manifest)
    history = []
    for epoch in range(1, 11):
        train = gpt._metric((2 - epoch / 20) * 100000, 100000, 60000)
        valid = gpt._metric((2.1 - epoch / 20) * 10000, 10000, 6500)
        history.append({"kind": "epoch", "mode": "full", "epoch": epoch, "epoch_complete": True,
                        "epoch_fraction": 1.0, "step": epoch, "train": train, "validation": valid,
                        "generalization_gap_cross_entropy": valid["cross_entropy"] - train["cross_entropy"],
                        "mean_gradient_norm_before_clipping": 0.5})
    generations = []
    for mode in ("greedy", "sampled"):
        continuation = "A bird sang. The dog smiled."
        generations.append({"prompt": "Once: ", "continuation": continuation,
                            "full_text": "Once: " + continuation, "decoding": mode,
                            "generation_seconds": 1.0, "generated_character_tokens": len(continuation),
                            "generation_tokens_per_second": len(continuation), "metrics": gpt.diversity([continuation])})
    model = gpt.CharacterGPT(len(vocabulary["tokens"]), cfg)
    summary = {"mode": "full", "is_final_training_run": True, "inference_reload_verified": True,
               "completed_full_epochs": 10, "train_stories": 100000, "validation_stories": 10000,
               "train_windows_per_epoch": 100000, "train_targets_per_epoch": 100000,
               "validation_windows": 10000, "validation_unknown_characters": 0, "validation_character_count": 0,
               "global_steps": 10, "planned_steps": 10, "best_checkpoint_epoch": 10,
               "best_validation": history[-1]["validation"], "best_validation_cross_entropy": 1.6,
               "last_epoch_metrics": history[-1], "manifest_sha256": manifest["manifest_sha256"],
               "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
               "generation_metrics": {mode: gpt.diversity([generations[0]["continuation"]]) for mode in ("greedy", "sampled")},
               "generation_tokens_per_second": len(generations[0]["continuation"]),
               "total_training_seconds": 100.0, "training_targets_per_second": 10000.0,
               "elapsed_seconds_including_prior_sessions": 120.0,
               "cuda_peak_allocated_mb": 0.0, "host_peak_rss_mb": 10.0}
    for name, value in (("config.json", cfg), ("summary.json", summary), ("history.json", history),
                        ("data_manifest.json", manifest), ("vocabulary.json", vocabulary), ("generations.json", generations)):
        write(run / name, value)
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    (run / "RUN_LOG.txt").write_text('{"event": "completed"}\n', encoding="utf-8")
    state = {"task": "character_gpt", "config": cfg, "vocabulary": vocabulary,
             "data_manifest": manifest, "model": model.state_dict(), "progress": {
                 "history": history, "epoch": 10, "global_step": 10, "evaluation_pending": False,
                 "next_batch": 0, "best_validation_loss": 1.6}}
    (run / "checkpoints").mkdir()
    for name in ("best.pt", "last.pt"):
        torch.save(state, run / "checkpoints" / name)
    hashes = {}
    for name in ("src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                 "task1_llm/srinidhi/src/gpt.py", "task1_llm/srinidhi/config.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# Test source fixture\n", encoding="utf-8")
        hashes[name] = publisher.digest(path)
    write(run / "run_summary.json", {"status": "completed", "task": "gpt", "mode": "full", "config": cfg,
                                     "summary": summary, "environment": {"fixture": True}, "source_sha256": hashes})
    return root, run


def test_full_evidence_gate_accepts_consistent_complete_metadata(full_evidence):
    root, run = full_evidence
    evidence = publisher.verify_training_evidence(root, run, verify_cache=False)
    assert evidence["summary"]["completed_full_epochs"] == 10
    assert evidence["selected"]["epoch"] == 10


@pytest.mark.parametrize("mutation, expected", [
    ("partial_epoch", "every complete epoch"),
    ("altered_perplexity", "perplexity"),
    ("altered_raw_epoch", "Raw epoch events"),
    ("altered_generation", "Generation text/count"),
    ("altered_source", "source changed"),
])
def test_gate_rejects_altered_training_claims(full_evidence, tmp_path, mutation, expected):
    fixture_root, fixture_run = full_evidence
    root = tmp_path / "repository"
    shutil.copytree(fixture_root, root)
    run = root / fixture_run.relative_to(fixture_root)
    if mutation in ("partial_epoch", "altered_perplexity"):
        history = publisher.read_json(run / "history.json")
        if mutation == "partial_epoch":
            history[2]["epoch_complete"] = False
        else:
            history[2]["validation"]["perplexity"] = 123.0
        write(run / "history.json", history)
    elif mutation == "altered_raw_epoch":
        events = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        events[2]["train"]["cross_entropy"] += 0.1
        (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    elif mutation == "altered_generation":
        generations = publisher.read_json(run / "generations.json")
        generations[0]["continuation"] += " Invented."
        write(run / "generations.json", generations)
    elif mutation == "altered_source":
        (root / "task1_llm/srinidhi/src/gpt.py").write_text("# Changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        publisher.verify_training_evidence(root, run, verify_cache=False)


def test_frozen_cache_rejects_story_tampering_and_overlap(tmp_path):
    cfg = {"dataset": "roneneldan/TinyStories", "dataset_revision": "main", "seed": 2342,
           "train_stories": 1, "validation_stories": 1, "selection": "indexed_permutation",
           "shuffle_buffer": 1000, "context_length": 8,
           "data_cache": "task1_llm/srinidhi/data_processed/full"}
    cache = {name: tmp_path / value for name, value in gpt.data_cache_paths(cfg).items()}
    stories = {"train": "abc", "validation": "abd"}
    manifest = {split: [{"index": 1, "sha256": gpt._text_hash(text)}] for split, text in stories.items()}
    for split, text in stories.items():
        cache[split].parent.mkdir(parents=True, exist_ok=True)
        cache[split].write_text(json.dumps({"index": 1, "text": text}) + "\n", encoding="utf-8")
    write(cache["manifest"], manifest)
    vocabulary = gpt.build_vocabulary(["abc"])
    summary = {"train_windows_per_epoch": 1, "train_targets_per_epoch": 4, "validation_windows": 1,
               "validation_unknown_characters": 1, "validation_character_count": 3}
    _, measured = publisher.verify_cached_data(tmp_path, cfg, manifest, summary, vocabulary)
    assert measured["validation"]["unknown"] == 1
    cache["validation"].write_text(json.dumps({"index": 1, "text": "abz"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/index mismatch"):
        publisher.verify_cached_data(tmp_path, cfg, manifest, summary, vocabulary)
    manifest["validation"][0]["sha256"] = gpt._text_hash("abc")
    write(cache["manifest"], manifest)
    cache["validation"].write_text(json.dumps({"index": 1, "text": "abc"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate, or overlapping"):
        publisher.verify_cached_data(tmp_path, cfg, manifest, summary, vocabulary)


def test_failure_analysis_requires_three_actual_continuation_excerpts(tmp_path):
    generations = [{"continuation": "The bird sang. It was happy."}, {"continuation": "The cat cat cat."}]
    path = tmp_path / "failure_analysis.md"
    text = "# Observations\n\n" + "\n".join(
        f"## Case {index}\n\nGeneration ID: 1\n\n```text\nThe cat cat cat.\n```\n\n"
        "Failure type: Repetition\nObservation: The noun repeats.\n" for index in range(1, 4))
    path.write_text(text, encoding="utf-8")
    assert len(publisher.verify_failure_analysis(path, generations)) == 3
    path.write_text(text.replace("The cat cat cat.", "An invented sentence.", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="exact generated continuation"):
        publisher.verify_failure_analysis(path, generations)


def test_notebook_gate_rejects_empty_outputs_and_execution_errors(tmp_path):
    path = tmp_path / "gpt.ipynb"
    cell = nbformat.v4.new_code_cell("print('verified')", execution_count=1,
                                    outputs=[nbformat.v4.new_output("stream", name="stdout", text="verified\n")])
    notebook = nbformat.v4.new_notebook(cells=[cell], metadata={"lab1": {"mode": "completed_full_run_evidence"}})
    nbformat.write(notebook, path)
    assert finalizer.verify_notebook(path)["all_outputs_visible"]
    notebook.cells[0].outputs = []
    nbformat.write(notebook, path)
    with pytest.raises(ValueError, match="visible output"):
        finalizer.verify_notebook(path)
    notebook.cells[0].outputs = [nbformat.v4.new_output("error", ename="ValueError", evalue="broken", traceback=[])]
    nbformat.write(notebook, path)
    with pytest.raises(ValueError, match="execution errors"):
        finalizer.verify_notebook(path)


def test_part1_archive_runtime_inventory_and_hashes(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    run_relative = "reproducibility/raw_logs/srinidhi/unit-test/part1-full"
    run = root / run_relative
    required = ["PART1_FINALIZATION.md", "src/lab1/__init__.py", "src/lab1/common.py", "src/lab1/run.py",
                "task1_llm/srinidhi/src/gpt.py", "task1_llm/srinidhi/src/gpt.ipynb",
                "task1_llm/srinidhi/checkpoints/best.pt", "task1_llm/srinidhi/checkpoints/last.pt",
                "task1_llm/srinidhi/metrics_report.csv", "task1_llm/srinidhi/failure_analysis.md",
                "scripts/finalize_gpt.py", "scripts/publish_gpt_results.py",
                "task1_llm/srinidhi/data_processed/full/cache/train.jsonl",
                "task1_llm/srinidhi/data_processed/full/cache/validation.jsonl",
                "task1_llm/srinidhi/data_processed/full/cache/manifest.json",
                f"{run_relative}/metrics.jsonl", f"{run_relative}/RUN_LOG.txt",
                *[f"{run_relative}/{name}" for name in ("summary.json", "history.json", "run_summary.json", "config.json",
                                                       "resolved_config.json", "data_manifest.json", "vocabulary.json", "generations.json")]]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture\n")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('verified')", execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="verified\n")])],
        metadata={"lab1": {"mode": "completed_full_run_evidence"}})
    nbformat.write(notebook, root / "task1_llm/srinidhi/src/gpt.ipynb")
    evidence = {"run": run_relative, "summary": {"completed_full_epochs": 10, "best_checkpoint_epoch": 10},
                "cache_paths": {split: root / f"task1_llm/srinidhi/data_processed/full/cache/{split}.jsonl"
                                for split in ("train", "validation")}}
    archive = root / "dist/Part1.zip"
    before = finalizer.raw_inventory(run)
    result = finalizer.create_archive(root, evidence, archive)
    assert result["crc_and_content_hashes_verified"]
    assert before == finalizer.raw_inventory(run)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert all("\\" not in name for name in names)
        assert not any(name.endswith("RUN_LOG.txt") for name in names)
        assert b"torch==" in handle.read("Part 1/requirements.txt")
        assert b"torchvision" not in handle.read("Part 1/pyproject.toml")
        assert handle.read("Part 1/README.md") == b"test fixture\n"
        entries = {name: handle.read(name) for name in names}
    # Rebuild valid-CRC ZIP with altered bytes: SHA verification must still fail.
    entries["Part 1/task1_llm/srinidhi/checkpoints/best.pt"] = b"tampered"
    corrupt = root / "dist/tampered.zip"
    with zipfile.ZipFile(corrupt, "w") as handle:
        for name, value in entries.items():
            handle.writestr(name, value)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        finalizer.verify_archive(corrupt)


def test_archive_verifier_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Part 1/../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe or noncanonical"):
        finalizer.verify_archive(archive)
