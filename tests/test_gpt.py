import copy
import json
from pathlib import Path
import shutil

import pytest
from lab1.common import task_config_path
import torch

from lab1 import gpt


@pytest.fixture
def config():
    return json.loads(task_config_path("gpt").read_text())["smoke"]


def test_causal_attention_cannot_see_future(config):
    torch.manual_seed(2342)
    model = gpt.CharacterGPT(20, config).eval()
    original = torch.tensor([[4, 5, 6, 7, 8, 9]])
    modified = torch.tensor([[4, 5, 6, 12, 13, 14]])
    with torch.no_grad():
        a, b = model(original), model(modified)
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
    assert not torch.equal(a[:, 3:], b[:, 3:])


def test_shifted_targets_and_full_story_coverage():
    stories = ["abcdefg", "xy"]
    vocabulary = gpt.build_vocabulary(stories)
    dataset = gpt.StoryWindows(stories, vocabulary, context_length=3)
    mapping = {char: i for i, char in enumerate(vocabulary["tokens"])}
    seen = {0: [], 1: []}
    for index, (story_index, start) in enumerate(dataset.windows):
        x, y = dataset[index]
        valid = y != gpt.PAD
        sequence = [gpt.BOS] + [mapping[c] for c in stories[story_index]] + [gpt.EOS]
        size = int(valid.sum())
        assert x[valid].tolist() == sequence[start:start + size]
        assert y[valid].tolist() == sequence[start + 1:start + size + 1]
        seen[story_index].extend(y[valid].tolist())
    assert seen[0] == [mapping[c] for c in stories[0]] + [gpt.EOS]
    assert seen[1] == [mapping[c] for c in stories[1]] + [gpt.EOS]
    assert sum(len(v) for v in seen.values()) == dataset.target_count


def test_train_only_vocabulary_and_unknown_count():
    vocabulary = gpt.build_vocabulary(["abc"])
    dataset = gpt.StoryWindows(["abz"], vocabulary, 8)
    assert "z" not in vocabulary["tokens"]
    assert dataset.unknown_characters == 1
    assert dataset[0][1].tolist()[:4] == [4, 5, gpt.UNK, gpt.EOS]


def test_loader_epoch_visits_all_windows_once():
    stories = ["abcde", "uvwxyz", "hi"]
    dataset = gpt.StoryWindows(stories, gpt.build_vocabulary(stories), 2)
    loader = gpt._loader(dataset, 3, 2342, epoch=3, shuffle=True)
    targets = sum(int(y.ne(gpt.PAD).sum()) for _, y in loader)
    assert targets == dataset.target_count
    sampler_indices = [i for batch in loader.batch_sampler for i in batch]
    assert sorted(sampler_indices) == list(range(len(dataset)))


@pytest.mark.parametrize("interruption_step", [2, 6])
def test_checkpoint_reload_and_interrupted_resume_exact(config, tmp_path, monkeypatch, interruption_step):
    config = copy.deepcopy(config)
    config.update({"generation_prompts": 1, "generation_characters": 3, "cpu_threads": 1})
    baseline_dir = tmp_path / "baseline"
    saved_at_two = tmp_path / f"step{interruption_step}.pt"
    real_save = gpt._save_checkpoint

    def capture(path, checkpoint):
        real_save(path, checkpoint)
        if checkpoint["progress"]["global_step"] == interruption_step and not saved_at_two.exists():
            shutil.copy2(path, saved_at_two)

    monkeypatch.setattr(gpt, "_save_checkpoint", capture)
    baseline = gpt.run(config, baseline_dir, "cpu")
    resumed = gpt.run(config, tmp_path / "resumed", "cpu", saved_at_two)
    a = gpt.load_checkpoint(baseline_dir / "checkpoints" / "last.pt")
    b = gpt.load_checkpoint(tmp_path / "resumed" / "checkpoints" / "last.pt")
    assert baseline["inference_reload_verified"] and resumed["inference_reload_verified"]
    assert baseline["is_final_training_run"] is False
    generated = json.loads((baseline_dir / "generations.json").read_text())
    assert all(g["generation_seconds"] > 0 for g in generated)
    assert all(g["generated_character_tokens"] == len(g["continuation"]) for g in generated)
    assert baseline["generation_tokens_per_second"] == pytest.approx(
        sum(len(g["continuation"]) for g in generated) / sum(g["generation_seconds"] for g in generated))
    assert a["progress"]["global_step"] == b["progress"]["global_step"] == config["max_steps"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["scheduler"] == b["scheduler"]


def test_full_mode_rejects_reduced_data(config, tmp_path):
    config["mode"] = "full"
    with pytest.raises(ValueError, match="Full mode requires"):
        gpt.run(config, tmp_path, "cpu")


def test_diversity_handles_short_text_without_fake_scores():
    scores = gpt.diversity(["one"])
    assert scores["distinct_1"] == 1.0
    assert scores["distinct_2"] is None
    assert scores["repeated_4gram_fraction"] is None


def test_offline_cache_is_hash_verified_without_download(config, tmp_path):
    config.update({"dataset": "roneneldan/TinyStories", "data_cache": str(tmp_path),
                   "selection": "streaming_buffer_shuffle", "offline": True,
                   "train_stories": 1, "validation_stories": 1})
    paths = {key: Path(value) for key, value in gpt.data_cache_paths(config).items()}
    records = {"train": [{"index": 3, "text": "A cat."}],
               "validation": [{"index": 4, "text": "A dog."}]}
    manifest = {"dataset": config["dataset"]}
    for split in ["train", "validation"]:
        manifest[split] = [{"index": r["index"], "sha256": gpt._text_hash(r["text"])} for r in records[split]]
        paths[split].parent.mkdir(parents=True, exist_ok=True)
        paths[split].write_text("\n".join(json.dumps(r) for r in records[split]) + "\n")
    manifest["manifest_sha256"] = gpt._digest(manifest)
    paths["manifest"].write_text(json.dumps(manifest))
    train, valid, actual = gpt.prepare_data(config)
    assert (train, valid) == (["A cat."], ["A dog."])
    assert actual == manifest
    paths["train"].write_text(json.dumps({"index": 3, "text": "Changed."}) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        gpt.prepare_data(config)


def test_cpu_checkpoint_can_resume_with_a_new_fp16_scaler():
    class EnabledScaler:
        def is_enabled(self):
            return True

        def load_state_dict(self, state):
            if not state:
                raise RuntimeError("Actual enabled GradScaler rejects empty states")
            self.loaded = state

    scaler = EnabledScaler()
    assert gpt._restore_scaler(scaler, {}) == "fresh_scaler_no_saved_fp16_state"
    assert gpt._restore_scaler(scaler, {"scale": 65536}) == "restored"
    assert scaler.loaded == {"scale": 65536}


def test_resume_allows_machine_local_settings_but_rejects_training_changes(config, tmp_path):
    config.update(generation_prompts=1, generation_characters=2)
    gpt.run(config, tmp_path / "source", "cpu")
    relocated = dict(config, data_cache=str(tmp_path / "new_cache"), raw_data_cache=str(tmp_path / "raw"),
                     offline=False, cpu_threads=1, checkpoint_every_steps=1, log_every_steps=2)
    checkpoint = tmp_path / "source" / "checkpoints" / "last.pt"
    summary = gpt.run(relocated, tmp_path / "destination", "cpu", checkpoint)
    assert summary["global_steps"] == config["max_steps"]
    assert summary["inference_reload_verified"]
    with pytest.raises(ValueError, match="contract"):
        gpt.run(dict(relocated, learning_rate=0.1), tmp_path / "bad", "cpu", checkpoint)
