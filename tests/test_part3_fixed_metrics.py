"""Check that the published diagnostic protocol matches its model and image evidence."""

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
MEMBER = ROOT / "task3_gan/srinidhi"
PROTOCOL = ROOT / "reproducibility/manifests/srinidhi/part3-metrics-20261004/protocol.json"


def test_metric_source_matches_frozen_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    source = ROOT / protocol["source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["source"]["sha256"]
    assert (
        protocol["checkpoint"]["sha256"]
        == "62d80f7ef2752b190fa496b7775b32dd77684bfceeb3b9f4c7356642848abc14"
    )
    assert protocol["device"] == "cpu"


def test_fixed_samples_match_seed_and_exported_source_mapping():
    protocol = json.loads(PROTOCOL.read_text())
    for direction, label in (("monet_to_photo", "pred_A2B"), ("photo_to_monet", "pred_B2A")):
        export = json.loads((MEMBER / f"outputs/{label}_export_manifest.json").read_text())
        mapping = {row["source"]: row for row in export["images"]}
        expected = sorted(mapping)
        if direction == "photo_to_monet":
            expected = sorted(random.Random(2342).sample(expected, 300))
        pairs = protocol["directions"][direction]["pairs"]
        assert [Path(row["input"]["path"]).name for row in pairs] == expected
        assert len(pairs) == 300
        for row in pairs:
            item = mapping[Path(row["input"]["path"]).name]
            assert row["prediction"]["path"] == f"task3_gan/srinidhi/outputs/{label}/{item['jpeg']}"
            assert row["prediction"]["sha256"] == item["jpeg_sha256"]


def test_target_references_are_the_declared_matching_real_subsets():
    directions = json.loads(PROTOCOL.read_text())["directions"]
    for first, second in (
        ("monet_to_photo", "photo_to_monet"),
        ("photo_to_monet", "monet_to_photo"),
    ):
        assert directions[first]["references"] == [r["input"] for r in directions[second]["pairs"]]
