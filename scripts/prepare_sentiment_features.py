"""Cache exact text features on CPU before renting a training GPU."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab1.sentiment import prepare_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "rehearsal", "full"], default="full")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "task2_sentiment/srinidhi/config.json").read_text())[args.mode]
    cfg["data_dir"] = str(args.data_dir)
    print(json.dumps(prepare_features(cfg, args.output), indent=2))
