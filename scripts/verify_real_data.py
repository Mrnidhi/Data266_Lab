"""Short CPU checks on actual public text; never final training or a GPU benchmark."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import task_config_path, utc_now, write_json
from lab1.run import execute
from lab1.sentiment import prefetch_data

stamp = utc_now().replace(":", "-")
output = ROOT / "runs" / ("real-data-cpu-check-" + stamp)
gpt = execute("gpt", "rehearsal", output / "gpt", "cpu", overrides=[
    'data_cache="data/rehearsal/tinystories"', "offline=true", "max_steps=2",
    "generation_prompts=1", "generation_characters=20"])
cfg = json.loads(task_config_path("sentiment").read_text())["rehearsal"]
cfg.update(train_limit=64, validation_limit=32, test_limit=64)
cache = ROOT / "data/local_checks/yelp"
prefetch_data(cfg, cache)
sentiment = execute("sentiment", "rehearsal", output / "sentiment", "cpu", overrides=[
    "data_dir=" + json.dumps(str(cache)), "train_limit=64", "validation_limit=32", "test_limit=64",
    "batch_size=16", "epochs=1", "bootstrap_samples=20"])
write_json(ROOT / "verification/real_data_pipeline.json", {
    "status": "passed", "device": "cpu", "final_lab_results": False,
    "description": "Full architecture dimensions; GPT two updates, three sentiment models one tiny-data epoch each. Vocabulary is fitted on the small training subsets, so vocabulary caps are not filled.",
    "runs": {"gpt": str((output / "gpt").relative_to(ROOT)), "sentiment": str((output / "sentiment").relative_to(ROOT))},
    "elapsed_seconds": {"gpt": gpt["elapsed_seconds"], "sentiment": sentiment["elapsed_seconds"]}})
print("Real-data CPU checks passed. These are not full training results.")
