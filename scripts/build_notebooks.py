"""Build three portable notebook entry points. Defaults never launch full training."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("gpt", "task1_llm", "Character GPT"),
         ("sentiment", "task2_sentiment", "Yelp sentiment classification"),
         ("cyclegan", "task3_gan", "CycleGAN style transfer")]

for task, folder, title in TASKS:
    cells = [nb.v4.new_markdown_cell(f"# {title}\n\nSrinidhi, SID4 2342. This notebook defaults to **CPU smoke verification**. "
        "Its small synthetic outputs are not final lab results. Review the source and configuration before full training. "
        "Every run writes raw logs, provenance, metrics, outputs, and checkpoints to a new folder."),
        nb.v4.new_code_cell('''SID4 = 2342
SEED = SID4
SLICE = SID4 % 1000
HP_ID = SID4 % 6
CLS_A = SID4 % 10
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10
print(dict(SID4=SID4, SEED=SEED, SLICE=SLICE, HP_ID=HP_ID, CLS_A=CLS_A, CLS_B=CLS_B))'''),
        nb.v4.new_code_cell('''from pathlib import Path
import os
import sys
roots = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in roots if (p / "src/lab1").is_dir() and (p / "pyproject.toml").is_file()), None)
if ROOT is None:
    raise RuntimeError("Extract the complete package and start this notebook inside its folder.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from lab1.common import environment, utc_now
from lab1.run import execute
print(environment())'''),
        nb.v4.new_code_cell(f'''TASK = "{task}"
MODE = "smoke"  # "rehearsal" is short; "full" selects the complete training configuration.
DEVICE = "cpu"  # Change to "cuda" on RunPod or the college NVIDIA machine.
OVERRIDES = []  # See task README for actual image-data paths and optional overrides.
OUTPUT = Path("reproducibility/raw_logs/srinidhi") / ("notebook-" + MODE + "-" + utc_now().replace(":", "-")) / TASK
print({{"task": TASK, "mode": MODE, "device": DEVICE, "output": str(OUTPUT)}})'''),
        nb.v4.new_code_cell('''result = execute(TASK, MODE, OUTPUT, DEVICE, overrides=OVERRIDES)
result["summary"]'''),
        nb.v4.new_code_cell('''from IPython.display import display, Image
figures = sorted(p for p in OUTPUT.rglob("*.png") if p.is_file() and
                 (TASK != "cyclegan" or "grids" in p.relative_to(OUTPUT).parts))
for path in figures:
    print(path.relative_to(OUTPUT))
    display(Image(filename=str(path), width=700))
print("Artifacts:")
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file() and path.suffix not in {".pt", ".pth"}:
        print(path.relative_to(OUTPUT))'''),
        nb.v4.new_markdown_cell("## Interpretation to write after the real run\n\n"
            "Explain the architecture and hyperparameters in your own words. Report the actual data split and hardware. "
            "Use generated evidence for metrics and failure analysis. Keep smoke, cloud rehearsal, and college full-run results separate. "
            "For the team report, compare all members on the agreed evaluation protocol.")]
    notebook = nb.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}})
    destination = ROOT / folder / "srinidhi" / "src" / f"{task}.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nb.write(notebook, destination)
    print(destination.relative_to(ROOT))
