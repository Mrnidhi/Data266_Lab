"""Run the unchanged class scoring notebook against the saved model predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

MEMBER = Path(__file__).resolve().parent
ROOT = MEMBER.parents[1]
NOTEBOOK = MEMBER / "src/Part3_Evaluation_Script.ipynb"
EVALUATOR_SHA256 = "702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef"


def validate_inputs(data_root: Path, predictions: Path, output: Path) -> None:
    if hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() != EVALUATOR_SHA256:
        raise ValueError("The supplied evaluation notebook has changed.")
    if output.exists():
        raise FileExistsError("Use a new output directory; saved results are never overwritten.")
    for parent, name, expected in (
        (data_root, "monet_jpg", 300),
        (data_root, "photo_jpg", 7038),
        (predictions, "pred_A2B", 300),
        (predictions, "pred_B2A", 7038),
    ):
        count = sum(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for p in (parent / name).glob("*")
            if p.is_file()
        )
        if count != expected:
            raise ValueError(f"{name}: expected {expected} images, found {count}.")


def evaluate(data_root: Path, predictions: Path, output: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    data_root, predictions, output = (p.resolve() for p in (data_root, predictions, output))
    validate_inputs(data_root, predictions, output)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="class-evaluation-") as temporary:
        working = Path(temporary)
        layout = working / "Part 3/Data"
        layout.mkdir(parents=True)
        for parent, names in (
            (data_root, ("monet_jpg", "photo_jpg")),
            (predictions, ("pred_A2B", "pred_B2A")),
        ):
            for name in names:
                shutil.copytree(parent / name, layout / name)
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        try:
            NotebookClient(
                notebook, timeout=7200, km=manager, resources={"metadata": {"path": str(working)}}
            ).execute()
        finally:
            nbformat.write(notebook, output / "Part3_Evaluation_Script.executed.ipynb")
        shutil.copyfile(working / "submission.csv", output / "submission.csv")
    with (output / "submission.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    result = {"FID": float(row["FID"]), "MiFID": float(row["MiFID"])}
    result["composite"] = (result["FID"] + result["MiFID"]) / 2
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "task3_gan/data")
    parser.add_argument("--predictions", type=Path, default=MEMBER / "outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data_root, args.predictions, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
