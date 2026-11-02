"""Execute the pipeline or feature-engineering experiment, preserving cell outputs."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 60 VOIs into a separate artifacts/smoke folder")
    notebook_selection = parser.add_mutually_exclusive_group()
    notebook_selection.add_argument("--feature-engineering", action="store_true", help="Run only notebook 03; requires existing 01/02 artifacts")
    notebook_selection.add_argument("--include-feature-engineering", action="store_true", help="Run notebooks 01, 02 and 03 in order")
    notebook_selection.add_argument("--f1", action="store_true", help="Run only notebook 04; requires existing 01/02/03 artifacts")
    notebook_selection.add_argument("--include-f1", action="store_true", help="Run notebooks 01 through 04 in order")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.smoke:
        os.environ["LUNA_MAX_CASES"] = "60"
        os.environ["LUNA_OUTPUT_SUBDIR"] = "smoke"
        os.environ["LUNA_SMOKE"] = "1"
    else:
        for key in ("LUNA_MAX_CASES", "LUNA_OUTPUT_SUBDIR", "LUNA_SMOKE"):
            os.environ.pop(key, None)

    # A temporary kernel points at exactly the interpreter running this script.
    with tempfile.TemporaryDirectory(prefix="luna-kernel-") as directory:
        kernel_dir = Path(directory) / "luna-radiomics-runner"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "LUNA radiomics runner", "language": "python",
        }), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[directory])
        names = ["01_voi_radiomics.ipynb", "02_malign_benign_modeling.ipynb"]
        if args.feature_engineering:
            names = ["03_feature_engineering_ablation.ipynb"]
        elif args.f1:
            names = ["04_f1_optimization.ipynb"]
        elif args.include_feature_engineering:
            names.append("03_feature_engineering_ablation.ipynb")
        elif args.include_f1:
            names.extend(["03_feature_engineering_ablation.ipynb", "04_f1_optimization.ipynb"])
        for name in names:
            path = root / name
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(
                notebook, timeout=3600, kernel_name="luna-radiomics-runner",
                resources={"metadata": {"path": str(root)}},
            )
            client.km = AsyncKernelManager(
                kernel_name="luna-radiomics-runner", kernel_spec_manager=manager,
                autorestart=False,
            )
            client.on_cell_start = lambda cell, cell_index, **kwargs: print(
                f"  cell {cell_index + 1}: {cell.cell_type}", flush=True
            )
            print(f"Executing {name}", flush=True)
            try:
                client.execute()
            finally:
                if args.smoke:
                    destination = root / "artifacts" / "smoke" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                else:
                    destination = path
                nbformat.write(notebook, destination)
            print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
