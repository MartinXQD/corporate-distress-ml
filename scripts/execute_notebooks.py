"""Execute the public notebooks without fitting research models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    records = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        if not args.check_only:
            # Match the current compatible interpreter, rather than an unrelated
            # system kernelspec. This never runs a training entry point.
            import sys
            client = NotebookClient(notebook, timeout=300, resources={"metadata": {"path": str(ROOT)}})
            client.kernel_manager_class = "jupyter_client.manager.KernelManager"
            from jupyter_client import KernelManager
            from jupyter_client.kernelspec import KernelSpec
            manager = KernelManager()
            manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                              display_name="Python 3", language="python")
            client.km = manager
            client.execute()
            nbformat.write(notebook, path)
        codes = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in codes)
        assert all(o.output_type != "error" for c in codes for o in c.outputs)
        assert "/Users/" not in path.read_text() and "/home/" not in path.read_text()
        records.append({"path": str(path.relative_to(ROOT)), "code_cells": len(codes),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {"passed": True, "fresh_execution": not args.check_only,
              "private_artifacts_present": (ROOT / "artifacts/models").exists(),
              "research_training_triggered": False, "notebooks": records}
    if not args.check_only:
        (ROOT / "results/notebook_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
