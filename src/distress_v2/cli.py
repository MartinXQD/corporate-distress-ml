"""Public entry points for the frozen financial distress project."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    if sys.argv[1:2] == ["infer"]:
        from .inference import main as infer
        sys.argv = ["distress-ml infer", *sys.argv[2:]]
        infer()
        return
    parser = argparse.ArgumentParser(description="Corporate Financial Distress Classification")
    parser.add_argument("command", choices=["status", "verify", "infer"])
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "status":
        if remaining:
            parser.error("status takes no additional arguments")
        choice = json.loads((root / "configs/final_model.json").read_text())
        print(json.dumps(choice, indent=2))
    elif args.command == "verify":
        subprocess.run([sys.executable, str(root / "scripts/verify_project.py"), *remaining], check=True)
    else:
        from .inference import main as infer
        sys.argv = ["distress-ml infer", *remaining]
        infer()


if __name__ == "__main__":
    main()
