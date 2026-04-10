"""Train deep learning models in separate runs without touching existing scripts.

Usage:
    python scripts/experiments/train_models_separately.py
    python scripts/experiments/train_models_separately.py --config configs/default.yaml
    python scripts/experiments/train_models_separately.py --models gcn mlp_baseline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MODELS = ["gcn", "rgcn", "mlp_baseline"]


def run_command(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Train models separately")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to train one-by-one",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop if any model training fails",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_script = root / "scripts" / "train.py"

    if not train_script.exists():
        print(f"Training script not found: {train_script}")
        return 1

    failed: list[str] = []

    for model in args.models:
        print("\n" + "=" * 70)
        print(f"Training model: {model}")
        print("=" * 70)

        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            args.config,
            "--model",
            model,
        ]
        code = run_command(cmd)
        if code != 0:
            failed.append(model)
            print(f"Model failed: {model} (exit {code})")
            if args.stop_on_error:
                break

    print("\n" + "=" * 70)
    if failed:
        print("Completed with failures.")
        print("Failed models:", ", ".join(failed))
        return 1

    print("All requested models finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
