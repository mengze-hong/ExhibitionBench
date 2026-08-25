from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_cmd(cmd: list[str]) -> None:
    print(f"[PIPELINE] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(BASE))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed (exit={completed.returncode}): {' '.join(cmd)}")


def run_validate() -> None:
    run_cmd([PY, "scripts/validate_data.py"])


def run_recompile() -> None:
    for shot in ("0", "1", "3"):
        run_cmd([PY, "scripts/compile_results.py", "--shot", shot, "--latex"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible local pipeline for ExhibitionBench: "
            "validate released data and recompile result tables"
        )
    )
    parser.add_argument(
        "--stage",
        choices=["validate", "recompile", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.stage in ("validate", "all"):
        run_validate()
    if args.stage in ("recompile", "all"):
        run_recompile()

    print("[PIPELINE] done")


if __name__ == "__main__":
    main()
