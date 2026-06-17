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


def run_qc() -> None:
    run_cmd([PY, "scripts/verify_experiment_completeness.py"])


def run_manifest() -> None:
    run_cmd([PY, "scripts/generate_run_manifest.py"])
    run_qc()


def run_backfill(execute: bool, retries: int, max_runs: int) -> None:
    cmd = [
        PY,
        "scripts/run_fullrun_backfill.py",
        "--retries",
        str(retries),
        "--max-runs",
        str(max_runs),
    ]
    if execute:
        cmd.append("--execute")
    run_cmd(cmd)


def run_recompile() -> None:
    for shot in ("0", "1", "3"):
        run_cmd([PY, "results/compile_sota_results.py", "--shot", shot, "--tag", "fullrun", "--latex"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified reproducible pipeline for ExhibitionBench: "
            "manifest -> quality check -> backfill -> recompile"
        )
    )
    parser.add_argument(
        "--stage",
        choices=["manifest", "qc", "backfill", "recompile", "all"],
        default="all",
    )
    parser.add_argument("--execute", action="store_true", help="For backfill stage: actually execute missing jobs")
    parser.add_argument("--retries", type=int, default=1, help="For backfill stage: retries per failed command")
    parser.add_argument("--max-runs", type=int, default=0, help="For backfill stage: max commands (0=all)")
    args = parser.parse_args()

    if args.stage in ("manifest", "all"):
        run_manifest()
    if args.stage in ("qc", "all"):
        run_qc()
    if args.stage in ("backfill", "all"):
        run_backfill(execute=args.execute, retries=max(0, args.retries), max_runs=max(0, args.max_runs))
    if args.stage in ("recompile", "all"):
        run_recompile()

    print("[PIPELINE] done")


if __name__ == "__main__":
    main()
