"""
run_until_done.py
=================
Keep restarting a baseline script until the output file reaches target_count unique samples.
Usage:
  python run_until_done.py --script "baselines/gpt_fewshot.py meip --input data/meip_samples.jsonl --output results/gpt5_fewshot_meip_pred.jsonl --model gpt-5.2" --target 500 --id-field id
"""
import json
import subprocess
import sys
import time
import argparse
from pathlib import Path

def count_unique(path: Path, id_field: str = "id") -> int:
    if not path.exists():
        return 0
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)[id_field])
            except Exception:
                pass
    return len(ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Command to run (without python)")
    parser.add_argument("--output", required=True, help="Output file to monitor")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300, help="Seconds before killing subprocess")
    args = parser.parse_args()

    output_path = Path(args.output)
    cmd = [sys.executable] + args.script.split()

    for round_num in range(1, args.max_rounds + 1):
        current = count_unique(output_path)
        print(f"\n[Round {round_num}] Current unique: {current}/{args.target}")
        if current >= args.target:
            print("✅ Target reached!")
            break
        print(f"Running: {' '.join(cmd)}")
        proc = None
        try:
            proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[WARN] Timeout after {args.timeout}s, killing and restarting...")
            if proc:
                proc.kill()
                proc.wait()
        except Exception as e:
            print(f"[WARN] Error: {e}, restarting...")
            if proc:
                try: proc.kill()
                except: pass
        time.sleep(2)

    final = count_unique(output_path)
    print(f"\nFinal: {final}/{args.target}")
    if final < args.target:
        print("⚠️  Did not reach target after max rounds")
        sys.exit(1)

if __name__ == "__main__":
    main()
