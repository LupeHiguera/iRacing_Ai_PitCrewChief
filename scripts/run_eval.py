"""
Quick Evaluation Runner - Run all evaluation steps.

Usage:
    python scripts/run_eval.py                    # Run comprehensive eval only
    python scripts/run_eval.py --all              # Run comprehensive + LLM judge
    python scripts/run_eval.py --quick            # Run 10 cases only (quick test)
    python scripts/run_eval.py --category fuel_critical  # Specific category
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--all", action="store_true", help="Run all evaluations including LLM judge")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 cases")
    parser.add_argument("--category", type=str, help="Specific category to evaluate")
    parser.add_argument("--adapter", type=str, default="models/race-engineer-llama", help="Adapter path")
    parser.add_argument("--llm-judge-api", type=str, default="anthropic", help="API for LLM judge")

    args = parser.parse_args()

    # Build comprehensive eval command
    comp_cmd = [
        sys.executable, "scripts/eval_comprehensive.py",
        "--adapter", args.adapter,
        "--output", "data/eval_comprehensive.json",
    ]

    if args.quick:
        comp_cmd.extend(["--cases", "10"])
    if args.category:
        comp_cmd.extend(["--category", args.category])

    # Run comprehensive evaluation
    success = run_command(comp_cmd, "Comprehensive Evaluation")

    if not success:
        print("\nComprehensive evaluation failed!")
        return 1

    # Run LLM judge if requested
    if args.all:
        judge_cmd = [
            sys.executable, "scripts/eval_llm_judge.py",
            "--input", "data/eval_comprehensive.json",
            "--output", "data/eval_llm_judge.json",
            "--api", args.llm_judge_api,
        ]

        if args.quick:
            judge_cmd.extend(["--cases", "10"])

        success = run_command(judge_cmd, "LLM Judge Evaluation")

        if not success:
            print("\nLLM judge evaluation failed!")
            return 1

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nResults saved to:")
    print("  - data/eval_comprehensive.json")
    if args.all:
        print("  - data/eval_llm_judge.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
