"""
LLM-as-Judge Evaluation for Race Engineer Model.

Uses Claude or another LLM to qualitatively score model responses.
This provides semantic evaluation that rule-based metrics cannot capture.

Features:
- Qualitative scoring on 5 dimensions
- Side-by-side comparison with blind judging
- Works with pre-generated responses or live inference
- Outputs detailed judgments with explanations

Usage:
    # Evaluate from saved results
    python scripts/eval_llm_judge.py --input data/eval_comprehensive.json

    # Use specific API
    python scripts/eval_llm_judge.py --input data/eval_comprehensive.json --api anthropic

    # Evaluate subset
    python scripts/eval_llm_judge.py --input data/eval_comprehensive.json --cases 10
"""

import json
import argparse
import random
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import time

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# JUDGING CRITERIA
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert racing simulator evaluator. You will judge AI race engineer responses based on 5 criteria.

For each response, score from 1-5 on:

1. **Urgency Match** (1-5): Does the response match the situation's urgency?
   - 5 = Perfect match (critical situations get immediate action calls)
   - 3 = Acceptable but not optimal
   - 1 = Completely wrong tone/urgency

2. **Actionability** (1-5): Does it give the driver something specific to do?
   - 5 = Clear, specific action ("Box this lap", "Defend into turn 3")
   - 3 = Vague action ("Be careful")
   - 1 = No actionable guidance

3. **Telemetry Grounding** (1-5): Does it reference actual telemetry values appropriately?
   - 5 = References specific relevant values (gaps, fuel, temps)
   - 3 = General references without specifics
   - 1 = Ignores telemetry or makes up values

4. **Conciseness** (1-5): Is it brief enough for TTS while racing?
   - 5 = Perfect length (15-30 words), easy to understand while driving
   - 3 = Too long or too short
   - 1 = Way too verbose or unintelligibly brief

5. **Authenticity** (1-5): Does it sound like a real race engineer?
   - 5 = Professional, calm, uses correct terminology
   - 3 = Acceptable but awkward or off-tone
   - 1 = Sounds robotic, uses wrong terms, hallucinations

Also identify any critical errors:
- Hallucinated driver names (Hamilton, Verstappen in non-F1 context)
- Wrong systems mentioned (DRS in GT3, KERS in modern cars)
- Dangerous advice (push when should pit)
- Values that don't match input

Output your judgment as JSON:
{
  "urgency_score": 1-5,
  "actionability_score": 1-5,
  "grounding_score": 1-5,
  "conciseness_score": 1-5,
  "authenticity_score": 1-5,
  "total_score": 5-25,
  "critical_errors": ["list of errors or empty"],
  "explanation": "Brief 1-2 sentence explanation"
}"""


COMPARISON_SYSTEM_PROMPT = """You are an expert racing simulator evaluator. Compare two race engineer responses (A and B) for the same racing situation.

Judge which response is BETTER for a driver racing in a simulator. Consider:
1. Does it match the urgency of the situation?
2. Does it give actionable guidance?
3. Does it reference relevant telemetry?
4. Is it concise enough for TTS while racing?
5. Does it sound authentic?

IMPORTANT: You are doing BLIND evaluation. Do not favor based on position (A vs B).

Output your judgment as JSON:
{
  "winner": "A" or "B" or "TIE",
  "winner_margin": "clear" or "slight" or "tie",
  "a_score": 1-10,
  "b_score": 1-10,
  "key_difference": "Brief explanation of main difference",
  "explanation": "1-2 sentence rationale"
}"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class JudgeScore:
    """Score from LLM judge for a single response."""
    urgency_score: int
    actionability_score: int
    grounding_score: int
    conciseness_score: int
    authenticity_score: int
    total_score: int
    critical_errors: List[str]
    explanation: str


@dataclass
class ComparisonResult:
    """Result of A/B comparison."""
    winner: str  # "A", "B", or "TIE"
    winner_margin: str
    a_score: int
    b_score: int
    key_difference: str
    explanation: str


@dataclass
class JudgedCase:
    """A test case with LLM judge scores."""
    name: str
    category: str
    input_summary: str
    base_response: str
    finetuned_response: str
    base_judge_score: Optional[JudgeScore]
    finetuned_judge_score: Optional[JudgeScore]
    comparison: Optional[ComparisonResult]


# =============================================================================
# LLM CLIENTS
# =============================================================================

def call_anthropic(prompt: str, system: str) -> str:
    """Call Anthropic API."""
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_openai(prompt: str, system: str, base_url: str = None) -> str:
    """Call OpenAI-compatible API (works with LM Studio)."""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")

    client = OpenAI(
        base_url=base_url or "http://localhost:1234/v1",
        api_key="not-needed"
    )
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content


def call_llm(prompt: str, system: str, api: str = "anthropic", base_url: str = None) -> str:
    """Call LLM based on selected API."""
    if api == "anthropic":
        return call_anthropic(prompt, system)
    elif api == "openai" or api == "lmstudio":
        return call_openai(prompt, system, base_url)
    else:
        raise ValueError(f"Unknown API: {api}")


# =============================================================================
# JUDGING FUNCTIONS
# =============================================================================

def judge_single_response(
    response: str,
    input_data: Dict[str, Any],
    test_case: Dict[str, Any],
    api: str = "anthropic",
    base_url: str = None,
) -> JudgeScore:
    """Get LLM judge score for a single response."""

    prompt = f"""Judge this race engineer response.

SITUATION:
- Car: {input_data.get('car', 'Unknown')} ({input_data.get('car_class', 'Unknown')})
- Track: {input_data.get('track', 'Unknown')}
- Position: P{input_data.get('position', '?')}
- Fuel: {input_data.get('fuel_laps_remaining', '?')} laps remaining
- Gap ahead: {input_data.get('gap_ahead', '?')}s
- Gap behind: {input_data.get('gap_behind', '?')}s
- Tire wear: FL {input_data.get('tire_wear', {}).get('fl', '?')}%, FR {input_data.get('tire_wear', {}).get('fr', '?')}%
- Tire temps: FL {input_data.get('tire_temps', {}).get('fl', '?')}°C, FR {input_data.get('tire_temps', {}).get('fr', '?')}°C
- Category: {test_case.get('category', 'unknown')} (urgency: {test_case.get('urgency', 'info')})

RESPONSE TO JUDGE:
"{response}"

Score this response on the 5 criteria and output JSON only."""

    try:
        result = call_llm(prompt, JUDGE_SYSTEM_PROMPT, api, base_url)

        # Parse JSON from response
        # Handle potential markdown code blocks
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]

        data = json.loads(result.strip())

        return JudgeScore(
            urgency_score=data.get("urgency_score", 3),
            actionability_score=data.get("actionability_score", 3),
            grounding_score=data.get("grounding_score", 3),
            conciseness_score=data.get("conciseness_score", 3),
            authenticity_score=data.get("authenticity_score", 3),
            total_score=data.get("total_score", 15),
            critical_errors=data.get("critical_errors", []),
            explanation=data.get("explanation", ""),
        )
    except Exception as e:
        print(f"  Warning: Judge failed - {e}")
        return JudgeScore(
            urgency_score=0,
            actionability_score=0,
            grounding_score=0,
            conciseness_score=0,
            authenticity_score=0,
            total_score=0,
            critical_errors=[f"Judge failed: {str(e)}"],
            explanation="Evaluation failed",
        )


def compare_responses(
    response_a: str,
    response_b: str,
    input_data: Dict[str, Any],
    test_case: Dict[str, Any],
    api: str = "anthropic",
    base_url: str = None,
) -> ComparisonResult:
    """Compare two responses using LLM judge."""

    prompt = f"""Compare these two race engineer responses for the same situation.

SITUATION:
- Car: {input_data.get('car', 'Unknown')} ({input_data.get('car_class', 'Unknown')})
- Track: {input_data.get('track', 'Unknown')}
- Position: P{input_data.get('position', '?')}
- Fuel: {input_data.get('fuel_laps_remaining', '?')} laps remaining
- Gap ahead: {input_data.get('gap_ahead', '?')}s
- Gap behind: {input_data.get('gap_behind', '?')}s
- Category: {test_case.get('category', 'unknown')} (urgency: {test_case.get('urgency', 'info')})

RESPONSE A:
"{response_a}"

RESPONSE B:
"{response_b}"

Which response is better for a driver racing in a simulator? Output JSON only."""

    try:
        result = call_llm(prompt, COMPARISON_SYSTEM_PROMPT, api, base_url)

        # Parse JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]

        data = json.loads(result.strip())

        return ComparisonResult(
            winner=data.get("winner", "TIE"),
            winner_margin=data.get("winner_margin", "tie"),
            a_score=data.get("a_score", 5),
            b_score=data.get("b_score", 5),
            key_difference=data.get("key_difference", ""),
            explanation=data.get("explanation", ""),
        )
    except Exception as e:
        print(f"  Warning: Comparison failed - {e}")
        return ComparisonResult(
            winner="TIE",
            winner_margin="tie",
            a_score=5,
            b_score=5,
            key_difference=f"Evaluation failed: {str(e)}",
            explanation="",
        )


# =============================================================================
# EVALUATION
# =============================================================================

def run_llm_judge_evaluation(
    results_path: str,
    api: str = "anthropic",
    base_url: str = None,
    num_cases: Optional[int] = None,
    blind_comparison: bool = True,
) -> List[JudgedCase]:
    """Run LLM judge evaluation on saved results."""

    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", data)  # Handle both formats
    if isinstance(results, list) and len(results) > 0 and "results" in results[0]:
        results = results[0]["results"]

    if num_cases:
        results = results[:num_cases]

    print(f"Evaluating {len(results)} cases with LLM judge (API: {api})")

    judged_cases = []

    for i, case in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] {case['name']}")

        # Reconstruct test case dict
        test_case = {
            "category": case.get("category", "unknown"),
            "urgency": case.get("urgency", "info"),
        }

        # Get input data
        if "input_data" in case:
            input_data = case["input_data"]
        else:
            # Try to parse from input_summary
            input_data = {}

        base_response = case.get("base_response", "")
        ft_response = case.get("finetuned_response", "")

        # Judge individual responses
        print("  Judging base response...")
        base_score = judge_single_response(
            base_response, input_data, test_case, api, base_url
        )
        time.sleep(0.5)  # Rate limiting

        print("  Judging fine-tuned response...")
        ft_score = judge_single_response(
            ft_response, input_data, test_case, api, base_url
        )
        time.sleep(0.5)

        # Blind comparison (randomize order)
        comparison = None
        if blind_comparison:
            print("  Running blind comparison...")
            if random.random() > 0.5:
                comparison = compare_responses(
                    base_response, ft_response, input_data, test_case, api, base_url
                )
                # A = base, B = ft
                if comparison.winner == "B":
                    comparison.winner = "finetuned"
                elif comparison.winner == "A":
                    comparison.winner = "base"
            else:
                comparison = compare_responses(
                    ft_response, base_response, input_data, test_case, api, base_url
                )
                # A = ft, B = base
                if comparison.winner == "A":
                    comparison.winner = "finetuned"
                elif comparison.winner == "B":
                    comparison.winner = "base"

            time.sleep(0.5)

        judged_case = JudgedCase(
            name=case["name"],
            category=case.get("category", "unknown"),
            input_summary=case.get("input_summary", ""),
            base_response=base_response,
            finetuned_response=ft_response,
            base_judge_score=base_score,
            finetuned_judge_score=ft_score,
            comparison=comparison,
        )
        judged_cases.append(judged_case)

        # Progress summary
        print(f"  Base score: {base_score.total_score}/25, FT score: {ft_score.total_score}/25")
        if comparison:
            print(f"  Comparison winner: {comparison.winner} ({comparison.winner_margin})")

    return judged_cases


def print_llm_judge_results(judged_cases: List[JudgedCase]):
    """Print LLM judge evaluation results."""

    print("\n" + "=" * 80)
    print("LLM JUDGE EVALUATION RESULTS")
    print("=" * 80)

    n = len(judged_cases)

    # Aggregate scores
    base_scores = [c.base_judge_score.total_score for c in judged_cases if c.base_judge_score]
    ft_scores = [c.finetuned_judge_score.total_score for c in judged_cases if c.finetuned_judge_score]

    base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
    ft_avg = sum(ft_scores) / len(ft_scores) if ft_scores else 0

    print(f"\n{'AGGREGATE LLM JUDGE SCORES':^80}")
    print("-" * 80)
    print(f"{'Model':<20} {'Avg Score':>15} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    print(f"{'Base':<20} {base_avg:>15.1f}/25 {min(base_scores) if base_scores else 0:>10} {max(base_scores) if base_scores else 0:>10}")
    print(f"{'Fine-tuned':<20} {ft_avg:>15.1f}/25 {min(ft_scores) if ft_scores else 0:>10} {max(ft_scores) if ft_scores else 0:>10}")

    # Per-dimension breakdown
    print(f"\n{'PER-DIMENSION BREAKDOWN':^80}")
    print("-" * 80)

    dimensions = ["urgency", "actionability", "grounding", "conciseness", "authenticity"]
    print(f"{'Dimension':<20} {'Base Avg':>15} {'FT Avg':>15} {'Winner':>15}")
    print("-" * 70)

    for dim in dimensions:
        base_dim = [getattr(c.base_judge_score, f"{dim}_score") for c in judged_cases if c.base_judge_score]
        ft_dim = [getattr(c.finetuned_judge_score, f"{dim}_score") for c in judged_cases if c.finetuned_judge_score]

        base_dim_avg = sum(base_dim) / len(base_dim) if base_dim else 0
        ft_dim_avg = sum(ft_dim) / len(ft_dim) if ft_dim else 0

        winner = "Fine-tuned" if ft_dim_avg > base_dim_avg else "Base" if base_dim_avg > ft_dim_avg else "Tie"
        print(f"{dim.capitalize():<20} {base_dim_avg:>15.2f}/5 {ft_dim_avg:>15.2f}/5 {winner:>15}")

    # Comparison results
    comparisons = [c.comparison for c in judged_cases if c.comparison]
    if comparisons:
        ft_wins = sum(1 for c in comparisons if c.winner == "finetuned")
        base_wins = sum(1 for c in comparisons if c.winner == "base")
        ties = len(comparisons) - ft_wins - base_wins

        print(f"\n{'BLIND COMPARISON RESULTS':^80}")
        print("-" * 80)
        print(f"Fine-tuned wins: {ft_wins} ({ft_wins/len(comparisons)*100:.1f}%)")
        print(f"Base wins:       {base_wins} ({base_wins/len(comparisons)*100:.1f}%)")
        print(f"Ties:            {ties} ({ties/len(comparisons)*100:.1f}%)")

    # Critical errors
    base_errors = sum(len(c.base_judge_score.critical_errors) for c in judged_cases if c.base_judge_score)
    ft_errors = sum(len(c.finetuned_judge_score.critical_errors) for c in judged_cases if c.finetuned_judge_score)

    print(f"\n{'CRITICAL ERRORS':^80}")
    print("-" * 80)
    print(f"Base model errors: {base_errors}")
    print(f"Fine-tuned errors: {ft_errors}")

    # Final verdict
    print(f"\n{'=' * 80}")
    print(f"{'LLM JUDGE VERDICT':^80}")
    print(f"{'=' * 80}")

    improvement = ft_avg - base_avg

    if comparisons:
        win_rate = ft_wins / len(comparisons) * 100
    else:
        win_rate = 50

    if improvement > 2 and win_rate > 55:
        verdict = "STRONG IMPROVEMENT - LLM judge strongly prefers fine-tuned model"
    elif improvement > 0 and win_rate > 45:
        verdict = "MODERATE IMPROVEMENT - Fine-tuned model shows gains"
    elif improvement > -2:
        verdict = "MARGINAL DIFFERENCE - Models perform similarly"
    else:
        verdict = "REGRESSION - Base model may be preferable"

    print(f"\nScore improvement: {improvement:+.1f}/25 points")
    if comparisons:
        print(f"Blind comparison win rate: {win_rate:.1f}%")
    print(f"\n>>> {verdict} <<<")
    print()


def save_llm_judge_results(judged_cases: List[JudgedCase], output_path: str):
    """Save LLM judge results to JSON."""

    data = {
        "summary": {
            "total_cases": len(judged_cases),
            "base_avg_score": sum(c.base_judge_score.total_score for c in judged_cases if c.base_judge_score) / len(judged_cases),
            "ft_avg_score": sum(c.finetuned_judge_score.total_score for c in judged_cases if c.finetuned_judge_score) / len(judged_cases),
        },
        "cases": []
    }

    for case in judged_cases:
        case_data = {
            "name": case.name,
            "category": case.category,
            "input_summary": case.input_summary,
            "base_response": case.base_response,
            "finetuned_response": case.finetuned_response,
        }

        if case.base_judge_score:
            case_data["base_judge_score"] = asdict(case.base_judge_score)
        if case.finetuned_judge_score:
            case_data["finetuned_judge_score"] = asdict(case.finetuned_judge_score)
        if case.comparison:
            case_data["comparison"] = asdict(case.comparison)

        data["cases"].append(case_data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"LLM judge results saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation for race engineer model"
    )
    parser.add_argument(
        "--input", type=str,
        default="data/eval_comprehensive.json",
        help="Input file with evaluation results"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/eval_llm_judge.json",
        help="Output path for LLM judge results"
    )
    parser.add_argument(
        "--api", type=str,
        choices=["anthropic", "openai", "lmstudio"],
        default="anthropic",
        help="API to use for judging"
    )
    parser.add_argument(
        "--base-url", type=str,
        default=None,
        help="Base URL for OpenAI-compatible API (default: localhost:1234)"
    )
    parser.add_argument(
        "--cases", type=int,
        default=None,
        help="Number of cases to evaluate (default: all)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip blind A/B comparison"
    )

    args = parser.parse_args()

    # Check API availability
    if args.api == "anthropic" and not HAS_ANTHROPIC:
        print("Error: anthropic package not installed. Install with: pip install anthropic")
        return
    if args.api in ["openai", "lmstudio"] and not HAS_OPENAI:
        print("Error: openai package not installed. Install with: pip install openai")
        return

    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run eval_comprehensive.py first to generate evaluation results.")
        return

    # Run evaluation
    judged_cases = run_llm_judge_evaluation(
        results_path=args.input,
        api=args.api,
        base_url=args.base_url,
        num_cases=args.cases,
        blind_comparison=not args.no_comparison,
    )

    # Print results
    print_llm_judge_results(judged_cases)

    # Save results
    save_llm_judge_results(judged_cases, args.output)


if __name__ == "__main__":
    main()
