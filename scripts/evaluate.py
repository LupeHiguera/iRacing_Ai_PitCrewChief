"""
Evaluation framework for race engineer model.

Compares base model vs fine-tuned model responses on held-out test cases.
Measures response quality: length, grounding, track/car awareness.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --adapter models/race-engineer-llama --output results.json
"""

import json
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# =============================================================================
# EVALUATION TEST CASES
# =============================================================================

# Held-out test cases not seen during training
EVAL_CASES = [
    {
        "name": "Fuel Critical - Imola",
        "category": "fuel_critical",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Imola",
            "track_type": "mixed",
            "lap": 22,
            "lap_pct": 0.45,
            "position": 5,
            "fuel_laps_remaining": 1.5,
            "tire_wear": {"fl": 35, "fr": 40, "rl": 28, "rr": 30},
            "tire_temps": {"fl": 94, "fr": 98, "rl": 88, "rr": 90},
            "gap_ahead": 2.8,
            "gap_behind": 1.5,
            "last_lap_time": 101.2,
            "best_lap_time": 100.5,
            "session_laps_remain": 12,
            "incident_count": 1,
            "track_temp_c": 34,
        },
        "expected_elements": ["box", "fuel", "pit"],
    },
    {
        "name": "Battle Mode - Nurburgring GP",
        "category": "position_battle",
        "input": {
            "car": "Mercedes-AMG GT3",
            "car_class": "GT3",
            "car_traits": ["front_engine", "stable", "strong_brakes"],
            "track": "Nurburgring GP",
            "track_type": "mixed",
            "lap": 10,
            "lap_pct": 0.72,
            "position": 3,
            "fuel_laps_remaining": 15.0,
            "tire_wear": {"fl": 18, "fr": 22, "rl": 15, "rr": 17},
            "tire_temps": {"fl": 92, "fr": 95, "rl": 86, "rr": 88},
            "gap_ahead": 0.4,
            "gap_behind": 3.2,
            "last_lap_time": 115.8,
            "best_lap_time": 115.2,
            "session_laps_remain": 20,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["gap", "attack", "ahead", "push"],
    },
    {
        "name": "Tire Management - Barcelona",
        "category": "tire_warning",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Barcelona",
            "track_type": "mixed",
            "lap": 18,
            "lap_pct": 0.35,
            "position": 7,
            "fuel_laps_remaining": 12.0,
            "tire_wear": {"fl": 45, "fr": 52, "rl": 38, "rr": 42},
            "tire_temps": {"fl": 96, "fr": 108, "rl": 90, "rr": 94},
            "gap_ahead": 1.8,
            "gap_behind": 2.5,
            "last_lap_time": 102.5,
            "best_lap_time": 101.8,
            "session_laps_remain": 15,
            "incident_count": 0,
            "track_temp_c": 38,
        },
        "expected_elements": ["tire", "front", "wear", "manage"],
    },
    {
        "name": "Pace Feedback - Suzuka",
        "category": "pace_feedback",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Suzuka",
            "track_type": "mixed",
            "lap": 8,
            "lap_pct": 0.68,
            "position": 4,
            "fuel_laps_remaining": 20.0,
            "tire_wear": {"fl": 12, "fr": 15, "rl": 10, "rr": 12},
            "tire_temps": {"fl": 88, "fr": 92, "rl": 84, "rr": 86},
            "gap_ahead": 4.5,
            "gap_behind": 5.2,
            "last_lap_time": 120.2,
            "best_lap_time": 121.5,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["pace", "lap", "time", "good"],
    },
    {
        "name": "Defend Under Pressure - Road America",
        "category": "position_battle",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Road America",
            "track_type": "high_speed",
            "lap": 12,
            "lap_pct": 0.88,
            "position": 2,
            "fuel_laps_remaining": 8.0,
            "tire_wear": {"fl": 28, "fr": 32, "rl": 22, "rr": 25},
            "tire_temps": {"fl": 94, "fr": 98, "rl": 88, "rr": 90},
            "gap_ahead": 6.5,
            "gap_behind": 0.3,
            "last_lap_time": 132.5,
            "best_lap_time": 131.8,
            "session_laps_remain": 10,
            "incident_count": 2,
            "track_temp_c": 30,
        },
        "expected_elements": ["defend", "behind", "pressure"],
    },
    {
        "name": "Pit Approach - Le Mans",
        "category": "pit_approach",
        "input": {
            "car": "Porsche 963",
            "car_class": "LMDh",
            "car_traits": ["hybrid", "high_downforce", "complex_systems"],
            "track": "Le Mans",
            "track_type": "high_speed",
            "lap": 45,
            "lap_pct": 0.92,
            "position": 3,
            "fuel_laps_remaining": 0.8,
            "tire_wear": {"fl": 55, "fr": 58, "rl": 48, "rr": 52},
            "tire_temps": {"fl": 95, "fr": 98, "rl": 90, "rr": 92},
            "gap_ahead": 8.5,
            "gap_behind": 12.0,
            "last_lap_time": 198.5,
            "best_lap_time": 197.2,
            "session_laps_remain": 60,
            "incident_count": 0,
            "track_temp_c": 32,
        },
        "expected_elements": ["box", "pit", "fuel"],
    },
    {
        "name": "Cold Tires - Bathurst",
        "category": "tire_cold",
        "input": {
            "car": "Audi R8 LMS GT3 Evo II",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "predictable"],
            "track": "Bathurst",
            "track_type": "mixed",
            "lap": 2,
            "lap_pct": 0.25,
            "position": 12,
            "fuel_laps_remaining": 28.0,
            "tire_wear": {"fl": 2, "fr": 3, "rl": 1, "rr": 2},
            "tire_temps": {"fl": 55, "fr": 58, "rl": 48, "rr": 52},
            "gap_ahead": 1.2,
            "gap_behind": 0.8,
            "last_lap_time": 145.5,
            "best_lap_time": 145.5,
            "session_laps_remain": 35,
            "incident_count": 0,
            "track_temp_c": 18,
        },
        "expected_elements": ["cold", "tire", "warm", "careful"],
    },
    {
        "name": "Routine Update - Watkins Glen",
        "category": "routine",
        "input": {
            "car": "Lamborghini Huracan GT3 Evo",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "forgiving"],
            "track": "Watkins Glen",
            "track_type": "high_speed",
            "lap": 15,
            "lap_pct": 0.50,
            "position": 6,
            "fuel_laps_remaining": 18.0,
            "tire_wear": {"fl": 20, "fr": 24, "rl": 16, "rr": 18},
            "tire_temps": {"fl": 88, "fr": 92, "rl": 84, "rr": 86},
            "gap_ahead": 3.5,
            "gap_behind": 4.2,
            "last_lap_time": 108.2,
            "best_lap_time": 107.8,
            "session_laps_remain": 20,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["good", "fuel", "lap"],
    },
]


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ResponseMetrics:
    """Metrics for a single response."""
    word_count: int
    is_concise: bool  # Under 40 words
    has_telemetry_reference: bool  # References actual values
    has_track_reference: bool  # Mentions track or corners
    has_car_appropriate_advice: bool  # Car-specific guidance
    expected_elements_found: int
    expected_elements_total: int
    contains_hallucination: bool  # Fake names, wrong systems


@dataclass
class EvalResult:
    """Result for a single test case."""
    name: str
    category: str
    input_summary: str
    base_response: str
    finetuned_response: str
    base_metrics: ResponseMetrics
    finetuned_metrics: ResponseMetrics


def analyze_response(response: str, test_case: Dict[str, Any]) -> ResponseMetrics:
    """Analyze a response and compute metrics."""
    response_lower = response.lower()
    words = response.split()
    word_count = len(words)

    # Check conciseness
    is_concise = word_count <= 40

    # Check telemetry references (numbers, temps, gaps, etc.)
    has_telemetry = bool(re.search(r'\d+\.?\d*', response))

    # Check track references
    track_name = test_case["input"]["track"].lower()
    track_words = track_name.split()
    has_track = any(word in response_lower for word in track_words)

    # Common corner names for different tracks
    corner_keywords = [
        "turn", "corner", "curve", "chicane", "hairpin", "esses",
        "straight", "kink", "corkscrew", "carousel", "dipper"
    ]
    has_track = has_track or any(kw in response_lower for kw in corner_keywords)

    # Check car-appropriate advice
    car_traits = test_case["input"].get("car_traits", [])
    trait_keywords = {
        "rear_engine": ["trail", "brake", "rear", "rotation"],
        "mid_engine": ["balance", "entry", "rotation"],
        "front_engine": ["understeer", "entry", "stable"],
        "awd": ["traction", "power", "stable"],
        "trail_brake_critical": ["trail", "brake", "entry"],
        "momentum_car": ["momentum", "carry", "speed"],
        "hybrid": ["deploy", "energy", "battery"],
    }

    has_car_advice = False
    for trait in car_traits:
        if trait in trait_keywords:
            if any(kw in response_lower for kw in trait_keywords[trait]):
                has_car_advice = True
                break

    # Check expected elements
    expected = test_case.get("expected_elements", [])
    found = sum(1 for elem in expected if elem.lower() in response_lower)

    # Check for hallucinations
    hallucination_patterns = [
        r'\b(hamilton|verstappen|leclerc|norris|bogdan|smith|jones)\b',  # Fake names
        r'\bkers\b',  # KERS in non-F1
        r'\bdrs\b' if test_case["input"]["car_class"] != "F1" else r'(?!)',  # DRS in non-F1
    ]
    contains_hallucination = any(
        re.search(pattern, response_lower)
        for pattern in hallucination_patterns
    )

    return ResponseMetrics(
        word_count=word_count,
        is_concise=is_concise,
        has_telemetry_reference=has_telemetry,
        has_track_reference=has_track,
        has_car_appropriate_advice=has_car_advice,
        expected_elements_found=found,
        expected_elements_total=len(expected),
        contains_hallucination=contains_hallucination,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_base_model(model_name: str):
    """Load the base model without adapter."""
    print(f"Loading base model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def load_finetuned_model(adapter_path: str, base_model: str):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"Loading fine-tuned model from: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


# =============================================================================
# GENERATION
# =============================================================================

def generate_response(model, tokenizer, input_data: dict, max_new_tokens: int = 80) -> str:
    """Generate a response from the model."""
    instruction = "You are a race engineer. Given the car, track, and telemetry, provide a brief callout to the driver."
    input_json = json.dumps(input_data)

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_json}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in response:
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()

    return response.strip()


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(
    base_model_name: str,
    adapter_path: Optional[str],
    test_cases: List[Dict[str, Any]],
) -> List[EvalResult]:
    """Run evaluation on all test cases."""

    results = []

    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    base_model, base_tokenizer = load_base_model(base_model_name)

    if adapter_path and Path(adapter_path).exists():
        ft_model, ft_tokenizer = load_finetuned_model(adapter_path, base_model_name)
    else:
        print(f"Warning: Adapter not found at {adapter_path}, using base model only")
        ft_model, ft_tokenizer = base_model, base_tokenizer

    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test_case['name']}")

        # Generate responses
        print("  Generating base response...")
        base_response = generate_response(base_model, base_tokenizer, test_case["input"])

        print("  Generating fine-tuned response...")
        ft_response = generate_response(ft_model, ft_tokenizer, test_case["input"])

        # Analyze responses
        base_metrics = analyze_response(base_response, test_case)
        ft_metrics = analyze_response(ft_response, test_case)

        # Create input summary
        inp = test_case["input"]
        input_summary = f"{inp['car']} @ {inp['track']}, P{inp['position']}, {inp['fuel_laps_remaining']} laps fuel"

        result = EvalResult(
            name=test_case["name"],
            category=test_case["category"],
            input_summary=input_summary,
            base_response=base_response,
            finetuned_response=ft_response,
            base_metrics=base_metrics,
            finetuned_metrics=ft_metrics,
        )
        results.append(result)

        # Print comparison
        print(f"  Base: {base_response[:80]}...")
        print(f"  Fine-tuned: {ft_response[:80]}...")

    return results


def print_results(results: List[EvalResult]):
    """Print evaluation results summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Per-case results
    for result in results:
        print(f"\n--- {result.name} ({result.category}) ---")
        print(f"Input: {result.input_summary}")
        print(f"\nBase Model:")
        print(f"  Response: {result.base_response}")
        print(f"  Words: {result.base_metrics.word_count}, Concise: {result.base_metrics.is_concise}")
        print(f"  Has telemetry: {result.base_metrics.has_telemetry_reference}, Track: {result.base_metrics.has_track_reference}")
        print(f"  Expected elements: {result.base_metrics.expected_elements_found}/{result.base_metrics.expected_elements_total}")

        print(f"\nFine-tuned Model:")
        print(f"  Response: {result.finetuned_response}")
        print(f"  Words: {result.finetuned_metrics.word_count}, Concise: {result.finetuned_metrics.is_concise}")
        print(f"  Has telemetry: {result.finetuned_metrics.has_telemetry_reference}, Track: {result.finetuned_metrics.has_track_reference}")
        print(f"  Expected elements: {result.finetuned_metrics.expected_elements_found}/{result.finetuned_metrics.expected_elements_total}")
        print("-" * 40)

    # Aggregate metrics
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)

    base_concise = sum(1 for r in results if r.base_metrics.is_concise)
    ft_concise = sum(1 for r in results if r.finetuned_metrics.is_concise)

    base_telemetry = sum(1 for r in results if r.base_metrics.has_telemetry_reference)
    ft_telemetry = sum(1 for r in results if r.finetuned_metrics.has_telemetry_reference)

    base_track = sum(1 for r in results if r.base_metrics.has_track_reference)
    ft_track = sum(1 for r in results if r.finetuned_metrics.has_track_reference)

    base_expected = sum(r.base_metrics.expected_elements_found for r in results)
    ft_expected = sum(r.finetuned_metrics.expected_elements_found for r in results)
    total_expected = sum(r.base_metrics.expected_elements_total for r in results)

    base_hallucinations = sum(1 for r in results if r.base_metrics.contains_hallucination)
    ft_hallucinations = sum(1 for r in results if r.finetuned_metrics.contains_hallucination)

    n = len(results)

    print(f"\n{'Metric':<30} {'Base':>12} {'Fine-tuned':>12} {'Winner':>12}")
    print("-" * 66)
    print(f"{'Concise (<40 words)':<30} {base_concise:>10}/{n} {ft_concise:>10}/{n} {'Fine-tuned' if ft_concise > base_concise else 'Base' if base_concise > ft_concise else 'Tie':>12}")
    print(f"{'Has telemetry reference':<30} {base_telemetry:>10}/{n} {ft_telemetry:>10}/{n} {'Fine-tuned' if ft_telemetry > base_telemetry else 'Base' if base_telemetry > ft_telemetry else 'Tie':>12}")
    print(f"{'Has track reference':<30} {base_track:>10}/{n} {ft_track:>10}/{n} {'Fine-tuned' if ft_track > base_track else 'Base' if base_track > ft_track else 'Tie':>12}")
    print(f"{'Expected elements found':<30} {base_expected:>10}/{total_expected} {ft_expected:>10}/{total_expected} {'Fine-tuned' if ft_expected > base_expected else 'Base' if base_expected > ft_expected else 'Tie':>12}")
    print(f"{'Hallucinations':<30} {base_hallucinations:>10}/{n} {ft_hallucinations:>10}/{n} {'Fine-tuned' if ft_hallucinations < base_hallucinations else 'Base' if base_hallucinations < ft_hallucinations else 'Tie':>12}")

    # Overall winner
    ft_wins = sum([
        ft_concise > base_concise,
        ft_telemetry > base_telemetry,
        ft_track > base_track,
        ft_expected > base_expected,
        ft_hallucinations < base_hallucinations,
    ])
    base_wins = sum([
        base_concise > ft_concise,
        base_telemetry > ft_telemetry,
        base_track > ft_track,
        base_expected > ft_expected,
        base_hallucinations < ft_hallucinations,
    ])

    print(f"\n{'OVERALL WINNER:':<30} {'Fine-tuned' if ft_wins > base_wins else 'Base' if base_wins > ft_wins else 'Tie'} ({ft_wins} vs {base_wins} metrics)")


def save_results(results: List[EvalResult], output_path: str):
    """Save results to JSON file."""
    data = []
    for result in results:
        data.append({
            "name": result.name,
            "category": result.category,
            "input_summary": result.input_summary,
            "base_response": result.base_response,
            "finetuned_response": result.finetuned_response,
            "base_metrics": asdict(result.base_metrics),
            "finetuned_metrics": asdict(result.finetuned_metrics),
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate race engineer model")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name")
    parser.add_argument("--adapter", type=str, default="models/race-engineer-llama",
                        help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, default="data/eval_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--cases", type=int, default=None,
                        help="Number of test cases to run (default: all)")

    args = parser.parse_args()

    # Select test cases
    test_cases = EVAL_CASES
    if args.cases:
        test_cases = test_cases[:args.cases]

    print(f"Running evaluation with {len(test_cases)} test cases")

    # Run evaluation
    results = run_evaluation(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        test_cases=test_cases,
    )

    # Print results
    print_results(results)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
