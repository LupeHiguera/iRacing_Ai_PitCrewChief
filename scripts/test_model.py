"""
Test the fine-tuned race engineer model.

Loads the LoRA adapter and runs inference on test cases.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --adapter models/race-engineer-lora
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# Test cases covering different scenarios
TEST_CASES = [
    {
        "name": "Fuel Critical",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 15,
            "lap_pct": 0.85,
            "position": 6,
            "fuel_laps_remaining": 1.2,
            "tire_wear": {"fl": 25, "fr": 30, "rl": 20, "rr": 22},
            "tire_temps": {"fl": 92, "fr": 95, "rl": 88, "rr": 90},
            "gap_ahead": 3.5,
            "gap_behind": 2.1,
            "last_lap_time": 108.5,
            "best_lap_time": 107.8,
            "session_laps_remain": 8,
            "incident_count": 0,
            "track_temp_c": 32,
        },
    },
    {
        "name": "Defending Position",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Spa",
            "track_type": "mixed",
            "lap": 8,
            "lap_pct": 0.12,
            "position": 4,
            "fuel_laps_remaining": 18.5,
            "tire_wear": {"fl": 12, "fr": 15, "rl": 10, "rr": 12},
            "tire_temps": {"fl": 88, "fr": 90, "rl": 85, "rr": 87},
            "gap_ahead": 4.2,
            "gap_behind": 0.6,
            "last_lap_time": 138.2,
            "best_lap_time": 137.5,
            "session_laps_remain": 22,
            "incident_count": 1,
            "track_temp_c": 28,
        },
    },
    {
        "name": "Tire Overheating",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 12,
            "lap_pct": 0.72,
            "position": 8,
            "fuel_laps_remaining": 14.0,
            "tire_wear": {"fl": 35, "fr": 42, "rl": 28, "rr": 30},
            "tire_temps": {"fl": 98, "fr": 112, "rl": 90, "rr": 92},
            "gap_ahead": 1.8,
            "gap_behind": 3.2,
            "last_lap_time": 118.5,
            "best_lap_time": 117.2,
            "session_laps_remain": 18,
            "incident_count": 0,
            "track_temp_c": 38,
        },
    },
    {
        "name": "Clean Air Push",
        "input": {
            "car": "Mazda MX-5 Cup",
            "car_class": "production",
            "car_traits": ["momentum_car", "draft_dependent", "forgiving"],
            "track": "Laguna Seca",
            "track_type": "technical",
            "lap": 6,
            "lap_pct": 0.55,
            "position": 3,
            "fuel_laps_remaining": 22.0,
            "tire_wear": {"fl": 8, "fr": 10, "rl": 6, "rr": 8},
            "tire_temps": {"fl": 82, "fr": 85, "rl": 78, "rr": 80},
            "gap_ahead": 5.5,
            "gap_behind": 4.8,
            "last_lap_time": 98.2,
            "best_lap_time": 97.8,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 30,
        },
    },
]


def load_model(adapter_path: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {adapter_path}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, input_data: dict, max_new_tokens: int = 100) -> str:
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
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

    return response.strip()


def run_tests(model, tokenizer):
    """Run all test cases and display results."""
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test in TEST_CASES:
        print(f"\n--- {test['name']} ---")
        print(f"Car: {test['input']['car']} at {test['input']['track']}")
        print(f"Position: P{test['input']['position']}, Lap {test['input']['lap']}")
        print(f"Fuel: {test['input']['fuel_laps_remaining']} laps")
        print(f"Gap behind: {test['input']['gap_behind']}s")
        print(f"FR tire: {test['input']['tire_temps']['fr']}°C, {test['input']['tire_wear']['fr']}% worn")

        response = generate_response(model, tokenizer, test["input"])
        print(f"\nResponse: {response}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned race engineer model")
    parser.add_argument("--adapter", type=str, default="models/race-engineer-lora",
                        help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name")

    args = parser.parse_args()

    if not Path(args.adapter).exists():
        print(f"Error: Adapter not found at {args.adapter}")
        print("Run fine-tuning first: python scripts/finetune.py")
        return

    model, tokenizer = load_model(args.adapter, args.base_model)
    run_tests(model, tokenizer)


if __name__ == "__main__":
    main()
