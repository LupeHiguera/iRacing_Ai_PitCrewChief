"""
Clean existing training data by removing unrealistic car/track combinations.
Also removes examples with hallucinated driver names.

Usage:
    python scripts/clean_data.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generate_data import is_compatible

HALLUCINATED_NAMES = [
    "hamilton", "verstappen", "leclerc", "max ", " max,", "lewis",
    "charles", "bogdan", "schumacher", "senna", "prost",
]


def clean_example(example: dict) -> bool:
    """Return True if example should be KEPT."""
    # Parse input
    try:
        inp = json.loads(example["input"]) if isinstance(example["input"], str) else example["input"]
    except (json.JSONDecodeError, KeyError):
        return False

    car_class = inp.get("car_class", "")
    track_type = inp.get("track_type", "")
    car_key = example.get("metadata", {}).get("car_key", "")

    # Check car/track compatibility
    if not is_compatible(car_key, car_class, track_type):
        return False

    # Check for hallucinated names in output
    output_lower = example.get("output", "").lower()
    for name in HALLUCINATED_NAMES:
        if name in output_lower:
            return False

    return True


def main():
    input_path = Path("data/synthetic/train_merged.json")
    output_path = Path("data/synthetic/train_clean.json")

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {input_path}")

    # Clean
    clean = [ex for ex in data if clean_example(ex)]
    removed = len(data) - len(clean)

    print(f"Removed {removed} examples ({removed/len(data)*100:.1f}%)")
    print(f"Remaining: {len(clean)} clean examples")

    # Category breakdown of cleaned data
    cats = {}
    for ex in clean:
        cat = ex.get("metadata", {}).get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print("\nCategory distribution after cleaning:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count} ({count/len(clean)*100:.1f}%)")

    # Save
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved clean data to {output_path}")


if __name__ == "__main__":
    main()
