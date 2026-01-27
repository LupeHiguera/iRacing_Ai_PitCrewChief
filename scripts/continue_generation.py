"""
Continue generating examples and merge with existing data.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate_data import DataGenerator, GeneratorConfig

def main():
    # Load existing general examples
    general_path = Path("data/synthetic/train.json")
    with open(general_path) as f:
        general_examples = json.load(f)
    print(f"Loaded {len(general_examples)} existing general examples")

    # Load targeted examples
    targeted_path = Path("data/synthetic/train_targeted_549.json")
    with open(targeted_path) as f:
        targeted_examples = json.load(f)
    print(f"Loaded {len(targeted_examples)} targeted examples")

    # Calculate how many more general examples needed to reach 1500
    target_general = 1500
    remaining = target_general - len(general_examples)

    if remaining > 0:
        print(f"\nGenerating {remaining} more general examples...")

        config = GeneratorConfig(
            output_dir=Path("data/synthetic"),
            total_examples=remaining,
            examples_per_batch=50,
            validate_responses=True,
        )

        generator = DataGenerator(config)

        # Generate remaining examples
        new_examples = generator.generate_batch(remaining)
        print(f"Generated {len(new_examples)} new examples")

        # Append to general examples
        general_examples.extend(new_examples)

        # Save updated general examples
        with open(general_path, "w", encoding="utf-8") as f:
            json.dump(general_examples, f, indent=2)
        print(f"Saved {len(general_examples)} total general examples")
    else:
        print(f"Already have {len(general_examples)} general examples (target: {target_general})")

    # Merge all examples
    all_examples = general_examples + targeted_examples
    print(f"\nTotal combined: {len(all_examples)} examples")

    # Save merged training data
    merged_path = Path("data/synthetic/train_merged.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2)
    print(f"Saved merged data to {merged_path}")

    # Calculate category distribution
    dist = {}
    for ex in all_examples:
        cat = ex.get("metadata", {}).get("category", "unknown")
        dist[cat] = dist.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()