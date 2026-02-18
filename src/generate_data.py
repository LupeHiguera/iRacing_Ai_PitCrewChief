"""
Synthetic training data generator for race engineer fine-tuning.

Generates Alpaca-format training examples by:
1. Selecting a category (fuel_critical, tire_warning, etc.)
2. Selecting a random car and track from metadata
3. Generating synthetic telemetry appropriate for the category
4. Calling Claude API to generate a race engineer response
5. Validating and saving the result
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

from . import metadata


# =============================================================================
# CAR/TRACK COMPATIBILITY
# =============================================================================

ROAD_COURSE_TYPES = {"high_speed", "mixed", "technical", "street"}
OVAL_TYPES = {"superspeedway", "intermediate", "short_track"}

# Car classes that can ONLY race on road courses
ROAD_ONLY_CLASSES = {
    "production", "TCX", "touring", "TCR", "supercars",
    "GT4", "GT3", "cup", "GTE", "GTP", "hypercar",
    "LMP2", "LMP3", "prototype",
    "F1", "super_formula", "F3", "F4",
    "historic_f1", "rallycross",
}

# IndyCar races everywhere
ALL_TRACK_CLASSES = {"IndyCar"}

# Specific open_wheel car keys that are Indy ladder (can do road + some ovals)
INDY_LADDER_KEYS = {"indy_pro_2000", "dallara_il15", "usf2000", "pm18"}


def is_compatible(car_key: str, car_class: str, track_type: str) -> bool:
    """Check if a car class is compatible with a track type."""
    if car_class in ALL_TRACK_CLASSES:
        return True

    if car_class in ROAD_ONLY_CLASSES:
        return track_type in ROAD_COURSE_TYPES

    # open_wheel: check if it's Indy ladder
    if car_class == "open_wheel":
        if car_key in INDY_LADDER_KEYS:
            # Indy ladder can do road courses + short ovals
            return track_type in (ROAD_COURSE_TYPES | {"short_track"})
        else:
            return track_type in ROAD_COURSE_TYPES

    # Default: road course only
    return track_type in ROAD_COURSE_TYPES


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for data generation."""
    output_dir: Path = Path("data/synthetic")
    examples_per_batch: int = 100
    total_examples: int = 10000
    model: str = "claude-sonnet-4-5-20250929"
    max_retries: int = 3
    retry_delay: float = 1.0
    validate_responses: bool = True
    append_file: Optional[str] = None  # Path to existing data to append to


# Category distribution - what percentage of examples for each situation
CATEGORIES = {
    "fuel_critical": 0.10,      # 10% - box now situations
    "fuel_warning": 0.12,       # 12% - fuel getting low
    "tire_critical": 0.10,      # 10% - tires gone
    "tire_warning": 0.12,       # 12% - tires wearing/overheating
    "tire_cold": 0.08,          # 8%  - tires not up to temp
    "position_battle": 0.12,    # 12% - defending/attacking
    "gap_management": 0.10,     # 10% - dirty air, clean air
    "pit_approach": 0.08,       # 8%  - coming into pits
    "pace_feedback": 0.08,      # 8%  - improving/dropping pace
    "routine": 0.10,            # 10% - everything is fine updates
}


# =============================================================================
# TELEMETRY GENERATORS
# =============================================================================

def generate_base_telemetry(track: Dict[str, Any]) -> Dict[str, Any]:
    """Generate base telemetry values that will be modified by category."""
    # Pick a random corner position
    corners = list(track["key_corners"].keys())
    lap_pct = random.choice(corners) if corners else random.uniform(0.1, 0.9)

    # Base lap time varies by track type
    base_lap_time = {
        "high_speed": random.uniform(85, 110),
        "mixed": random.uniform(95, 130),
        "technical": random.uniform(100, 140),
        "street": random.uniform(100, 130),
        "superspeedway": random.uniform(45, 55),
        "intermediate": random.uniform(28, 35),
        "short_track": random.uniform(15, 25),
    }.get(track["type"], random.uniform(90, 120))

    return {
        "lap": random.randint(5, 40),
        "lap_pct": round(lap_pct, 2),
        "position": random.randint(1, 20),
        "fuel_laps_remaining": round(random.uniform(8, 25), 1),
        "tire_wear": {
            "fl": random.randint(5, 30),
            "fr": random.randint(5, 30),
            "rl": random.randint(5, 25),
            "rr": random.randint(5, 25),
        },
        "tire_temps": {
            "fl": random.randint(80, 100),
            "fr": random.randint(80, 100),
            "rl": random.randint(78, 95),
            "rr": random.randint(78, 95),
        },
        "gap_ahead": round(random.uniform(1.5, 8.0), 1),
        "gap_behind": round(random.uniform(1.5, 8.0), 1),
        "last_lap_time": round(base_lap_time + random.uniform(-1, 2), 1),
        "best_lap_time": round(base_lap_time, 1),
        "session_laps_remain": random.randint(10, 35),
        "incident_count": random.choice([0, 0, 0, 1, 2, 4]),
        "track_temp_c": random.randint(22, 42),
    }


def generate_fuel_critical(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for fuel critical situation."""
    base["fuel_laps_remaining"] = round(random.uniform(0.5, 2.0), 1)
    base["session_laps_remain"] = random.randint(3, 10)
    return base


def generate_fuel_warning(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for fuel warning situation."""
    base["fuel_laps_remaining"] = round(random.uniform(2.5, 5.0), 1)
    base["session_laps_remain"] = random.randint(6, 15)
    return base


def generate_tire_critical(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for tire critical situation."""
    # One or more tires heavily worn
    worst_corner = random.choice(["fl", "fr", "rl", "rr"])
    base["tire_wear"][worst_corner] = random.randint(75, 95)
    # Maybe a second tire also bad
    if random.random() > 0.5:
        other = random.choice([c for c in ["fl", "fr", "rl", "rr"] if c != worst_corner])
        base["tire_wear"][other] = random.randint(60, 80)
    # Hot temps on worn tires
    base["tire_temps"][worst_corner] = random.randint(108, 125)
    return base


def generate_tire_warning(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for tire warning - overheating or wearing."""
    # Either overheating or wearing, pick one
    if random.random() > 0.5:
        # Overheating
        hot_corner = random.choice(["fl", "fr", "rl", "rr"])
        base["tire_temps"][hot_corner] = random.randint(102, 115)
    else:
        # Wearing
        worn_corner = random.choice(["fl", "fr", "rl", "rr"])
        base["tire_wear"][worn_corner] = random.randint(45, 70)
    return base


def generate_tire_cold(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for cold tires - early in stint or after pit."""
    base["lap"] = random.randint(1, 3)
    # Cold temps
    for corner in ["fl", "fr", "rl", "rr"]:
        base["tire_temps"][corner] = random.randint(45, 65)
    # Fresh tires
    for corner in ["fl", "fr", "rl", "rr"]:
        base["tire_wear"][corner] = random.randint(0, 8)
    return base


def generate_position_battle(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for position battle - close gaps."""
    if random.random() > 0.5:
        # Defending
        base["gap_behind"] = round(random.uniform(0.3, 1.2), 1)
        base["gap_ahead"] = round(random.uniform(2.0, 6.0), 1)
    else:
        # Attacking
        base["gap_ahead"] = round(random.uniform(0.4, 1.5), 1)
        base["gap_behind"] = round(random.uniform(2.0, 6.0), 1)
    return base


def generate_gap_management(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for gap management - dirty air or clean air."""
    if random.random() > 0.5:
        # Dirty air - stuck behind someone
        base["gap_ahead"] = round(random.uniform(0.8, 1.8), 1)
        base["gap_behind"] = round(random.uniform(3.0, 8.0), 1)
        # Slightly elevated front temps from dirty air
        base["tire_temps"]["fl"] += random.randint(3, 8)
        base["tire_temps"]["fr"] += random.randint(3, 8)
    else:
        # Clean air - good gap
        base["gap_ahead"] = round(random.uniform(3.0, 8.0), 1)
        base["gap_behind"] = round(random.uniform(2.5, 6.0), 1)
    return base


def generate_pit_approach(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for approaching pit stop."""
    base["lap_pct"] = round(random.uniform(0.85, 0.98), 2)
    # Could be fuel or tire driven
    if random.random() > 0.5:
        base["fuel_laps_remaining"] = round(random.uniform(1.0, 3.0), 1)
    else:
        worn = random.choice(["fl", "fr", "rl", "rr"])
        base["tire_wear"][worn] = random.randint(55, 80)
    return base


def generate_pace_feedback(base: Dict[str, Any]) -> Dict[str, Any]:
    """Modify telemetry for pace feedback - improving or dropping."""
    if random.random() > 0.5:
        # Pace improving - last lap faster than best
        base["last_lap_time"] = base["best_lap_time"] - random.uniform(0.1, 0.5)
    else:
        # Pace dropping
        base["last_lap_time"] = base["best_lap_time"] + random.uniform(0.5, 1.5)
        # Maybe tires going off
        if random.random() > 0.5:
            base["tire_wear"]["fr"] = random.randint(35, 55)
    base["last_lap_time"] = round(base["last_lap_time"], 1)
    return base


def generate_routine(base: Dict[str, Any]) -> Dict[str, Any]:
    """Routine update - everything is fine."""
    # Keep base values, maybe small variations
    base["fuel_laps_remaining"] = round(random.uniform(10, 20), 1)
    # Comfortable gaps
    base["gap_ahead"] = round(random.uniform(2.0, 5.0), 1)
    base["gap_behind"] = round(random.uniform(2.0, 5.0), 1)
    return base


CATEGORY_GENERATORS = {
    "fuel_critical": generate_fuel_critical,
    "fuel_warning": generate_fuel_warning,
    "tire_critical": generate_tire_critical,
    "tire_warning": generate_tire_warning,
    "tire_cold": generate_tire_cold,
    "position_battle": generate_position_battle,
    "gap_management": generate_gap_management,
    "pit_approach": generate_pit_approach,
    "pace_feedback": generate_pace_feedback,
    "routine": generate_routine,
}


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def get_upcoming_corner(track: Dict[str, Any], lap_pct: float) -> Optional[str]:
    """Get the next corner based on lap percentage."""
    corners = track.get("key_corners", {})
    if not corners:
        return None

    # Find the next corner after current lap_pct
    for pct, name in sorted(corners.items()):
        if pct > lap_pct:
            return name

    # Wrap around to first corner
    if corners:
        return list(sorted(corners.items()))[0][1]
    return None


def build_claude_prompt(
    category: str,
    car: Dict[str, Any],
    track: Dict[str, Any],
    telemetry: Dict[str, Any],
) -> str:
    """Build the prompt for Claude to generate a race engineer response."""

    upcoming_corner = get_upcoming_corner(track, telemetry["lap_pct"])
    corner_info = f"(approaching {upcoming_corner})" if upcoming_corner else ""

    # Describe the situation based on category
    situation_hints = {
        "fuel_critical": "CRITICAL fuel situation - must pit immediately",
        "fuel_warning": "fuel is getting low, need to plan pit stop",
        "tire_critical": "tires are critically worn or overheating",
        "tire_warning": "tire temps elevated or wear increasing",
        "tire_cold": "tires are cold, need to build temperature",
        "position_battle": "in close battle with another car",
        "gap_management": "managing gap to cars ahead/behind",
        "pit_approach": "approaching pit entry",
        "pace_feedback": "pace has changed from previous laps",
        "routine": "normal racing conditions, general update",
    }

    prompt = f"""Generate a race engineer radio callout for this situation:

**Car:** {car["name"]} ({car["class"]} class)
- Traits: {", ".join(car["traits"])}
- Advice style: {car["advice_style"]}

**Track:** {track["name"]} ({track["type"]})
- Characteristics: {track["characteristics"]}

**Situation:** {situation_hints.get(category, "racing")}

**Telemetry:**
- Lap {telemetry["lap"]}, Position P{telemetry["position"]}
- Lap %: {telemetry["lap_pct"]} {corner_info}
- Fuel: {telemetry["fuel_laps_remaining"]} laps remaining
- Tire wear: FL {telemetry["tire_wear"]["fl"]}%, FR {telemetry["tire_wear"]["fr"]}%, RL {telemetry["tire_wear"]["rl"]}%, RR {telemetry["tire_wear"]["rr"]}%
- Tire temps: FL {telemetry["tire_temps"]["fl"]}°C, FR {telemetry["tire_temps"]["fr"]}°C, RL {telemetry["tire_temps"]["rl"]}°C, RR {telemetry["tire_temps"]["rr"]}°C
- Gap ahead: {telemetry["gap_ahead"]}s, Gap behind: {telemetry["gap_behind"]}s
- Last lap: {telemetry["last_lap_time"]}s, Best: {telemetry["best_lap_time"]}s
- Laps remaining in session: {telemetry["session_laps_remain"]}
- Incidents: {telemetry["incident_count"]}x
- Track temp: {telemetry["track_temp_c"]}°C

**Rules:**
1. Under 25 words - this will be spoken via TTS
2. Reference the corner by name if relevant to the advice
3. Use car-appropriate advice (match the car's traits and advice style)
4. NO fake driver names (no "Hamilton", "Max", etc.)
5. NO references to systems the car doesn't have (no DRS in GT3, no KERS, etc.)
6. Ground your response in the actual telemetry values provided
7. Be direct and actionable - this is a radio callout, not a conversation

Respond with ONLY the race engineer callout, nothing else."""

    return prompt


def build_training_input(
    car: Dict[str, Any],
    track: Dict[str, Any],
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the input dict for the training example (what the model will see)."""
    return {
        "car": car["name"],
        "car_class": car["class"],
        "car_traits": car["traits"],
        "track": track["name"],
        "track_type": track["type"],
        "lap": telemetry["lap"],
        "lap_pct": telemetry["lap_pct"],
        "position": telemetry["position"],
        "fuel_laps_remaining": telemetry["fuel_laps_remaining"],
        "tire_wear": telemetry["tire_wear"],
        "tire_temps": telemetry["tire_temps"],
        "gap_ahead": telemetry["gap_ahead"],
        "gap_behind": telemetry["gap_behind"],
        "last_lap_time": telemetry["last_lap_time"],
        "best_lap_time": telemetry["best_lap_time"],
        "session_laps_remain": telemetry["session_laps_remain"],
        "incident_count": telemetry["incident_count"],
        "track_temp_c": telemetry["track_temp_c"],
    }


# =============================================================================
# RESPONSE VALIDATION
# =============================================================================

def validate_response(response: str, car: Dict[str, Any]) -> tuple[bool, str]:
    """Validate that the response meets quality rules."""

    # Check length
    word_count = len(response.split())
    if word_count > 30:
        return False, f"Too long: {word_count} words (max 25)"

    # Check for hallucinated driver names
    fake_names = ["hamilton", "verstappen", "leclerc", "max", "lewis", "charles",
                  "bogdan", "driver", "mate", "buddy"]
    response_lower = response.lower()
    for name in fake_names:
        if name in response_lower:
            return False, f"Contains fake name: {name}"

    # Check for systems the car doesn't have
    car_class = car["class"].lower()

    # DRS only in F1
    if "drs" in response_lower and car_class != "f1":
        return False, "References DRS but car is not F1"

    # KERS/ERS only in hybrids
    hybrid_classes = ["f1", "lmdh", "hypercar", "gtp"]
    if ("kers" in response_lower or "ers" in response_lower) and car_class not in hybrid_classes:
        return False, "References KERS/ERS but car is not hybrid"

    # Push to pass only in IndyCar
    if "push to pass" in response_lower and "indycar" not in car_class:
        return False, "References push-to-pass but car is not IndyCar"

    return True, "OK"


# =============================================================================
# DATA GENERATOR CLASS
# =============================================================================

class DataGenerator:
    """Generates synthetic training data using Claude API."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.client = anthropic.Anthropic()
        self.cars = list(metadata.CARS.items())
        self.tracks = list(metadata.TRACKS.items())

        # Pre-compute compatible tracks for each car
        self._compatible_tracks = {}
        for car_key, car in self.cars:
            compatible = [
                (tk, t) for tk, t in self.tracks
                if is_compatible(car_key, car["class"], t["type"])
            ]
            self._compatible_tracks[car_key] = compatible

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.generated = 0
        self.failed = 0
        self.validation_failures = 0

    def select_category(self) -> str:
        """Select a category based on distribution."""
        rand = random.random()
        cumulative = 0.0
        for category, prob in CATEGORIES.items():
            cumulative += prob
            if rand <= cumulative:
                return category
        return "routine"  # Fallback

    def generate_example(self) -> Optional[Dict[str, Any]]:
        """Generate a single training example."""

        # Select category, car, then a compatible track
        category = self.select_category()
        car_key, car = random.choice(self.cars)
        compatible = self._compatible_tracks[car_key]
        if not compatible:
            return None
        track_key, track = random.choice(compatible)

        # Generate telemetry
        base_telemetry = generate_base_telemetry(track)
        generator = CATEGORY_GENERATORS.get(category, generate_routine)
        telemetry = generator(base_telemetry)

        # Build prompt for Claude
        prompt = build_claude_prompt(category, car, track, telemetry)

        # Call Claude API
        for attempt in range(self.config.max_retries):
            try:
                message = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=100,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response = message.content[0].text.strip()
                break
            except Exception as e:
                print(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    self.failed += 1
                    return None

        # Validate response
        if self.config.validate_responses:
            is_valid, reason = validate_response(response, car)
            if not is_valid:
                print(f"Validation failed: {reason}")
                print(f"  Response: {response[:100]}...")
                self.validation_failures += 1
                return None

        # Build training example in Alpaca format
        training_input = build_training_input(car, track, telemetry)

        example = {
            "instruction": "You are a race engineer. Given the car, track, and telemetry, provide a brief callout to the driver.",
            "input": json.dumps(training_input),
            "output": response,
            "metadata": {
                "category": category,
                "car_key": car_key,
                "track_key": track_key,
            }
        }

        self.generated += 1
        return example

    def generate_batch(self, count: int) -> List[Dict[str, Any]]:
        """Generate a batch of examples."""
        examples = []
        for i in range(count):
            example = self.generate_example()
            if example:
                examples.append(example)

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count} examples...")

            # Small delay to avoid rate limits
            time.sleep(0.1)

        return examples

    def generate_all(self) -> None:
        """Generate all training examples and save to files."""
        # Load existing data if appending
        all_examples = []
        if self.config.append_file:
            append_path = Path(self.config.append_file)
            if append_path.exists():
                with open(append_path, "r") as f:
                    all_examples = json.load(f)
                print(f"Loaded {len(all_examples)} existing examples from {append_path}")

        target = self.config.total_examples
        needed = target - len(all_examples)
        if needed <= 0:
            print(f"Already have {len(all_examples)} examples (target: {target}). Nothing to generate.")
            return

        print(f"Starting data generation: {needed} new examples (have {len(all_examples)}, target {target})")
        print(f"Output directory: {self.config.output_dir}")

        batches = (needed + self.config.examples_per_batch - 1) // self.config.examples_per_batch

        for batch_num in range(batches):
            remaining = target - len(all_examples)
            if remaining <= 0:
                break
            batch_size = min(self.config.examples_per_batch, remaining)

            print(f"\nBatch {batch_num + 1}/{batches} ({batch_size} examples)")
            examples = self.generate_batch(batch_size)
            all_examples.extend(examples)

            # Save intermediate results
            self._save_examples(all_examples, "train.json")

            print(f"Total: {len(all_examples)} examples, {self.failed} API failures, {self.validation_failures} validation failures")

        # Final save
        self._save_examples(all_examples, "train.json")

        # Save stats
        stats = {
            "total_generated": len(all_examples),
            "api_failures": self.failed,
            "validation_failures": self.validation_failures,
            "category_distribution": self._calculate_distribution(all_examples),
        }
        stats_path = self.config.output_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nDone! Generated {len(all_examples)} examples")
        print(f"Saved to {self.config.output_dir / 'train.json'}")

    def _save_examples(self, examples: List[Dict[str, Any]], filename: str) -> None:
        """Save examples to JSON file."""
        path = self.config.output_dir / filename
        with open(path, "w") as f:
            json.dump(examples, f, indent=2)

    def _calculate_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate category distribution in generated examples."""
        dist = {}
        for ex in examples:
            cat = ex.get("metadata", {}).get("category", "unknown")
            dist[cat] = dist.get(cat, 0) + 1
        return dist


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--count", type=int, default=100, help="Total number of examples (target)")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=100, help="Examples per batch")
    parser.add_argument("--no-validate", action="store_true", help="Skip response validation")
    parser.add_argument("--append", type=str, default=None, help="Path to existing data file to append to")

    args = parser.parse_args()

    config = GeneratorConfig(
        output_dir=Path(args.output),
        total_examples=args.count,
        examples_per_batch=args.batch_size,
        validate_responses=not args.no_validate,
        append_file=args.append,
    )

    generator = DataGenerator(config)
    generator.generate_all()


if __name__ == "__main__":
    main()
