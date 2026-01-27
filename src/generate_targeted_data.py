"""
Targeted training data generator - focused on corner awareness and car traits.

This generates examples that specifically emphasize:
1. Corner-specific callouts (must reference the upcoming corner by name)
2. Car-trait-specific advice (must use car-appropriate guidance)
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
import anthropic

load_dotenv()

from . import metadata


@dataclass
class TargetedConfig:
    """Configuration for targeted data generation."""
    output_dir: Path = Path("data/synthetic")
    model: str = "claude-sonnet-4-5-20250929"
    max_retries: int = 3
    retry_delay: float = 1.0


# Car traits mapped to specific advice keywords
CAR_TRAIT_ADVICE = {
    "rear_engine": [
        "trail brake into {corner}, let the rear rotate",
        "early apex {corner}, use the rear weight",
        "smooth throttle out of {corner}, respect the pendulum",
        "brake earlier into {corner}, rotation comes naturally",
    ],
    "mid_engine": [
        "balanced entry into {corner}, use the rotation",
        "carry more speed through {corner}, mid-engine balance",
        "commit to {corner}, the balance is there",
    ],
    "front_engine": [
        "early turn-in for {corner}, manage the understeer",
        "trail off the brake into {corner}, help rotation",
        "patient throttle out of {corner}, wait for the front to bite",
    ],
    "trail_brake_critical": [
        "trail brake deep into {corner}",
        "keep pressure on the brake through {corner} entry",
        "trail braking is key for {corner}",
    ],
    "momentum_car": [
        "carry speed through {corner}, keep momentum",
        "don't scrub speed at {corner}, flow through",
        "roll speed through {corner}",
    ],
    "awd": [
        "use the traction out of {corner}",
        "power down early out of {corner}, AWD will hook",
        "aggressive on power through {corner}",
    ],
    "aero_dependent": [
        "stay close through {corner}, use the aero",
        "commit to {corner}, trust the downforce",
        "flat through {corner} if possible, aero is working",
    ],
    "draft_dependent": [
        "use the tow down the straight after {corner}",
        "stay in the draft through {corner}",
    ],
    "hybrid": [
        "deploy the hybrid out of {corner}",
        "save energy through {corner}, deploy on exit",
    ],
    "high_downforce": [
        "full commit to {corner}, downforce is there",
        "trust the aero through {corner}",
    ],
    "strong_brakes": [
        "use those brakes hard into {corner}",
        "brake late into {corner}, this car stops",
    ],
    "rear_limited": [
        "easy on throttle out of {corner}, protect the rears",
        "smooth power through {corner}, rear traction is key",
    ],
    "forgiving": [
        "push through {corner}, the car will forgive small mistakes",
        "attack {corner}, plenty of grip to play with",
    ],
    "knife_edge": [
        "precision through {corner}, no margin for error",
        "respect {corner}, this car bites back",
    ],
}

# Famous corners with characteristics
CORNER_ADVICE = {
    # Monza
    "Rettifilo": "brake hard, late apex, power down the straight",
    "Curva Grande": "full throttle commitment, trust the grip",
    "Lesmo 1": "brake deep, trail in, quick direction change",
    "Lesmo 2": "double apex, patience on entry",
    "Ascari": "flick left, commit right, smooth through",
    "Parabolica": "late apex, use all the exit kerb",

    # Spa
    "La Source": "tight hairpin, cut to the apex",
    "Eau Rouge": "full send, flat if brave",
    "Raidillon": "commit or lift, no half measures",
    "Les Combes": "heavy braking, defensive line available",
    "Pouhon": "double apex, carry the speed",
    "Blanchimont": "flat through, trust the aero",
    "Bus Stop": "heavy braking, tight chicane",

    # Silverstone
    "Copse": "fast entry, trail brake to apex",
    "Maggotts-Becketts": "rhythm through the esses, flow",
    "Chapel": "full throttle exit onto straight",
    "Stowe": "heavy brake, tight apex",
    "Vale": "slow corner, traction on exit",
    "Club": "double apex, good passing spot",
    "Abbey": "fast chicane, commitment",

    # Suzuka
    "First Curve": "double apex, flow through",
    "S Curves": "rhythm is everything, stay smooth",
    "Dunlop Curve": "blind entry, commit early",
    "Degner 1": "brake hard, fast direction change",
    "Degner 2": "carry speed to hairpin",
    "Hairpin": "slowest point, traction critical",
    "Spoon Curve": "long corner, patience",
    "130R": "flat if brave, huge commitment",
    "Casio Triangle": "chicane, late brake",

    # Nurburgring GP
    "Turn 1": "heavy braking, late apex",
    "Mercedes Arena": "flow through the complex",
    "Dunlop": "fast kink, full throttle",
    "Schumacher S": "rhythm through esses",
    "Veedol": "chicane, precise inputs",

    # Bathurst
    "Hell Corner": "tight entry, power up the hill",
    "Griffins Bend": "blind crest, commitment",
    "The Cutting": "fast through, trust it",
    "Skyline": "brake over the crest, tricky",
    "The Dipper": "dive down, full commit",
    "Forrest Elbow": "tight, slow, traction limited",
    "The Chase": "fast chicane, commitment",

    # Road America
    "Turn 1": "heavy brake, carousel entry",
    "Turn 5": "kink, flat through",
    "Turn 8": "carousel, long corner",
    "Canada Corner": "fast right, use the kerb",
    "Turn 14": "hairpin, traction zone",

    # Le Mans
    "Dunlop": "chicane, fast through",
    "Esses": "flow through, rhythm",
    "Tertre Rouge": "fast exit onto Mulsanne",
    "Mulsanne": "full power, draft zone",
    "Indianapolis": "tight complex, patience",
    "Arnage": "slow hairpin, traction",
    "Porsche Curves": "fast complex, commitment",
    "Ford Chicane": "slow before line, late brake",
}


def get_corner_advice(corner_name: str) -> str:
    """Get specific advice for a corner."""
    return CORNER_ADVICE.get(corner_name, "stay focused, execute cleanly")


def get_trait_advice(traits: List[str], corner_name: str) -> str:
    """Get car-trait-specific advice for a corner."""
    for trait in traits:
        if trait in CAR_TRAIT_ADVICE:
            templates = CAR_TRAIT_ADVICE[trait]
            template = random.choice(templates)
            return template.format(corner=corner_name)
    return f"execute {corner_name} cleanly"


def build_corner_focused_prompt(
    car: Dict[str, Any],
    track: Dict[str, Any],
    telemetry: Dict[str, Any],
    corner_name: str,
    situation: str,
) -> str:
    """Build prompt that REQUIRES corner reference."""

    corner_advice = get_corner_advice(corner_name)
    trait_advice = get_trait_advice(car["traits"], corner_name)

    prompt = f"""Generate a race engineer radio callout that MUST reference the corner by name.

**Car:** {car["name"]} ({car["class"]} class)
- Traits: {", ".join(car["traits"])}
- Advice style: {car["advice_style"]}

**Track:** {track["name"]}
**Corner:** {corner_name} - {corner_advice}

**Situation:** {situation}

**Telemetry:**
- Lap {telemetry["lap"]}, Position P{telemetry["position"]}
- Fuel: {telemetry["fuel_laps_remaining"]} laps
- Tire wear: FL {telemetry["tire_wear"]["fl"]}%, FR {telemetry["tire_wear"]["fr"]}%
- Tire temps: FR {telemetry["tire_temps"]["fr"]}°C
- Gap behind: {telemetry["gap_behind"]}s

**CRITICAL REQUIREMENTS:**
1. MUST mention "{corner_name}" by name in the response
2. Under 25 words
3. Reference car trait: {trait_advice}
4. Be direct and actionable

Respond with ONLY the race engineer callout, nothing else."""

    return prompt


def build_trait_focused_prompt(
    car: Dict[str, Any],
    track: Dict[str, Any],
    telemetry: Dict[str, Any],
    corner_name: str,
    primary_trait: str,
) -> str:
    """Build prompt that REQUIRES car-trait-specific advice."""

    trait_templates = CAR_TRAIT_ADVICE.get(primary_trait, [])
    trait_hint = random.choice(trait_templates).format(corner=corner_name) if trait_templates else ""

    prompt = f"""Generate a race engineer radio callout that uses CAR-SPECIFIC advice.

**Car:** {car["name"]} ({car["class"]} class)
- KEY TRAIT: {primary_trait}
- All traits: {", ".join(car["traits"])}
- Advice style: {car["advice_style"]}

**Track:** {track["name"]}
**Corner:** {corner_name}

**Telemetry:**
- Lap {telemetry["lap"]}, Position P{telemetry["position"]}
- Fuel: {telemetry["fuel_laps_remaining"]} laps
- Tire temps: FR {telemetry["tire_temps"]["fr"]}°C
- Gap ahead: {telemetry["gap_ahead"]}s, Gap behind: {telemetry["gap_behind"]}s

**CRITICAL REQUIREMENTS:**
1. Advice MUST be specific to the "{primary_trait}" characteristic
2. Example of trait-specific advice: "{trait_hint}"
3. Under 25 words
4. Reference {corner_name} if natural
5. Be direct and actionable

DO NOT give generic advice. The callout must show understanding of how this specific car handles.

Respond with ONLY the race engineer callout, nothing else."""

    return prompt


class TargetedDataGenerator:
    """Generates targeted training data for corners and car traits."""

    def __init__(self, config: Optional[TargetedConfig] = None):
        self.config = config or TargetedConfig()
        self.client = anthropic.Anthropic()
        self.cars = list(metadata.CARS.items())
        self.tracks = list(metadata.TRACKS.items())

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.generated = 0
        self.failed = 0

    def generate_corner_example(self) -> Optional[Dict[str, Any]]:
        """Generate a corner-focused example."""

        # Pick car and track
        car_key, car = random.choice(self.cars)
        track_key, track = random.choice(self.tracks)

        # Pick a corner from the track
        corners = track.get("key_corners", {})
        if not corners:
            return None

        lap_pct = random.choice(list(corners.keys()))
        corner_name = corners[lap_pct]

        # Generate telemetry
        telemetry = {
            "lap": random.randint(5, 30),
            "lap_pct": lap_pct,
            "position": random.randint(1, 15),
            "fuel_laps_remaining": round(random.uniform(5, 20), 1),
            "tire_wear": {
                "fl": random.randint(10, 45),
                "fr": random.randint(10, 50),
                "rl": random.randint(8, 40),
                "rr": random.randint(8, 40),
            },
            "tire_temps": {
                "fl": random.randint(82, 105),
                "fr": random.randint(82, 108),
                "rl": random.randint(80, 98),
                "rr": random.randint(80, 98),
            },
            "gap_ahead": round(random.uniform(1.0, 5.0), 1),
            "gap_behind": round(random.uniform(0.5, 4.0), 1),
            "last_lap_time": round(random.uniform(90, 140), 1),
            "best_lap_time": round(random.uniform(88, 138), 1),
            "session_laps_remain": random.randint(8, 25),
            "incident_count": random.choice([0, 0, 1, 2]),
            "track_temp_c": random.randint(24, 38),
        }

        # Pick a situation
        situations = [
            "approaching key corner, give guidance",
            "defending position, corner advice needed",
            "attacking position, corner execution is key",
            "tire management through this corner",
            "general corner advice for rhythm",
        ]
        situation = random.choice(situations)

        # Build prompt
        prompt = build_corner_focused_prompt(car, track, telemetry, corner_name, situation)

        # Call Claude
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text.strip()
        except Exception as e:
            print(f"API error: {e}")
            self.failed += 1
            return None

        # Validate corner is mentioned
        if corner_name.lower() not in response.lower():
            # Try to extract any corner mention
            found_corner = False
            for c in corners.values():
                if c.lower() in response.lower():
                    found_corner = True
                    break
            if not found_corner:
                print(f"Corner not mentioned: {corner_name} in '{response[:50]}...'")
                return None

        # Build training example
        training_input = {
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

        self.generated += 1
        return {
            "instruction": "You are a race engineer. Given the car, track, and telemetry, provide a brief callout to the driver.",
            "input": json.dumps(training_input),
            "output": response,
            "metadata": {
                "category": "corner_specific",
                "car_key": car_key,
                "track_key": track_key,
                "corner": corner_name,
            }
        }

    def generate_trait_example(self) -> Optional[Dict[str, Any]]:
        """Generate a car-trait-focused example."""

        # Pick car with interesting traits
        car_key, car = random.choice(self.cars)

        # Pick a trait that has advice templates
        available_traits = [t for t in car["traits"] if t in CAR_TRAIT_ADVICE]
        if not available_traits:
            return None

        primary_trait = random.choice(available_traits)

        # Pick track and corner
        track_key, track = random.choice(self.tracks)
        corners = track.get("key_corners", {})
        if not corners:
            return None

        lap_pct = random.choice(list(corners.keys()))
        corner_name = corners[lap_pct]

        # Generate telemetry
        telemetry = {
            "lap": random.randint(5, 30),
            "lap_pct": lap_pct,
            "position": random.randint(1, 15),
            "fuel_laps_remaining": round(random.uniform(5, 20), 1),
            "tire_wear": {
                "fl": random.randint(10, 45),
                "fr": random.randint(10, 50),
                "rl": random.randint(8, 40),
                "rr": random.randint(8, 40),
            },
            "tire_temps": {
                "fl": random.randint(82, 105),
                "fr": random.randint(82, 108),
                "rl": random.randint(80, 98),
                "rr": random.randint(80, 98),
            },
            "gap_ahead": round(random.uniform(1.0, 5.0), 1),
            "gap_behind": round(random.uniform(0.5, 4.0), 1),
            "last_lap_time": round(random.uniform(90, 140), 1),
            "best_lap_time": round(random.uniform(88, 138), 1),
            "session_laps_remain": random.randint(8, 25),
            "incident_count": random.choice([0, 0, 1, 2]),
            "track_temp_c": random.randint(24, 38),
        }

        # Build prompt
        prompt = build_trait_focused_prompt(car, track, telemetry, corner_name, primary_trait)

        # Call Claude
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text.strip()
        except Exception as e:
            print(f"API error: {e}")
            self.failed += 1
            return None

        # Build training example
        training_input = {
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

        self.generated += 1
        return {
            "instruction": "You are a race engineer. Given the car, track, and telemetry, provide a brief callout to the driver.",
            "input": json.dumps(training_input),
            "output": response,
            "metadata": {
                "category": "car_trait_specific",
                "car_key": car_key,
                "track_key": track_key,
                "primary_trait": primary_trait,
            }
        }

    def generate_targeted(self, corner_count: int, trait_count: int) -> List[Dict[str, Any]]:
        """Generate targeted examples."""
        examples = []

        print(f"Generating {corner_count} corner-specific examples...")
        for i in range(corner_count):
            example = self.generate_corner_example()
            if example:
                examples.append(example)
            if (i + 1) % 10 == 0:
                print(f"  Corner examples: {i+1}/{corner_count}")
            time.sleep(0.15)  # Rate limit

        print(f"Generating {trait_count} car-trait-specific examples...")
        for i in range(trait_count):
            example = self.generate_trait_example()
            if example:
                examples.append(example)
            if (i + 1) % 10 == 0:
                print(f"  Trait examples: {i+1}/{trait_count}")
            time.sleep(0.15)  # Rate limit

        return examples


def main():
    """Generate targeted data and merge with existing."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate targeted training data")
    parser.add_argument("--corners", type=int, default=250, help="Corner-specific examples")
    parser.add_argument("--traits", type=int, default=250, help="Car-trait-specific examples")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Merge with existing train.json")

    args = parser.parse_args()

    config = TargetedConfig(output_dir=Path(args.output))
    generator = TargetedDataGenerator(config)

    # Generate targeted examples
    new_examples = generator.generate_targeted(args.corners, args.traits)

    print(f"\nGenerated {len(new_examples)} new examples")
    print(f"  Failed: {generator.failed}")

    # Load existing if merging
    output_path = Path(args.output) / "train.json"
    if args.merge and output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        print(f"Merging with {len(existing)} existing examples")
        all_examples = existing + new_examples
    else:
        all_examples = new_examples

    # Save
    with open(output_path, "w") as f:
        json.dump(all_examples, f, indent=2)

    print(f"Saved {len(all_examples)} total examples to {output_path}")

    # Category distribution
    dist = {}
    for ex in all_examples:
        cat = ex.get("metadata", {}).get("category", "unknown")
        dist[cat] = dist.get(cat, 0) + 1
    print(f"Category distribution: {dist}")


if __name__ == "__main__":
    main()
