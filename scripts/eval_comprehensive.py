"""
Comprehensive Evaluation Framework for Race Engineer Model.

Compares base Llama 3.1 8B vs fine-tuned model (1500 examples) on 50+ test cases.

Features:
- 50+ diverse test cases across all 10 categories
- Enhanced metrics: urgency-appropriate response, numeric accuracy, TTS suitability
- LLM-as-judge option for qualitative scoring
- Category-specific success criteria
- Statistical analysis with win rates by category

Usage:
    python scripts/eval_comprehensive.py
    python scripts/eval_comprehensive.py --adapter models/race-engineer-llama --llm-judge
    python scripts/eval_comprehensive.py --output data/eval_comprehensive.json
"""

import json
import argparse
import re
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# =============================================================================
# COMPREHENSIVE TEST CASES (50+)
# =============================================================================

EVAL_CASES = [
    # ---------------------------------------------------------------------
    # FUEL CRITICAL (5 cases) - Must urgently tell driver to box
    # ---------------------------------------------------------------------
    {
        "name": "Fuel Critical - Imola mid-lap",
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
        "urgency": "critical",
        "must_contain_any": ["box", "pit", "fuel"],
    },
    {
        "name": "Fuel Critical - Spa approaching pit entry",
        "category": "fuel_critical",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.85,
            "position": 3,
            "fuel_laps_remaining": 0.8,
            "tire_wear": {"fl": 42, "fr": 48, "rl": 35, "rr": 38},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 86, "rr": 88},
            "gap_ahead": 4.2,
            "gap_behind": 2.1,
            "last_lap_time": 138.5,
            "best_lap_time": 137.8,
            "session_laps_remain": 15,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["box", "pit", "now"],
        "urgency": "critical",
        "must_contain_any": ["box", "pit"],
    },
    {
        "name": "Fuel Critical - Monza end of race",
        "category": "fuel_critical",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 28,
            "lap_pct": 0.35,
            "position": 7,
            "fuel_laps_remaining": 1.2,
            "tire_wear": {"fl": 52, "fr": 58, "rl": 45, "rr": 48},
            "tire_temps": {"fl": 98, "fr": 102, "rl": 92, "rr": 94},
            "gap_ahead": 3.5,
            "gap_behind": 4.8,
            "last_lap_time": 108.2,
            "best_lap_time": 107.5,
            "session_laps_remain": 4,
            "incident_count": 2,
            "track_temp_c": 36,
        },
        "expected_elements": ["box", "fuel", "critical"],
        "urgency": "critical",
        "must_contain_any": ["box", "pit", "fuel"],
    },
    {
        "name": "Fuel Critical - Le Mans prototype",
        "category": "fuel_critical",
        "input": {
            "car": "Porsche 963",
            "car_class": "LMDh",
            "car_traits": ["hybrid", "high_downforce", "complex_systems"],
            "track": "Le Mans",
            "track_type": "high_speed",
            "lap": 45,
            "lap_pct": 0.92,
            "position": 2,
            "fuel_laps_remaining": 0.5,
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
        "urgency": "critical",
        "must_contain_any": ["box", "pit"],
    },
    {
        "name": "Fuel Critical - Silverstone start of stint",
        "category": "fuel_critical",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 12,
            "lap_pct": 0.15,
            "position": 4,
            "fuel_laps_remaining": 1.0,
            "tire_wear": {"fl": 25, "fr": 28, "rl": 20, "rr": 22},
            "tire_temps": {"fl": 90, "fr": 94, "rl": 85, "rr": 88},
            "gap_ahead": 2.2,
            "gap_behind": 1.8,
            "last_lap_time": 118.5,
            "best_lap_time": 117.8,
            "session_laps_remain": 25,
            "incident_count": 1,
            "track_temp_c": 24,
        },
        "expected_elements": ["box", "fuel", "pit"],
        "urgency": "critical",
        "must_contain_any": ["box", "pit", "fuel"],
    },

    # ---------------------------------------------------------------------
    # FUEL WARNING (5 cases) - Alert about low fuel, plan pit stop
    # ---------------------------------------------------------------------
    {
        "name": "Fuel Warning - Road America 3 laps",
        "category": "fuel_warning",
        "input": {
            "car": "Audi R8 LMS GT3 Evo II",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "predictable"],
            "track": "Road America",
            "track_type": "high_speed",
            "lap": 15,
            "lap_pct": 0.55,
            "position": 6,
            "fuel_laps_remaining": 3.2,
            "tire_wear": {"fl": 32, "fr": 38, "rl": 28, "rr": 30},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 86, "rr": 88},
            "gap_ahead": 2.5,
            "gap_behind": 3.8,
            "last_lap_time": 132.8,
            "best_lap_time": 131.5,
            "session_laps_remain": 18,
            "incident_count": 0,
            "track_temp_c": 30,
        },
        "expected_elements": ["fuel", "lap", "pit"],
        "urgency": "warning",
        "must_contain_any": ["fuel", "pit", "box"],
    },
    {
        "name": "Fuel Warning - Suzuka 4 laps remaining",
        "category": "fuel_warning",
        "input": {
            "car": "Mercedes-AMG GT3",
            "car_class": "GT3",
            "car_traits": ["front_engine", "stable", "strong_brakes"],
            "track": "Suzuka",
            "track_type": "mixed",
            "lap": 20,
            "lap_pct": 0.72,
            "position": 8,
            "fuel_laps_remaining": 4.0,
            "tire_wear": {"fl": 45, "fr": 50, "rl": 38, "rr": 42},
            "tire_temps": {"fl": 94, "fr": 98, "rl": 88, "rr": 90},
            "gap_ahead": 4.5,
            "gap_behind": 2.2,
            "last_lap_time": 122.5,
            "best_lap_time": 121.8,
            "session_laps_remain": 12,
            "incident_count": 1,
            "track_temp_c": 28,
        },
        "expected_elements": ["fuel", "laps", "manage"],
        "urgency": "warning",
        "must_contain_any": ["fuel", "pit", "laps"],
    },
    {
        "name": "Fuel Warning - COTA endurance",
        "category": "fuel_warning",
        "input": {
            "car": "Ferrari 488 GTE",
            "car_class": "GTE",
            "car_traits": ["mid_engine", "balanced", "high_downforce"],
            "track": "COTA",
            "track_type": "mixed",
            "lap": 35,
            "lap_pct": 0.28,
            "position": 4,
            "fuel_laps_remaining": 3.5,
            "tire_wear": {"fl": 55, "fr": 60, "rl": 48, "rr": 52},
            "tire_temps": {"fl": 96, "fr": 100, "rl": 90, "rr": 92},
            "gap_ahead": 12.0,
            "gap_behind": 8.5,
            "last_lap_time": 125.2,
            "best_lap_time": 124.5,
            "session_laps_remain": 40,
            "incident_count": 0,
            "track_temp_c": 35,
        },
        "expected_elements": ["fuel", "pit", "window"],
        "urgency": "warning",
        "must_contain_any": ["fuel", "pit"],
    },
    {
        "name": "Fuel Warning - Laguna Seca sprint",
        "category": "fuel_warning",
        "input": {
            "car": "Mazda MX-5 Cup",
            "car_class": "production",
            "car_traits": ["momentum_car", "draft_dependent", "forgiving"],
            "track": "Laguna Seca",
            "track_type": "technical",
            "lap": 12,
            "lap_pct": 0.65,
            "position": 5,
            "fuel_laps_remaining": 4.5,
            "tire_wear": {"fl": 15, "fr": 18, "rl": 12, "rr": 14},
            "tire_temps": {"fl": 82, "fr": 86, "rl": 78, "rr": 80},
            "gap_ahead": 0.8,
            "gap_behind": 1.2,
            "last_lap_time": 98.5,
            "best_lap_time": 97.8,
            "session_laps_remain": 8,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["fuel", "manage", "laps"],
        "urgency": "warning",
        "must_contain_any": ["fuel", "laps", "save"],
    },
    {
        "name": "Fuel Warning - Nurburgring long stint",
        "category": "fuel_warning",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Nurburgring GP",
            "track_type": "mixed",
            "lap": 22,
            "lap_pct": 0.42,
            "position": 3,
            "fuel_laps_remaining": 3.8,
            "tire_wear": {"fl": 48, "fr": 52, "rl": 42, "rr": 45},
            "tire_temps": {"fl": 94, "fr": 98, "rl": 88, "rr": 90},
            "gap_ahead": 1.5,
            "gap_behind": 2.8,
            "last_lap_time": 116.2,
            "best_lap_time": 115.5,
            "session_laps_remain": 14,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["fuel", "pit", "soon"],
        "urgency": "warning",
        "must_contain_any": ["fuel", "pit", "box"],
    },

    # ---------------------------------------------------------------------
    # TIRE CRITICAL (5 cases) - Tires completely worn, box needed
    # ---------------------------------------------------------------------
    {
        "name": "Tire Critical - Barcelona worn out",
        "category": "tire_critical",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Barcelona",
            "track_type": "mixed",
            "lap": 28,
            "lap_pct": 0.35,
            "position": 9,
            "fuel_laps_remaining": 8.0,
            "tire_wear": {"fl": 78, "fr": 85, "rl": 70, "rr": 75},
            "tire_temps": {"fl": 98, "fr": 105, "rl": 92, "rr": 96},
            "gap_ahead": 3.5,
            "gap_behind": 2.2,
            "last_lap_time": 103.5,
            "best_lap_time": 101.2,
            "session_laps_remain": 12,
            "incident_count": 2,
            "track_temp_c": 38,
        },
        "expected_elements": ["tire", "worn", "box", "pit"],
        "urgency": "critical",
        "must_contain_any": ["tire", "box", "pit", "worn"],
    },
    {
        "name": "Tire Critical - Monza rear gone",
        "category": "tire_critical",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 25,
            "lap_pct": 0.52,
            "position": 6,
            "fuel_laps_remaining": 12.0,
            "tire_wear": {"fl": 65, "fr": 70, "rl": 82, "rr": 88},
            "tire_temps": {"fl": 96, "fr": 100, "rl": 108, "rr": 112},
            "gap_ahead": 2.8,
            "gap_behind": 4.5,
            "last_lap_time": 109.5,
            "best_lap_time": 107.8,
            "session_laps_remain": 18,
            "incident_count": 1,
            "track_temp_c": 34,
        },
        "expected_elements": ["rear", "tire", "box"],
        "urgency": "critical",
        "must_contain_any": ["rear", "tire", "box", "pit"],
    },
    {
        "name": "Tire Critical - Spa front right destroyed",
        "category": "tire_critical",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.68,
            "position": 4,
            "fuel_laps_remaining": 6.0,
            "tire_wear": {"fl": 72, "fr": 90, "rl": 65, "rr": 68},
            "tire_temps": {"fl": 100, "fr": 115, "rl": 94, "rr": 96},
            "gap_ahead": 5.2,
            "gap_behind": 1.8,
            "last_lap_time": 140.2,
            "best_lap_time": 138.5,
            "session_laps_remain": 12,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["front", "right", "tire", "box"],
        "urgency": "critical",
        "must_contain_any": ["front", "tire", "box", "pit"],
    },
    {
        "name": "Tire Critical - Bathurst all tires gone",
        "category": "tire_critical",
        "input": {
            "car": "Mercedes-AMG GT3",
            "car_class": "GT3",
            "car_traits": ["front_engine", "stable", "strong_brakes"],
            "track": "Bathurst",
            "track_type": "mixed",
            "lap": 22,
            "lap_pct": 0.45,
            "position": 11,
            "fuel_laps_remaining": 10.0,
            "tire_wear": {"fl": 85, "fr": 88, "rl": 80, "rr": 82},
            "tire_temps": {"fl": 102, "fr": 106, "rl": 98, "rr": 100},
            "gap_ahead": 6.5,
            "gap_behind": 3.2,
            "last_lap_time": 148.5,
            "best_lap_time": 145.2,
            "session_laps_remain": 15,
            "incident_count": 3,
            "track_temp_c": 32,
        },
        "expected_elements": ["tire", "worn", "box", "pit"],
        "urgency": "critical",
        "must_contain_any": ["tire", "box", "pit"],
    },
    {
        "name": "Tire Critical - Road America long stint",
        "category": "tire_critical",
        "input": {
            "car": "Audi R8 LMS GT3 Evo II",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "predictable"],
            "track": "Road America",
            "track_type": "high_speed",
            "lap": 30,
            "lap_pct": 0.78,
            "position": 7,
            "fuel_laps_remaining": 5.0,
            "tire_wear": {"fl": 75, "fr": 82, "rl": 72, "rr": 78},
            "tire_temps": {"fl": 96, "fr": 102, "rl": 92, "rr": 96},
            "gap_ahead": 4.2,
            "gap_behind": 5.8,
            "last_lap_time": 135.2,
            "best_lap_time": 132.5,
            "session_laps_remain": 10,
            "incident_count": 1,
            "track_temp_c": 30,
        },
        "expected_elements": ["tire", "worn", "box"],
        "urgency": "critical",
        "must_contain_any": ["tire", "box", "pit", "worn"],
    },

    # ---------------------------------------------------------------------
    # TIRE WARNING (5 cases) - Tires degrading, manage them
    # ---------------------------------------------------------------------
    {
        "name": "Tire Warning - Barcelona FR hot",
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
        "urgency": "warning",
        "must_contain_any": ["tire", "front", "temp", "manage"],
    },
    {
        "name": "Tire Warning - Silverstone rear degradation",
        "category": "tire_warning",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 15,
            "lap_pct": 0.62,
            "position": 5,
            "fuel_laps_remaining": 18.0,
            "tire_wear": {"fl": 35, "fr": 38, "rl": 55, "rr": 58},
            "tire_temps": {"fl": 92, "fr": 94, "rl": 104, "rr": 106},
            "gap_ahead": 2.2,
            "gap_behind": 3.5,
            "last_lap_time": 119.2,
            "best_lap_time": 118.5,
            "session_laps_remain": 22,
            "incident_count": 0,
            "track_temp_c": 25,
        },
        "expected_elements": ["rear", "tire", "manage", "traction"],
        "urgency": "warning",
        "must_contain_any": ["rear", "tire", "manage", "traction"],
    },
    {
        "name": "Tire Warning - Suzuka wear building",
        "category": "tire_warning",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Suzuka",
            "track_type": "mixed",
            "lap": 12,
            "lap_pct": 0.45,
            "position": 4,
            "fuel_laps_remaining": 20.0,
            "tire_wear": {"fl": 42, "fr": 48, "rl": 38, "rr": 42},
            "tire_temps": {"fl": 94, "fr": 100, "rl": 90, "rr": 92},
            "gap_ahead": 1.5,
            "gap_behind": 2.8,
            "last_lap_time": 121.5,
            "best_lap_time": 120.8,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["tire", "wear", "manage", "smooth"],
        "urgency": "warning",
        "must_contain_any": ["tire", "wear", "manage"],
    },
    {
        "name": "Tire Warning - COTA uneven wear",
        "category": "tire_warning",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "COTA",
            "track_type": "mixed",
            "lap": 20,
            "lap_pct": 0.28,
            "position": 8,
            "fuel_laps_remaining": 15.0,
            "tire_wear": {"fl": 55, "fr": 42, "rl": 48, "rr": 38},
            "tire_temps": {"fl": 102, "fr": 94, "rl": 96, "rr": 90},
            "gap_ahead": 3.2,
            "gap_behind": 4.5,
            "last_lap_time": 125.8,
            "best_lap_time": 124.5,
            "session_laps_remain": 18,
            "incident_count": 1,
            "track_temp_c": 35,
        },
        "expected_elements": ["tire", "front", "left", "wear"],
        "urgency": "warning",
        "must_contain_any": ["tire", "front", "wear"],
    },
    {
        "name": "Tire Warning - Watkins Glen mid-stint",
        "category": "tire_warning",
        "input": {
            "car": "Lamborghini Huracan GT3 Evo",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "forgiving"],
            "track": "Watkins Glen",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.55,
            "position": 6,
            "fuel_laps_remaining": 14.0,
            "tire_wear": {"fl": 48, "fr": 52, "rl": 45, "rr": 48},
            "tire_temps": {"fl": 96, "fr": 102, "rl": 92, "rr": 94},
            "gap_ahead": 2.8,
            "gap_behind": 3.2,
            "last_lap_time": 109.2,
            "best_lap_time": 108.5,
            "session_laps_remain": 20,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["tire", "wear", "manage"],
        "urgency": "warning",
        "must_contain_any": ["tire", "wear", "manage"],
    },

    # ---------------------------------------------------------------------
    # TIRE COLD (5 cases) - Tires not up to temperature
    # ---------------------------------------------------------------------
    {
        "name": "Tire Cold - Bathurst out-lap",
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
        "urgency": "warning",
        "must_contain_any": ["cold", "warm", "tire", "careful"],
    },
    {
        "name": "Tire Cold - Spa morning session",
        "category": "tire_cold",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 1,
            "lap_pct": 0.45,
            "position": 8,
            "fuel_laps_remaining": 30.0,
            "tire_wear": {"fl": 1, "fr": 2, "rl": 1, "rr": 1},
            "tire_temps": {"fl": 52, "fr": 55, "rl": 45, "rr": 48},
            "gap_ahead": 2.5,
            "gap_behind": 1.5,
            "last_lap_time": 142.5,
            "best_lap_time": 142.5,
            "session_laps_remain": 40,
            "incident_count": 0,
            "track_temp_c": 14,
        },
        "expected_elements": ["cold", "tire", "rear", "careful"],
        "urgency": "warning",
        "must_contain_any": ["cold", "rear", "tire", "careful"],
    },
    {
        "name": "Tire Cold - Nurburgring pit exit",
        "category": "tire_cold",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Nurburgring GP",
            "track_type": "mixed",
            "lap": 15,
            "lap_pct": 0.12,
            "position": 6,
            "fuel_laps_remaining": 22.0,
            "tire_wear": {"fl": 0, "fr": 0, "rl": 0, "rr": 0},
            "tire_temps": {"fl": 48, "fr": 50, "rl": 42, "rr": 45},
            "gap_ahead": 4.5,
            "gap_behind": 3.2,
            "last_lap_time": 118.5,
            "best_lap_time": 116.2,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 20,
        },
        "expected_elements": ["cold", "tire", "fresh", "careful"],
        "urgency": "warning",
        "must_contain_any": ["cold", "fresh", "tire", "careful"],
    },
    {
        "name": "Tire Cold - Silverstone wet to dry",
        "category": "tire_cold",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 8,
            "lap_pct": 0.32,
            "position": 5,
            "fuel_laps_remaining": 25.0,
            "tire_wear": {"fl": 5, "fr": 6, "rl": 4, "rr": 5},
            "tire_temps": {"fl": 62, "fr": 65, "rl": 58, "rr": 60},
            "gap_ahead": 3.8,
            "gap_behind": 2.5,
            "last_lap_time": 121.5,
            "best_lap_time": 120.2,
            "session_laps_remain": 30,
            "incident_count": 0,
            "track_temp_c": 18,
        },
        "expected_elements": ["cold", "tire", "temperature", "careful"],
        "urgency": "warning",
        "must_contain_any": ["cold", "temp", "tire", "careful"],
    },
    {
        "name": "Tire Cold - Monza green track",
        "category": "tire_cold",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 3,
            "lap_pct": 0.58,
            "position": 10,
            "fuel_laps_remaining": 28.0,
            "tire_wear": {"fl": 3, "fr": 4, "rl": 2, "rr": 3},
            "tire_temps": {"fl": 68, "fr": 72, "rl": 62, "rr": 65},
            "gap_ahead": 1.8,
            "gap_behind": 0.9,
            "last_lap_time": 109.5,
            "best_lap_time": 108.8,
            "session_laps_remain": 35,
            "incident_count": 0,
            "track_temp_c": 22,
        },
        "expected_elements": ["tire", "temperature", "building", "careful"],
        "urgency": "info",
        "must_contain_any": ["tire", "temp", "cold", "careful"],
    },

    # ---------------------------------------------------------------------
    # POSITION BATTLE (5 cases) - Attacking or defending
    # ---------------------------------------------------------------------
    {
        "name": "Battle - Nurburgring attacking P2",
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
        "urgency": "info",
        "must_contain_any": ["gap", "attack", "push", "ahead"],
    },
    {
        "name": "Battle - Road America defending P2",
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
        "urgency": "warning",
        "must_contain_any": ["defend", "behind", "pressure", "close"],
    },
    {
        "name": "Battle - Monza DRS zone attack",
        "category": "position_battle",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 15,
            "lap_pct": 0.92,
            "position": 4,
            "fuel_laps_remaining": 12.0,
            "tire_wear": {"fl": 25, "fr": 28, "rl": 22, "rr": 24},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 0.6,
            "gap_behind": 2.5,
            "last_lap_time": 108.2,
            "best_lap_time": 107.5,
            "session_laps_remain": 18,
            "incident_count": 0,
            "track_temp_c": 32,
        },
        "expected_elements": ["gap", "attack", "straight", "push"],
        "urgency": "info",
        "must_contain_any": ["gap", "attack", "push", "close"],
    },
    {
        "name": "Battle - Spa Eau Rouge closing",
        "category": "position_battle",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 8,
            "lap_pct": 0.18,
            "position": 5,
            "fuel_laps_remaining": 22.0,
            "tire_wear": {"fl": 15, "fr": 18, "rl": 12, "rr": 14},
            "tire_temps": {"fl": 88, "fr": 92, "rl": 84, "rr": 86},
            "gap_ahead": 0.8,
            "gap_behind": 4.5,
            "last_lap_time": 139.5,
            "best_lap_time": 138.8,
            "session_laps_remain": 28,
            "incident_count": 0,
            "track_temp_c": 24,
        },
        "expected_elements": ["gap", "closing", "push", "attack"],
        "urgency": "info",
        "must_contain_any": ["gap", "push", "attack", "close"],
    },
    {
        "name": "Battle - Suzuka final laps pressure",
        "category": "position_battle",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Suzuka",
            "track_type": "mixed",
            "lap": 22,
            "lap_pct": 0.65,
            "position": 3,
            "fuel_laps_remaining": 5.0,
            "tire_wear": {"fl": 48, "fr": 52, "rl": 42, "rr": 45},
            "tire_temps": {"fl": 96, "fr": 100, "rl": 90, "rr": 92},
            "gap_ahead": 1.2,
            "gap_behind": 0.5,
            "last_lap_time": 122.2,
            "best_lap_time": 121.5,
            "session_laps_remain": 5,
            "incident_count": 1,
            "track_temp_c": 28,
        },
        "expected_elements": ["defend", "behind", "laps", "position"],
        "urgency": "warning",
        "must_contain_any": ["defend", "behind", "close", "pressure"],
    },

    # ---------------------------------------------------------------------
    # GAP MANAGEMENT (5 cases) - Dirty air, clean air situations
    # ---------------------------------------------------------------------
    {
        "name": "Gap - Dirty air following closely",
        "category": "gap_management",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Barcelona",
            "track_type": "mixed",
            "lap": 12,
            "lap_pct": 0.42,
            "position": 6,
            "fuel_laps_remaining": 18.0,
            "tire_wear": {"fl": 28, "fr": 32, "rl": 24, "rr": 26},
            "tire_temps": {"fl": 98, "fr": 104, "rl": 92, "rr": 94},
            "gap_ahead": 0.8,
            "gap_behind": 3.5,
            "last_lap_time": 102.8,
            "best_lap_time": 101.5,
            "session_laps_remain": 22,
            "incident_count": 0,
            "track_temp_c": 35,
        },
        "expected_elements": ["dirty", "air", "temp", "manage"],
        "urgency": "info",
        "must_contain_any": ["dirty", "air", "temp", "manage", "behind"],
    },
    {
        "name": "Gap - Clean air pushing",
        "category": "gap_management",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 10,
            "lap_pct": 0.55,
            "position": 2,
            "fuel_laps_remaining": 22.0,
            "tire_wear": {"fl": 20, "fr": 24, "rl": 18, "rr": 20},
            "tire_temps": {"fl": 90, "fr": 94, "rl": 86, "rr": 88},
            "gap_ahead": 8.5,
            "gap_behind": 4.2,
            "last_lap_time": 118.2,
            "best_lap_time": 117.8,
            "session_laps_remain": 28,
            "incident_count": 0,
            "track_temp_c": 24,
        },
        "expected_elements": ["clean", "air", "push", "gap"],
        "urgency": "info",
        "must_contain_any": ["clean", "air", "push", "gap"],
    },
    {
        "name": "Gap - Building gap to follower",
        "category": "gap_management",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Road America",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.68,
            "position": 4,
            "fuel_laps_remaining": 12.0,
            "tire_wear": {"fl": 35, "fr": 40, "rl": 30, "rr": 32},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 5.2,
            "gap_behind": 2.8,
            "last_lap_time": 133.2,
            "best_lap_time": 132.5,
            "session_laps_remain": 15,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["gap", "behind", "comfortable"],
        "urgency": "info",
        "must_contain_any": ["gap", "behind", "good", "comfortable"],
    },
    {
        "name": "Gap - Undercut window",
        "category": "gap_management",
        "input": {
            "car": "Audi R8 LMS GT3 Evo II",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "predictable"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 14,
            "lap_pct": 0.82,
            "position": 5,
            "fuel_laps_remaining": 8.0,
            "tire_wear": {"fl": 42, "fr": 48, "rl": 38, "rr": 40},
            "tire_temps": {"fl": 94, "fr": 98, "rl": 90, "rr": 92},
            "gap_ahead": 2.5,
            "gap_behind": 3.8,
            "last_lap_time": 108.8,
            "best_lap_time": 108.2,
            "session_laps_remain": 20,
            "incident_count": 0,
            "track_temp_c": 32,
        },
        "expected_elements": ["gap", "pit", "window"],
        "urgency": "info",
        "must_contain_any": ["gap", "pit", "ahead", "window"],
    },
    {
        "name": "Gap - Traffic management",
        "category": "gap_management",
        "input": {
            "car": "Mercedes-AMG GT3",
            "car_class": "GT3",
            "car_traits": ["front_engine", "stable", "strong_brakes"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 20,
            "lap_pct": 0.38,
            "position": 3,
            "fuel_laps_remaining": 10.0,
            "tire_wear": {"fl": 38, "fr": 42, "rl": 32, "rr": 35},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 4.8,
            "gap_behind": 1.5,
            "last_lap_time": 140.2,
            "best_lap_time": 139.5,
            "session_laps_remain": 12,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["gap", "behind", "closing"],
        "urgency": "info",
        "must_contain_any": ["gap", "behind", "close", "manage"],
    },

    # ---------------------------------------------------------------------
    # PIT APPROACH (5 cases) - Coming into pits
    # ---------------------------------------------------------------------
    {
        "name": "Pit Approach - Standard stop",
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
        "urgency": "critical",
        "must_contain_any": ["box", "pit"],
    },
    {
        "name": "Pit Approach - Splash and dash",
        "category": "pit_approach",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 28,
            "lap_pct": 0.88,
            "position": 4,
            "fuel_laps_remaining": 1.2,
            "tire_wear": {"fl": 32, "fr": 36, "rl": 28, "rr": 30},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 3.5,
            "gap_behind": 5.2,
            "last_lap_time": 108.5,
            "best_lap_time": 107.8,
            "session_laps_remain": 8,
            "incident_count": 0,
            "track_temp_c": 34,
        },
        "expected_elements": ["box", "pit", "fuel"],
        "urgency": "critical",
        "must_contain_any": ["box", "pit", "fuel"],
    },
    {
        "name": "Pit Approach - Tire change only",
        "category": "pit_approach",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 22,
            "lap_pct": 0.95,
            "position": 6,
            "fuel_laps_remaining": 15.0,
            "tire_wear": {"fl": 72, "fr": 78, "rl": 68, "rr": 72},
            "tire_temps": {"fl": 98, "fr": 104, "rl": 94, "rr": 96},
            "gap_ahead": 4.2,
            "gap_behind": 6.5,
            "last_lap_time": 120.2,
            "best_lap_time": 118.5,
            "session_laps_remain": 18,
            "incident_count": 1,
            "track_temp_c": 26,
        },
        "expected_elements": ["box", "pit", "tire"],
        "urgency": "warning",
        "must_contain_any": ["box", "pit", "tire"],
    },
    {
        "name": "Pit Approach - Strategic stop",
        "category": "pit_approach",
        "input": {
            "car": "Audi R8 LMS GT3 Evo II",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "predictable"],
            "track": "Suzuka",
            "track_type": "mixed",
            "lap": 18,
            "lap_pct": 0.85,
            "position": 5,
            "fuel_laps_remaining": 4.0,
            "tire_wear": {"fl": 55, "fr": 60, "rl": 50, "rr": 52},
            "tire_temps": {"fl": 94, "fr": 98, "rl": 90, "rr": 92},
            "gap_ahead": 2.8,
            "gap_behind": 8.5,
            "last_lap_time": 122.5,
            "best_lap_time": 121.8,
            "session_laps_remain": 22,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["box", "pit", "window"],
        "urgency": "warning",
        "must_contain_any": ["box", "pit"],
    },
    {
        "name": "Pit Approach - Safety car window",
        "category": "pit_approach",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Road America",
            "track_type": "high_speed",
            "lap": 15,
            "lap_pct": 0.78,
            "position": 8,
            "fuel_laps_remaining": 3.5,
            "tire_wear": {"fl": 48, "fr": 52, "rl": 42, "rr": 45},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 12.0,
            "gap_behind": 2.5,
            "last_lap_time": 134.5,
            "best_lap_time": 132.8,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 30,
        },
        "expected_elements": ["box", "pit", "fuel"],
        "urgency": "warning",
        "must_contain_any": ["box", "pit"],
    },

    # ---------------------------------------------------------------------
    # PACE FEEDBACK (5 cases) - Lap time feedback
    # ---------------------------------------------------------------------
    {
        "name": "Pace - Personal best lap",
        "category": "pace_feedback",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Suzuka",
            "track_type": "mixed",
            "lap": 8,
            "lap_pct": 0.08,
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
        "expected_elements": ["pace", "lap", "good", "best"],
        "urgency": "info",
        "must_contain_any": ["best", "good", "pace", "lap"],
    },
    {
        "name": "Pace - Slower than best",
        "category": "pace_feedback",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.05,
            "position": 6,
            "fuel_laps_remaining": 10.0,
            "tire_wear": {"fl": 42, "fr": 48, "rl": 38, "rr": 40},
            "tire_temps": {"fl": 96, "fr": 100, "rl": 90, "rr": 92},
            "gap_ahead": 3.2,
            "gap_behind": 2.5,
            "last_lap_time": 110.5,
            "best_lap_time": 107.8,
            "session_laps_remain": 15,
            "incident_count": 1,
            "track_temp_c": 34,
        },
        "expected_elements": ["pace", "lap", "time", "slower"],
        "urgency": "info",
        "must_contain_any": ["pace", "lap", "time", "down"],
    },
    {
        "name": "Pace - Consistent laps",
        "category": "pace_feedback",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Barcelona",
            "track_type": "mixed",
            "lap": 15,
            "lap_pct": 0.12,
            "position": 5,
            "fuel_laps_remaining": 14.0,
            "tire_wear": {"fl": 35, "fr": 40, "rl": 30, "rr": 32},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 2.8,
            "gap_behind": 3.5,
            "last_lap_time": 102.2,
            "best_lap_time": 101.8,
            "session_laps_remain": 20,
            "incident_count": 0,
            "track_temp_c": 32,
        },
        "expected_elements": ["pace", "good", "consistent"],
        "urgency": "info",
        "must_contain_any": ["good", "pace", "consistent", "lap"],
    },
    {
        "name": "Pace - Traffic affected",
        "category": "pace_feedback",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 12,
            "lap_pct": 0.02,
            "position": 3,
            "fuel_laps_remaining": 18.0,
            "tire_wear": {"fl": 25, "fr": 28, "rl": 22, "rr": 24},
            "tire_temps": {"fl": 90, "fr": 94, "rl": 86, "rr": 88},
            "gap_ahead": 5.5,
            "gap_behind": 2.2,
            "last_lap_time": 142.5,
            "best_lap_time": 138.8,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 24,
        },
        "expected_elements": ["lap", "time", "traffic"],
        "urgency": "info",
        "must_contain_any": ["lap", "time", "down", "traffic"],
    },
    {
        "name": "Pace - Improving trend",
        "category": "pace_feedback",
        "input": {
            "car": "Audi R8 LMS GT3 Evo II",
            "car_class": "GT3",
            "car_traits": ["awd", "stable", "predictable"],
            "track": "Nurburgring GP",
            "track_type": "mixed",
            "lap": 10,
            "lap_pct": 0.15,
            "position": 7,
            "fuel_laps_remaining": 20.0,
            "tire_wear": {"fl": 18, "fr": 22, "rl": 15, "rr": 17},
            "tire_temps": {"fl": 88, "fr": 92, "rl": 84, "rr": 86},
            "gap_ahead": 3.8,
            "gap_behind": 4.5,
            "last_lap_time": 116.5,
            "best_lap_time": 116.8,
            "session_laps_remain": 28,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["pace", "good", "improving"],
        "urgency": "info",
        "must_contain_any": ["good", "pace", "best", "improving"],
    },

    # ---------------------------------------------------------------------
    # ROUTINE (5 cases) - Everything is fine updates
    # ---------------------------------------------------------------------
    {
        "name": "Routine - Watkins Glen mid-race",
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
        "urgency": "info",
        "must_contain_any": ["good", "fuel", "position", "gap"],
    },
    {
        "name": "Routine - Monza clean run",
        "category": "routine",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 12,
            "lap_pct": 0.35,
            "position": 4,
            "fuel_laps_remaining": 16.0,
            "tire_wear": {"fl": 22, "fr": 26, "rl": 18, "rr": 20},
            "tire_temps": {"fl": 90, "fr": 94, "rl": 86, "rr": 88},
            "gap_ahead": 2.8,
            "gap_behind": 3.5,
            "last_lap_time": 108.5,
            "best_lap_time": 108.2,
            "session_laps_remain": 22,
            "incident_count": 0,
            "track_temp_c": 32,
        },
        "expected_elements": ["good", "position", "fuel"],
        "urgency": "info",
        "must_contain_any": ["good", "position", "fuel", "gap"],
    },
    {
        "name": "Routine - Silverstone stable",
        "category": "routine",
        "input": {
            "car": "McLaren 720S GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "smooth_inputs"],
            "track": "Silverstone",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.62,
            "position": 5,
            "fuel_laps_remaining": 14.0,
            "tire_wear": {"fl": 28, "fr": 32, "rl": 24, "rr": 26},
            "tire_temps": {"fl": 90, "fr": 94, "rl": 86, "rr": 88},
            "gap_ahead": 4.2,
            "gap_behind": 5.8,
            "last_lap_time": 119.2,
            "best_lap_time": 118.5,
            "session_laps_remain": 18,
            "incident_count": 0,
            "track_temp_c": 24,
        },
        "expected_elements": ["good", "fuel", "position"],
        "urgency": "info",
        "must_contain_any": ["good", "fuel", "position", "lap"],
    },
    {
        "name": "Routine - Barcelona comfortable",
        "category": "routine",
        "input": {
            "car": "Porsche 911 GT3 R (992)",
            "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Barcelona",
            "track_type": "mixed",
            "lap": 10,
            "lap_pct": 0.72,
            "position": 3,
            "fuel_laps_remaining": 20.0,
            "tire_wear": {"fl": 18, "fr": 22, "rl": 15, "rr": 17},
            "tire_temps": {"fl": 88, "fr": 92, "rl": 84, "rr": 86},
            "gap_ahead": 5.5,
            "gap_behind": 6.2,
            "last_lap_time": 102.5,
            "best_lap_time": 102.2,
            "session_laps_remain": 25,
            "incident_count": 0,
            "track_temp_c": 30,
        },
        "expected_elements": ["good", "fuel", "gap"],
        "urgency": "info",
        "must_contain_any": ["good", "fuel", "gap", "position"],
    },
    {
        "name": "Routine - Road America endurance",
        "category": "routine",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Road America",
            "track_type": "high_speed",
            "lap": 25,
            "lap_pct": 0.45,
            "position": 7,
            "fuel_laps_remaining": 12.0,
            "tire_wear": {"fl": 35, "fr": 40, "rl": 30, "rr": 32},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 3.8,
            "gap_behind": 4.5,
            "last_lap_time": 133.5,
            "best_lap_time": 132.8,
            "session_laps_remain": 30,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["good", "fuel", "lap"],
        "urgency": "info",
        "must_contain_any": ["good", "fuel", "position", "gap"],
    },

    # ---------------------------------------------------------------------
    # EDGE CASES (5 cases) - Unusual or complex scenarios
    # ---------------------------------------------------------------------
    {
        "name": "Edge - Multiple issues (fuel + tires)",
        "category": "fuel_critical",
        "input": {
            "car": "Mercedes-AMG GT3",
            "car_class": "GT3",
            "car_traits": ["front_engine", "stable", "strong_brakes"],
            "track": "Spa",
            "track_type": "high_speed",
            "lap": 22,
            "lap_pct": 0.75,
            "position": 8,
            "fuel_laps_remaining": 1.5,
            "tire_wear": {"fl": 75, "fr": 82, "rl": 70, "rr": 74},
            "tire_temps": {"fl": 100, "fr": 108, "rl": 96, "rr": 98},
            "gap_ahead": 4.5,
            "gap_behind": 6.2,
            "last_lap_time": 142.5,
            "best_lap_time": 139.8,
            "session_laps_remain": 12,
            "incident_count": 2,
            "track_temp_c": 30,
        },
        "expected_elements": ["box", "pit", "fuel", "tire"],
        "urgency": "critical",
        "must_contain_any": ["box", "pit"],
    },
    {
        "name": "Edge - Production car (MX-5)",
        "category": "position_battle",
        "input": {
            "car": "Mazda MX-5 Cup",
            "car_class": "production",
            "car_traits": ["momentum_car", "draft_dependent", "forgiving"],
            "track": "Laguna Seca",
            "track_type": "technical",
            "lap": 8,
            "lap_pct": 0.55,
            "position": 3,
            "fuel_laps_remaining": 15.0,
            "tire_wear": {"fl": 12, "fr": 15, "rl": 10, "rr": 12},
            "tire_temps": {"fl": 82, "fr": 86, "rl": 78, "rr": 80},
            "gap_ahead": 0.5,
            "gap_behind": 0.8,
            "last_lap_time": 98.2,
            "best_lap_time": 97.8,
            "session_laps_remain": 12,
            "incident_count": 0,
            "track_temp_c": 26,
        },
        "expected_elements": ["gap", "draft", "momentum"],
        "urgency": "info",
        "must_contain_any": ["gap", "push", "attack", "close"],
    },
    {
        "name": "Edge - LMDh hybrid prototype",
        "category": "routine",
        "input": {
            "car": "Porsche 963",
            "car_class": "LMDh",
            "car_traits": ["hybrid", "high_downforce", "complex_systems"],
            "track": "Le Mans",
            "track_type": "high_speed",
            "lap": 30,
            "lap_pct": 0.45,
            "position": 2,
            "fuel_laps_remaining": 8.0,
            "tire_wear": {"fl": 35, "fr": 38, "rl": 30, "rr": 32},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 12.5,
            "gap_behind": 18.0,
            "last_lap_time": 198.2,
            "best_lap_time": 197.5,
            "session_laps_remain": 80,
            "incident_count": 0,
            "track_temp_c": 28,
        },
        "expected_elements": ["position", "fuel", "good"],
        "urgency": "info",
        "must_contain_any": ["good", "fuel", "position", "gap"],
    },
    {
        "name": "Edge - High incidents",
        "category": "routine",
        "input": {
            "car": "Ferrari 296 GT3",
            "car_class": "GT3",
            "car_traits": ["mid_engine", "high_downforce", "aggressive_diff"],
            "track": "Monza",
            "track_type": "high_speed",
            "lap": 18,
            "lap_pct": 0.28,
            "position": 12,
            "fuel_laps_remaining": 10.0,
            "tire_wear": {"fl": 32, "fr": 36, "rl": 28, "rr": 30},
            "tire_temps": {"fl": 92, "fr": 96, "rl": 88, "rr": 90},
            "gap_ahead": 2.5,
            "gap_behind": 3.8,
            "last_lap_time": 109.5,
            "best_lap_time": 108.2,
            "session_laps_remain": 15,
            "incident_count": 12,
            "track_temp_c": 34,
        },
        "expected_elements": ["clean", "careful", "incident"],
        "urgency": "warning",
        "must_contain_any": ["clean", "careful", "incident", "good"],
    },
    {
        "name": "Edge - Final lap scenario",
        "category": "position_battle",
        "input": {
            "car": "BMW M4 GT3",
            "car_class": "GT3",
            "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
            "track": "Nurburgring GP",
            "track_type": "mixed",
            "lap": 25,
            "lap_pct": 0.15,
            "position": 4,
            "fuel_laps_remaining": 2.0,
            "tire_wear": {"fl": 58, "fr": 62, "rl": 52, "rr": 55},
            "tire_temps": {"fl": 96, "fr": 100, "rl": 92, "rr": 94},
            "gap_ahead": 0.8,
            "gap_behind": 1.2,
            "last_lap_time": 117.2,
            "best_lap_time": 116.5,
            "session_laps_remain": 1,
            "incident_count": 1,
            "track_temp_c": 26,
        },
        "expected_elements": ["final", "lap", "push", "attack"],
        "urgency": "critical",
        "must_contain_any": ["final", "last", "push", "attack", "lap"],
    },
]


# =============================================================================
# ENHANCED METRICS
# =============================================================================

@dataclass
class EnhancedMetrics:
    """Enhanced metrics for comprehensive evaluation."""
    # Basic metrics
    word_count: int = 0
    is_concise: bool = False  # Under 40 words
    is_tts_suitable: bool = False  # 10-35 words, no special chars

    # Content quality
    has_telemetry_reference: bool = False  # References actual values
    has_track_reference: bool = False  # Mentions track or corners
    has_car_appropriate_advice: bool = False  # Car-specific guidance

    # Situation awareness
    urgency_appropriate: bool = False  # Response matches situation urgency
    contains_required_element: bool = False  # Has at least one must_contain element
    expected_elements_found: int = 0
    expected_elements_total: int = 0

    # Quality guards
    contains_hallucination: bool = False  # Fake names, wrong systems
    is_actionable: bool = False  # Contains action verb
    is_specific: bool = False  # Contains specific values or references

    # Numeric accuracy
    references_correct_values: bool = False  # Numbers match input

    # Overall score (0-100)
    composite_score: float = 0.0


@dataclass
class EvalResult:
    """Result for a single test case."""
    name: str
    category: str
    urgency: str
    input_summary: str
    input_data: Dict[str, Any]
    base_response: str
    finetuned_response: str
    base_metrics: EnhancedMetrics
    finetuned_metrics: EnhancedMetrics
    base_latency_ms: float = 0.0
    finetuned_latency_ms: float = 0.0


def analyze_response(
    response: str,
    test_case: Dict[str, Any],
    input_data: Dict[str, Any]
) -> EnhancedMetrics:
    """Analyze a response with enhanced metrics."""
    response_lower = response.lower()
    words = response.split()
    word_count = len(words)

    metrics = EnhancedMetrics()
    metrics.word_count = word_count

    # Conciseness and TTS suitability
    metrics.is_concise = word_count <= 40
    metrics.is_tts_suitable = (
        10 <= word_count <= 35 and
        not re.search(r'[*_#\[\]]', response)  # No markdown
    )

    # Telemetry references (numbers in response)
    metrics.has_telemetry_reference = bool(re.search(r'\d+\.?\d*', response))

    # Track references
    track_name = input_data.get("track", "").lower()
    track_words = track_name.split()
    corner_keywords = [
        "turn", "corner", "curve", "chicane", "hairpin", "esses",
        "straight", "kink", "corkscrew", "carousel", "dipper",
        "lesmo", "parabolica", "ascari", "variante", "eau rouge",
        "blanchimont", "bus stop", "copse", "maggots", "becketts"
    ]
    metrics.has_track_reference = (
        any(word in response_lower for word in track_words if len(word) > 2) or
        any(kw in response_lower for kw in corner_keywords)
    )

    # Car-appropriate advice
    car_traits = input_data.get("car_traits", [])
    trait_keywords = {
        "rear_engine": ["trail", "brake", "rear", "rotation", "oversteer"],
        "mid_engine": ["balance", "entry", "rotation", "neutral"],
        "front_engine": ["understeer", "entry", "stable", "front"],
        "awd": ["traction", "power", "stable", "all-wheel"],
        "trail_brake_critical": ["trail", "brake", "entry", "rotation"],
        "momentum_car": ["momentum", "carry", "speed", "flow"],
        "hybrid": ["deploy", "energy", "battery", "harvest"],
        "high_downforce": ["aero", "downforce", "grip", "load"],
        "strong_brakes": ["brake", "braking", "stop"],
        "draft_dependent": ["draft", "slipstream", "tow"],
    }

    metrics.has_car_appropriate_advice = False
    for trait in car_traits:
        if trait in trait_keywords:
            if any(kw in response_lower for kw in trait_keywords[trait]):
                metrics.has_car_appropriate_advice = True
                break

    # Urgency appropriateness
    urgency = test_case.get("urgency", "info")
    urgency_keywords = {
        "critical": ["box", "pit", "now", "critical", "immediately", "urgent"],
        "warning": ["manage", "careful", "watch", "warn", "soon", "attention"],
        "info": ["good", "looking", "gap", "position", "lap", "fuel"],
    }

    if urgency in urgency_keywords:
        metrics.urgency_appropriate = any(
            kw in response_lower for kw in urgency_keywords[urgency]
        )

    # Required elements check
    must_contain = test_case.get("must_contain_any", [])
    metrics.contains_required_element = any(
        elem.lower() in response_lower for elem in must_contain
    )

    # Expected elements
    expected = test_case.get("expected_elements", [])
    metrics.expected_elements_found = sum(
        1 for elem in expected if elem.lower() in response_lower
    )
    metrics.expected_elements_total = len(expected)

    # Hallucination detection
    hallucination_patterns = [
        r'\b(hamilton|verstappen|leclerc|norris|bogdan|smith|jones|driver)\b',
        r'\bkers\b',
    ]
    car_class = input_data.get("car_class", "")
    if car_class != "F1":
        hallucination_patterns.append(r'\bdrs\b')

    metrics.contains_hallucination = any(
        re.search(pattern, response_lower)
        for pattern in hallucination_patterns
    )

    # Actionability (contains action verbs)
    action_verbs = [
        "box", "pit", "push", "defend", "attack", "manage", "watch",
        "careful", "save", "maintain", "hold", "stay", "go", "keep"
    ]
    metrics.is_actionable = any(verb in response_lower for verb in action_verbs)

    # Specificity (references specific values)
    metrics.is_specific = (
        metrics.has_telemetry_reference or
        metrics.has_track_reference or
        re.search(r'(p\d|position\s*\d|lap\s*\d)', response_lower) is not None
    )

    # Numeric accuracy check
    fuel_laps = input_data.get("fuel_laps_remaining", 0)
    position = input_data.get("position", 0)
    gap_ahead = input_data.get("gap_ahead", 0)
    gap_behind = input_data.get("gap_behind", 0)

    # Check if any referenced numbers are approximately correct
    numbers_in_response = re.findall(r'\d+\.?\d*', response)
    metrics.references_correct_values = False
    for num_str in numbers_in_response:
        try:
            num = float(num_str)
            # Allow some tolerance
            if (abs(num - fuel_laps) < 1 or
                abs(num - position) < 1 or
                abs(num - gap_ahead) < 0.5 or
                abs(num - gap_behind) < 0.5):
                metrics.references_correct_values = True
                break
        except ValueError:
            pass

    # Calculate composite score (0-100)
    score = 0.0
    weights = {
        "is_concise": 10,
        "is_tts_suitable": 10,
        "has_telemetry_reference": 10,
        "has_track_reference": 8,
        "has_car_appropriate_advice": 8,
        "urgency_appropriate": 15,
        "contains_required_element": 15,
        "is_actionable": 10,
        "is_specific": 8,
        "references_correct_values": 6,
    }

    if metrics.is_concise:
        score += weights["is_concise"]
    if metrics.is_tts_suitable:
        score += weights["is_tts_suitable"]
    if metrics.has_telemetry_reference:
        score += weights["has_telemetry_reference"]
    if metrics.has_track_reference:
        score += weights["has_track_reference"]
    if metrics.has_car_appropriate_advice:
        score += weights["has_car_appropriate_advice"]
    if metrics.urgency_appropriate:
        score += weights["urgency_appropriate"]
    if metrics.contains_required_element:
        score += weights["contains_required_element"]
    if metrics.is_actionable:
        score += weights["is_actionable"]
    if metrics.is_specific:
        score += weights["is_specific"]
    if metrics.references_correct_values:
        score += weights["references_correct_values"]

    # Penalties
    if metrics.contains_hallucination:
        score -= 20
    if word_count > 50:
        score -= 10
    if word_count < 5:
        score -= 15

    metrics.composite_score = max(0, min(100, score))

    return metrics


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

def generate_response(
    model,
    tokenizer,
    input_data: dict,
    max_new_tokens: int = 80
) -> Tuple[str, float]:
    """Generate a response from the model. Returns (response, latency_ms)."""
    instruction = "You are a race engineer. Given the car, track, and telemetry, provide a brief callout to the driver."
    input_json = json.dumps(input_data)

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_json}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start_time) * 1000

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in response:
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()

    return response.strip(), latency_ms


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(
    base_model_name: str,
    adapter_path: Optional[str],
    test_cases: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[EvalResult]:
    """Run evaluation on all test cases."""

    results = []

    # Load models
    if verbose:
        print("\n" + "=" * 70)
        print("LOADING MODELS")
        print("=" * 70)

    base_model, base_tokenizer = load_base_model(base_model_name)

    if adapter_path and Path(adapter_path).exists():
        ft_model, ft_tokenizer = load_finetuned_model(adapter_path, base_model_name)
    else:
        print(f"Warning: Adapter not found at {adapter_path}, using base model only")
        ft_model, ft_tokenizer = base_model, base_tokenizer

    if verbose:
        print("\n" + "=" * 70)
        print(f"RUNNING EVALUATION ({len(test_cases)} test cases)")
        print("=" * 70)

    for i, test_case in enumerate(test_cases):
        if verbose:
            print(f"\n[{i+1}/{len(test_cases)}] {test_case['name']} ({test_case['category']})")

        input_data = test_case["input"]

        # Generate responses
        if verbose:
            print("  Generating base response...")
        base_response, base_latency = generate_response(
            base_model, base_tokenizer, input_data
        )

        if verbose:
            print("  Generating fine-tuned response...")
        ft_response, ft_latency = generate_response(
            ft_model, ft_tokenizer, input_data
        )

        # Analyze responses
        base_metrics = analyze_response(base_response, test_case, input_data)
        ft_metrics = analyze_response(ft_response, test_case, input_data)

        # Create input summary
        inp = input_data
        input_summary = (
            f"{inp['car']} @ {inp['track']}, P{inp['position']}, "
            f"{inp['fuel_laps_remaining']} laps fuel, "
            f"gap ahead {inp['gap_ahead']}s, behind {inp['gap_behind']}s"
        )

        result = EvalResult(
            name=test_case["name"],
            category=test_case["category"],
            urgency=test_case.get("urgency", "info"),
            input_summary=input_summary,
            input_data=input_data,
            base_response=base_response,
            finetuned_response=ft_response,
            base_metrics=base_metrics,
            finetuned_metrics=ft_metrics,
            base_latency_ms=base_latency,
            finetuned_latency_ms=ft_latency,
        )
        results.append(result)

        if verbose:
            print(f"  Base ({base_metrics.composite_score:.0f}): {base_response[:70]}...")
            print(f"  FT   ({ft_metrics.composite_score:.0f}): {ft_response[:70]}...")

    return results


# =============================================================================
# REPORTING
# =============================================================================

def calculate_category_stats(results: List[EvalResult]) -> Dict[str, Dict]:
    """Calculate per-category statistics."""
    categories = defaultdict(lambda: {
        "count": 0,
        "base_scores": [],
        "ft_scores": [],
        "base_wins": 0,
        "ft_wins": 0,
        "ties": 0,
    })

    for result in results:
        cat = result.category
        categories[cat]["count"] += 1
        categories[cat]["base_scores"].append(result.base_metrics.composite_score)
        categories[cat]["ft_scores"].append(result.finetuned_metrics.composite_score)

        if result.finetuned_metrics.composite_score > result.base_metrics.composite_score:
            categories[cat]["ft_wins"] += 1
        elif result.base_metrics.composite_score > result.finetuned_metrics.composite_score:
            categories[cat]["base_wins"] += 1
        else:
            categories[cat]["ties"] += 1

    # Calculate averages
    for cat in categories:
        categories[cat]["base_avg"] = statistics.mean(categories[cat]["base_scores"])
        categories[cat]["ft_avg"] = statistics.mean(categories[cat]["ft_scores"])
        if len(categories[cat]["base_scores"]) > 1:
            categories[cat]["base_std"] = statistics.stdev(categories[cat]["base_scores"])
            categories[cat]["ft_std"] = statistics.stdev(categories[cat]["ft_scores"])
        else:
            categories[cat]["base_std"] = 0
            categories[cat]["ft_std"] = 0

    return dict(categories)


def print_results(results: List[EvalResult]):
    """Print comprehensive evaluation results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)

    # Overall summary
    n = len(results)
    base_scores = [r.base_metrics.composite_score for r in results]
    ft_scores = [r.finetuned_metrics.composite_score for r in results]

    base_avg = statistics.mean(base_scores)
    ft_avg = statistics.mean(ft_scores)
    base_std = statistics.stdev(base_scores) if n > 1 else 0
    ft_std = statistics.stdev(ft_scores) if n > 1 else 0

    ft_wins = sum(1 for r in results if r.finetuned_metrics.composite_score > r.base_metrics.composite_score)
    base_wins = sum(1 for r in results if r.base_metrics.composite_score > r.finetuned_metrics.composite_score)
    ties = n - ft_wins - base_wins

    print(f"\n{'OVERALL SUMMARY':^80}")
    print("-" * 80)
    print(f"Total test cases: {n}")
    print(f"Fine-tuned wins: {ft_wins} ({ft_wins/n*100:.1f}%)")
    print(f"Base model wins:  {base_wins} ({base_wins/n*100:.1f}%)")
    print(f"Ties:             {ties} ({ties/n*100:.1f}%)")
    print()
    print(f"{'Model':<20} {'Avg Score':>12} {'Std Dev':>12} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    print(f"{'Base':<20} {base_avg:>12.1f} {base_std:>12.1f} {min(base_scores):>8.1f} {max(base_scores):>8.1f}")
    print(f"{'Fine-tuned':<20} {ft_avg:>12.1f} {ft_std:>12.1f} {min(ft_scores):>8.1f} {max(ft_scores):>8.1f}")

    # Per-category breakdown
    print(f"\n{'CATEGORY BREAKDOWN':^80}")
    print("-" * 80)

    cat_stats = calculate_category_stats(results)
    print(f"{'Category':<20} {'Count':>6} {'Base Avg':>10} {'FT Avg':>10} {'FT Wins':>10} {'Winner':>12}")
    print("-" * 80)

    for cat in sorted(cat_stats.keys()):
        stats = cat_stats[cat]
        winner = "Fine-tuned" if stats["ft_wins"] > stats["base_wins"] else "Base" if stats["base_wins"] > stats["ft_wins"] else "Tie"
        print(f"{cat:<20} {stats['count']:>6} {stats['base_avg']:>10.1f} {stats['ft_avg']:>10.1f} {stats['ft_wins']:>10} {winner:>12}")

    # Detailed metrics comparison
    print(f"\n{'DETAILED METRICS COMPARISON':^80}")
    print("-" * 80)

    metric_names = [
        ("is_concise", "Concise (<40 words)"),
        ("is_tts_suitable", "TTS Suitable"),
        ("has_telemetry_reference", "Has Telemetry Ref"),
        ("has_track_reference", "Has Track Ref"),
        ("has_car_appropriate_advice", "Car-Appropriate"),
        ("urgency_appropriate", "Urgency Match"),
        ("contains_required_element", "Required Element"),
        ("is_actionable", "Is Actionable"),
        ("is_specific", "Is Specific"),
        ("contains_hallucination", "Has Hallucination"),
    ]

    print(f"{'Metric':<25} {'Base':>15} {'Fine-tuned':>15} {'Winner':>15}")
    print("-" * 70)

    for attr, name in metric_names:
        base_count = sum(1 for r in results if getattr(r.base_metrics, attr))
        ft_count = sum(1 for r in results if getattr(r.finetuned_metrics, attr))

        # For hallucination, lower is better
        if attr == "contains_hallucination":
            winner = "Fine-tuned" if ft_count < base_count else "Base" if base_count < ft_count else "Tie"
        else:
            winner = "Fine-tuned" if ft_count > base_count else "Base" if base_count > ft_count else "Tie"

        print(f"{name:<25} {base_count:>13}/{n} {ft_count:>13}/{n} {winner:>15}")

    # Latency comparison
    base_latencies = [r.base_latency_ms for r in results]
    ft_latencies = [r.finetuned_latency_ms for r in results]

    print(f"\n{'LATENCY COMPARISON':^80}")
    print("-" * 80)
    print(f"{'Model':<20} {'Avg (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
    print("-" * 60)
    print(f"{'Base':<20} {statistics.mean(base_latencies):>12.1f} {min(base_latencies):>12.1f} {max(base_latencies):>12.1f}")
    print(f"{'Fine-tuned':<20} {statistics.mean(ft_latencies):>12.1f} {min(ft_latencies):>12.1f} {max(ft_latencies):>12.1f}")

    # Best and worst cases
    print(f"\n{'TOP 5 FINE-TUNED WINS (by score difference)':^80}")
    print("-" * 80)

    sorted_by_improvement = sorted(
        results,
        key=lambda r: r.finetuned_metrics.composite_score - r.base_metrics.composite_score,
        reverse=True
    )

    for r in sorted_by_improvement[:5]:
        diff = r.finetuned_metrics.composite_score - r.base_metrics.composite_score
        print(f"\n{r.name} (diff: +{diff:.1f})")
        print(f"  Base ({r.base_metrics.composite_score:.0f}): {r.base_response[:60]}...")
        print(f"  FT   ({r.finetuned_metrics.composite_score:.0f}): {r.finetuned_response[:60]}...")

    print(f"\n{'TOP 5 BASE MODEL WINS (by score difference)':^80}")
    print("-" * 80)

    for r in sorted_by_improvement[-5:]:
        diff = r.finetuned_metrics.composite_score - r.base_metrics.composite_score
        print(f"\n{r.name} (diff: {diff:.1f})")
        print(f"  Base ({r.base_metrics.composite_score:.0f}): {r.base_response[:60]}...")
        print(f"  FT   ({r.finetuned_metrics.composite_score:.0f}): {r.finetuned_response[:60]}...")

    # Final verdict
    print(f"\n{'=' * 80}")
    print(f"{'FINAL VERDICT':^80}")
    print(f"{'=' * 80}")

    improvement = ft_avg - base_avg
    win_rate = ft_wins / n * 100

    if improvement > 5 and win_rate > 60:
        verdict = "STRONG IMPROVEMENT - Fine-tuned model significantly outperforms base"
    elif improvement > 0 and win_rate > 50:
        verdict = "MODERATE IMPROVEMENT - Fine-tuned model shows consistent gains"
    elif improvement > -5 and win_rate > 40:
        verdict = "MARGINAL DIFFERENCE - Models perform similarly"
    else:
        verdict = "REGRESSION - Base model may be preferable"

    print(f"\nScore improvement: {improvement:+.1f} points")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"\n>>> {verdict} <<<")
    print()


def save_results(results: List[EvalResult], output_path: str):
    """Save detailed results to JSON."""
    data = {
        "summary": {
            "total_cases": len(results),
            "base_avg_score": statistics.mean([r.base_metrics.composite_score for r in results]),
            "ft_avg_score": statistics.mean([r.finetuned_metrics.composite_score for r in results]),
            "ft_wins": sum(1 for r in results if r.finetuned_metrics.composite_score > r.base_metrics.composite_score),
            "base_wins": sum(1 for r in results if r.base_metrics.composite_score > r.finetuned_metrics.composite_score),
        },
        "category_stats": calculate_category_stats(results),
        "results": []
    }

    for result in results:
        data["results"].append({
            "name": result.name,
            "category": result.category,
            "urgency": result.urgency,
            "input_summary": result.input_summary,
            "base_response": result.base_response,
            "finetuned_response": result.finetuned_response,
            "base_metrics": asdict(result.base_metrics),
            "finetuned_metrics": asdict(result.finetuned_metrics),
            "base_latency_ms": result.base_latency_ms,
            "finetuned_latency_ms": result.finetuned_latency_ms,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of race engineer model"
    )
    parser.add_argument(
        "--base-model", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter", type=str,
        default="models/race-engineer-llama",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/eval_comprehensive.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--cases", type=int,
        default=None,
        help="Number of test cases to run (default: all)"
    )
    parser.add_argument(
        "--category", type=str,
        default=None,
        help="Filter to specific category"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Select test cases
    test_cases = EVAL_CASES

    if args.category:
        test_cases = [tc for tc in test_cases if tc["category"] == args.category]
        print(f"Filtered to {len(test_cases)} cases in category: {args.category}")

    if args.cases:
        test_cases = test_cases[:args.cases]

    print(f"Running comprehensive evaluation with {len(test_cases)} test cases")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")

    # Run evaluation
    results = run_evaluation(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        test_cases=test_cases,
        verbose=not args.quiet,
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()