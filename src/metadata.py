"""
Car and track metadata for race engineer context.

Provides domain knowledge about cars and tracks for enriching LLM prompts
with semantic context (car class, traits, track corners, etc.).
"""

from typing import Optional, Dict, List, Any


# =============================================================================
# CAR METADATA
# =============================================================================

CARS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # PRODUCTION / ENTRY LEVEL
    # =========================================================================
    "mx5_cup": {
        "name": "Mazda MX-5 Cup",
        "class": "production",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "low",
        "traits": ["momentum_car", "draft_dependent", "forgiving", "rear_wheel_drive"],
        "advice_style": "focus on momentum and consistency, stay in the draft"
    },
    "bmw_m2_csr": {
        "name": "BMW M2 CS Racing",
        "class": "TCX",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["balanced", "rear_wheel_drive", "forgiving", "stable"],
        "advice_style": "smooth inputs, manage rear traction on exit"
    },
    "ford_mustang_fr500s": {
        "name": "Ford Mustang FR500S",
        "class": "production",
        "power": "medium",
        "braking": "medium",
        "aero": "minimal",
        "tire_deg": "medium",
        "traits": ["rear_wheel_drive", "tail_happy", "powerful", "front_engine"],
        "advice_style": "careful throttle on exit, use weight transfer"
    },
    "pontiac_solstice": {
        "name": "Pontiac Solstice",
        "class": "production",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "low",
        "traits": ["lightweight", "momentum_car", "rear_wheel_drive", "forgiving"],
        "advice_style": "carry speed through corners, be smooth"
    },
    "cadillac_cts_v": {
        "name": "Cadillac CTS-V Racecar",
        "class": "production",
        "power": "high",
        "braking": "heavy",
        "aero": "low",
        "tire_deg": "medium",
        "traits": ["heavy", "powerful", "rear_wheel_drive", "front_engine"],
        "advice_style": "manage weight transfer, smooth throttle application"
    },
    "spec_racer_ford": {
        "name": "SCCA Spec Racer Ford",
        "class": "production",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "low",
        "traits": ["momentum_car", "equal_racing", "forgiving", "lightweight"],
        "advice_style": "pure momentum, late apex for better exit"
    },
    "ruf_rt12r": {
        "name": "Ruf RT 12R",
        "class": "production",
        "power": "very_high",
        "braking": "heavy",
        "aero": "low",
        "tire_deg": "medium",
        "traits": ["rear_engine", "powerful", "tail_happy", "precise_inputs"],
        "advice_style": "respect the rear engine, smooth throttle application"
    },
    "ruf_rt12r_track": {
        "name": "Ruf RT 12R Track",
        "class": "production",
        "power": "very_high",
        "braking": "heavy",
        "aero": "medium",
        "tire_deg": "medium",
        "traits": ["rear_engine", "powerful", "more_downforce", "precise_inputs"],
        "advice_style": "use the added downforce, manage rear weight transfer"
    },
    "vw_jetta_tdi": {
        "name": "Volkswagen Jetta TDI Cup",
        "class": "touring",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "low",
        "traits": ["front_wheel_drive", "diesel_torque", "forgiving", "momentum_car"],
        "advice_style": "use torque out of slow corners, trail brake to rotate"
    },
    "kia_optima": {
        "name": "Kia Optima",
        "class": "touring",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "low",
        "traits": ["front_wheel_drive", "forgiving", "momentum_car", "equal_racing"],
        "advice_style": "carry speed, trail brake for rotation"
    },

    # =========================================================================
    # TCR / TOURING CARS
    # =========================================================================
    "audi_rs3_lms": {
        "name": "Audi RS 3 LMS",
        "class": "TCR",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_wheel_drive", "torque_steer", "stable_braking", "forgiving"],
        "advice_style": "manage torque steer on exit, trail brake for rotation"
    },
    "honda_civic_type_r_tcr": {
        "name": "Honda Civic Type R TCR",
        "class": "TCR",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_wheel_drive", "balanced", "responsive", "nimble"],
        "advice_style": "use the balance, aggressive trail braking"
    },
    "hyundai_elantra_n_tcr": {
        "name": "Hyundai Elantra N TCR",
        "class": "TCR",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_wheel_drive", "responsive", "stable", "forgiving"],
        "advice_style": "trail brake for rotation, smooth throttle"
    },
    "hyundai_veloster_n_tcr": {
        "name": "Hyundai Veloster N TCR",
        "class": "TCR",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_wheel_drive", "nimble", "short_wheelbase", "agile"],
        "advice_style": "use the nimbleness, careful on kerbs"
    },

    # =========================================================================
    # SUPERCARS (V8 SUPERCARS)
    # =========================================================================
    "supercars_ford_mustang": {
        "name": "Supercars Ford Mustang GT",
        "class": "supercars",
        "power": "high",
        "braking": "heavy",
        "aero": "medium",
        "tire_deg": "high",
        "traits": ["rear_wheel_drive", "tail_happy", "sequential_gearbox", "no_abs"],
        "advice_style": "manage rear grip, smooth throttle, no ABS so modulate brakes"
    },
    "supercars_chevy_camaro": {
        "name": "Supercars Chevrolet Camaro ZL1",
        "class": "supercars",
        "power": "high",
        "braking": "heavy",
        "aero": "medium",
        "tire_deg": "high",
        "traits": ["rear_wheel_drive", "powerful", "sequential_gearbox", "no_abs"],
        "advice_style": "manage the power on exit, careful brake modulation"
    },

    # =========================================================================
    # GT4 CARS
    # =========================================================================
    "aston_martin_vantage_gt4": {
        "name": "Aston Martin Vantage GT4",
        "class": "GT4",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_engine", "stable", "rear_wheel_drive", "predictable"],
        "advice_style": "use the stability, smooth inputs"
    },
    "bmw_m4_gt4": {
        "name": "BMW M4 GT4",
        "class": "GT4",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_engine", "balanced", "forgiving", "hard_to_spin"],
        "advice_style": "push the limits, very forgiving platform"
    },
    "ford_mustang_gt4": {
        "name": "Ford Mustang GT4",
        "class": "GT4",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_engine", "stable", "forgiving", "rear_wheel_drive"],
        "advice_style": "use the stability, manage rear on exit"
    },
    "mclaren_570s_gt4": {
        "name": "McLaren 570S GT4",
        "class": "GT4",
        "power": "medium",
        "braking": "medium",
        "aero": "medium",
        "tire_deg": "low",
        "traits": ["mid_engine", "responsive", "agile", "aero_sensitive"],
        "advice_style": "aggressive turn-in, use the mid-engine balance"
    },
    "mercedes_amg_gt4": {
        "name": "Mercedes-AMG GT4",
        "class": "GT4",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["front_engine", "torquey", "good_over_kerbs", "stable"],
        "advice_style": "use the torque on exit, attack the kerbs"
    },
    "porsche_718_cayman_gt4": {
        "name": "Porsche 718 Cayman GT4 Clubsport MR",
        "class": "GT4",
        "power": "medium",
        "braking": "medium",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["mid_engine", "nimble", "forgiving", "balanced"],
        "advice_style": "use the mid-engine agility, more forgiving than 911"
    },

    # =========================================================================
    # GT3 CARS
    # =========================================================================
    "acura_nsx_gt3": {
        "name": "Acura NSX GT3 Evo 22",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "twin_turbo", "short_wheelbase", "responsive"],
        "advice_style": "use the short wheelbase for rotation, manage turbo lag"
    },
    "aston_martin_vantage_gt3": {
        "name": "Aston Martin Vantage GT3 EVO",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["front_engine", "stable", "forgiving", "predictable"],
        "advice_style": "use the front-engine stability, push under braking"
    },
    "audi_r8_lms_gt3": {
        "name": "Audi R8 LMS EVO II GT3",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "balanced", "predictable", "v10_power"],
        "advice_style": "use the balance, consistent through stints"
    },
    "bmw_m4_gt3": {
        "name": "BMW M4 GT3",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["front_engine", "balanced", "strong_brakes", "forgiving"],
        "advice_style": "use the strong brakes, balanced handling rewards smooth inputs"
    },
    "corvette_z06_gt3r": {
        "name": "Chevrolet Corvette Z06 GT3.R",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["front_mid_engine", "aggressive", "rear_limited", "powerful"],
        "advice_style": "manage rear tires, use the low center of gravity"
    },
    "ferrari_296_gt3": {
        "name": "Ferrari 296 GT3",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "responsive", "agile", "hybrid_heritage"],
        "advice_style": "aggressive turn-in, manage front temps in fast corners"
    },
    "ferrari_488_gt3": {
        "name": "Ferrari 488 GT3 Evo 2020",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "beginner_friendly", "slight_understeer", "forgiving"],
        "advice_style": "great starter GT3, manage the understeer with trail braking"
    },
    "ford_mustang_gt3": {
        "name": "Ford Mustang GT3",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["front_engine", "stable", "strong_brakes", "v8_power"],
        "advice_style": "use the stable platform, strong under braking"
    },
    "lamborghini_huracan_gt3": {
        "name": "Lamborghini Huracán GT3 EVO",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "predictable", "v10_power", "balanced"],
        "advice_style": "use the predictability, smooth inputs rewarded"
    },
    "mclaren_720s_gt3": {
        "name": "McLaren 720S GT3 EVO",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "agile", "aero_dependent", "twin_turbo"],
        "advice_style": "use the aero, aggressive mid-corner"
    },
    "mercedes_amg_gt3": {
        "name": "Mercedes-AMG GT3 2020",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["front_engine", "stable_braking", "predictable", "easy_to_catch"],
        "advice_style": "use stability under braking, manage front temps"
    },
    "porsche_911_gt3_r": {
        "name": "Porsche 911 GT3 R (992)",
        "class": "GT3",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["rear_engine", "technical", "trail_brake_critical", "knife_edge"],
        "advice_style": "trail braking is essential, respect the rear engine"
    },

    # =========================================================================
    # CUP / ONE-MAKE SERIES
    # =========================================================================
    "porsche_911_gt3_cup_992": {
        "name": "Porsche 911 GT3 Cup (992)",
        "class": "cup",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["rear_engine", "no_abs", "no_tc", "trail_brake_friendly"],
        "advice_style": "no aids, trail braking essential, manage rear weight transfer"
    },
    "porsche_911_gt3_cup_992_2": {
        "name": "Porsche 911 Cup (992.2)",
        "class": "cup",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["rear_engine", "no_abs", "no_tc", "improved_aero"],
        "advice_style": "no aids, use improved aero, smooth brake application"
    },

    # =========================================================================
    # GTE CARS
    # =========================================================================
    "bmw_m8_gte": {
        "name": "BMW M8 GTE",
        "class": "GTE",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["front_engine", "heavy", "stable", "v8_turbo"],
        "advice_style": "use the stability, manage weight in direction changes"
    },
    "corvette_c8r": {
        "name": "Chevrolet Corvette C8.R",
        "class": "GTE",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "balanced", "responsive", "v8_power"],
        "advice_style": "use the mid-engine balance, aggressive on turn-in"
    },
    "ferrari_488_gte": {
        "name": "Ferrari 488 GTE",
        "class": "GTE",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "agile", "responsive", "v8_turbo"],
        "advice_style": "aggressive inputs, manage turbo response"
    },
    "ford_gt_gte": {
        "name": "Ford GT GTE",
        "class": "GTE",
        "power": "high",
        "braking": "heavy",
        "aero": "very_high_downforce",
        "tire_deg": "medium",
        "traits": ["mid_engine", "aero_dependent", "low_drag", "v6_turbo"],
        "advice_style": "use the aero, manage turbo lag on exit"
    },
    "porsche_911_rsr": {
        "name": "Porsche 911 RSR",
        "class": "GTE",
        "power": "high",
        "braking": "heavy",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["rear_engine", "trail_brake_friendly", "stable_braking", "flat_six"],
        "advice_style": "trail braking essential, use rear weight for rotation"
    },

    # =========================================================================
    # GTP / LMDH / HYPERCAR
    # =========================================================================
    "acura_arx_06": {
        "name": "Acura ARX-06 GTP",
        "class": "GTP",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["hybrid_boost", "mid_engine", "aero_platform", "massive_brakes"],
        "advice_style": "use hybrid zones, manage brake temps in heavy zones"
    },
    "bmw_m_hybrid_v8": {
        "name": "BMW M Hybrid V8",
        "class": "GTP",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["hybrid_boost", "aero_platform", "v8_turbo", "heavy"],
        "advice_style": "use hybrid deployment, manage aero platform"
    },
    "cadillac_v_series_r": {
        "name": "Cadillac V-Series.R GTP",
        "class": "GTP",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["hybrid_boost", "balanced", "strong_brakes", "v8_power"],
        "advice_style": "use the balance, deploy hybrid on exit"
    },
    "ferrari_499p": {
        "name": "Ferrari 499P",
        "class": "hypercar",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["hybrid_boost", "v6_turbo", "aero_dominant", "agile"],
        "advice_style": "extreme downforce, use hybrid, manage temps in long stints"
    },
    "porsche_963": {
        "name": "Porsche 963 GTP",
        "class": "GTP",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["hybrid_boost", "massive_brakes", "aero_platform", "v8_turbo"],
        "advice_style": "use hybrid zones, manage brake temps"
    },

    # =========================================================================
    # LMP2 / LMP3 / PROTOTYPE
    # =========================================================================
    "dallara_p217": {
        "name": "Dallara P217 LMP2",
        "class": "LMP2",
        "power": "high",
        "braking": "heavy",
        "aero": "very_high_downforce",
        "tire_deg": "medium",
        "traits": ["aero_sensitive", "downforce_dependent", "precise_inputs", "no_hybrid"],
        "advice_style": "respect the aero, smooth inputs, no hybrid to manage"
    },
    "ligier_js_p320": {
        "name": "Ligier JS P320",
        "class": "LMP3",
        "power": "medium",
        "braking": "medium",
        "aero": "high_downforce",
        "tire_deg": "low",
        "traits": ["aero_sensitive", "lightweight", "no_abs", "sequential_gearbox"],
        "advice_style": "use the lightweight agility, no ABS so modulate brakes"
    },
    "radical_sr10": {
        "name": "Radical SR10",
        "class": "prototype",
        "power": "medium",
        "braking": "medium",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["lightweight", "aero_dependent", "no_abs", "responsive"],
        "advice_style": "lightweight and agile, no ABS, use the downforce"
    },
    "radical_sr8": {
        "name": "Radical SR8",
        "class": "prototype",
        "power": "medium",
        "braking": "medium",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["lightweight", "high_grip", "no_abs", "v8_motorcycle_engine"],
        "advice_style": "revs are key, keep the momentum, no ABS"
    },

    # =========================================================================
    # FORMULA / OPEN WHEEL - F1 LEVEL
    # =========================================================================
    "mercedes_w13": {
        "name": "Mercedes-AMG W13 E Performance",
        "class": "F1",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "high",
        "traits": ["hybrid", "drs", "extreme_aero", "very_dirty_air_affected"],
        "advice_style": "manage tires carefully, use DRS zones, hybrid deployment critical"
    },
    "mercedes_w12": {
        "name": "Mercedes-AMG W12 E Performance",
        "class": "F1",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "high",
        "traits": ["hybrid", "drs", "extreme_aero", "dirty_air_affected"],
        "advice_style": "manage tires, use DRS, hybrid deployment matters"
    },
    "mclaren_mp4_30": {
        "name": "McLaren MP4-30",
        "class": "F1",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "high",
        "traits": ["hybrid", "drs", "high_downforce", "precise_inputs"],
        "advice_style": "precise inputs required, manage hybrid, tire management"
    },
    "williams_fw31": {
        "name": "Williams-Toyota FW31",
        "class": "F1",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["no_hybrid", "kers", "high_downforce", "responsive"],
        "advice_style": "use KERS for boost, high commitment corners"
    },

    # =========================================================================
    # FORMULA / OPEN WHEEL - INDYCAR
    # =========================================================================
    "dallara_ir18": {
        "name": "Dallara IR18 INDYCAR",
        "class": "IndyCar",
        "power": "very_high",
        "braking": "heavy",
        "aero": "very_high_downforce",
        "tire_deg": "medium",
        "traits": ["hybrid", "push_to_pass", "aero_kits", "versatile"],
        "advice_style": "use push to pass strategically, manage tire deg on ovals vs road"
    },
    "dallara_ir01": {
        "name": "Dallara iR-01",
        "class": "open_wheel",
        "power": "very_high",
        "braking": "very_heavy",
        "aero": "extreme_downforce",
        "tire_deg": "medium",
        "traits": ["v10_power", "fantasy_car", "extreme_grip", "precise_inputs"],
        "advice_style": "push hard, extreme grip available, smooth inputs"
    },

    # =========================================================================
    # FORMULA / OPEN WHEEL - SUPER FORMULA
    # =========================================================================
    "super_formula_sf23_honda": {
        "name": "Super Formula SF23 - Honda",
        "class": "super_formula",
        "power": "high",
        "braking": "heavy",
        "aero": "very_high_downforce",
        "tire_deg": "medium",
        "traits": ["high_downforce", "responsive", "no_hybrid", "precise_inputs"],
        "advice_style": "use the downforce, very responsive to inputs"
    },
    "super_formula_sf23_toyota": {
        "name": "Super Formula SF23 - Toyota",
        "class": "super_formula",
        "power": "high",
        "braking": "heavy",
        "aero": "very_high_downforce",
        "tire_deg": "medium",
        "traits": ["high_downforce", "responsive", "no_hybrid", "precise_inputs"],
        "advice_style": "use the downforce, very responsive to inputs"
    },
    "super_formula_lights": {
        "name": "Super Formula Lights",
        "class": "open_wheel",
        "power": "medium",
        "braking": "medium",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["high_downforce", "stepping_stone", "aero_sensitive", "precise_inputs"],
        "advice_style": "great prep for SF, manage the aero"
    },

    # =========================================================================
    # FORMULA / OPEN WHEEL - JUNIOR FORMULA
    # =========================================================================
    "dallara_f3": {
        "name": "Dallara F3",
        "class": "F3",
        "power": "medium",
        "braking": "medium",
        "aero": "very_high_downforce",
        "tire_deg": "high",
        "traits": ["aero_sensitive", "dirty_air_affected", "precise_inputs", "high_deg"],
        "advice_style": "clean air critical, precise on kerbs, manage tire deg"
    },
    "fia_f4": {
        "name": "FIA F4",
        "class": "F4",
        "power": "low",
        "braking": "light",
        "aero": "medium",
        "tire_deg": "low",
        "traits": ["beginner_friendly", "aero_sensitive", "forgiving", "good_stepping_stone"],
        "advice_style": "great for learning aero cars, smooth inputs"
    },
    "formula_renault_3_5": {
        "name": "Formula Renault 3.5",
        "class": "open_wheel",
        "power": "high",
        "braking": "heavy",
        "aero": "very_high_downforce",
        "tire_deg": "medium",
        "traits": ["high_downforce", "precise_inputs", "aero_dependent", "v8_power"],
        "advice_style": "use the downforce, precise inputs required"
    },
    "formula_renault_2_0": {
        "name": "Formula Renault 2.0",
        "class": "open_wheel",
        "power": "medium",
        "braking": "medium",
        "aero": "medium",
        "tire_deg": "low",
        "traits": ["balanced", "forgiving", "good_stepping_stone", "momentum_car"],
        "advice_style": "good transition to aero cars, carry momentum"
    },

    # =========================================================================
    # FORMULA / OPEN WHEEL - INDY LADDER
    # =========================================================================
    "indy_pro_2000": {
        "name": "Indy Pro 2000",
        "class": "open_wheel",
        "power": "medium",
        "braking": "medium",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["high_downforce", "precise_inputs", "indy_ladder", "responsive"],
        "advice_style": "use the downforce, prep for IndyCar"
    },
    "dallara_il15": {
        "name": "Dallara IL-15 (Indy Lights)",
        "class": "open_wheel",
        "power": "medium",
        "braking": "medium",
        "aero": "high_downforce",
        "tire_deg": "medium",
        "traits": ["high_downforce", "indy_ladder", "responsive", "aero_sensitive"],
        "advice_style": "last step before IndyCar, manage the aero"
    },
    "usf2000": {
        "name": "USF 2000",
        "class": "open_wheel",
        "power": "low",
        "braking": "light",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["momentum_car", "indy_ladder", "forgiving", "lightweight"],
        "advice_style": "carry momentum, good starting point"
    },
    "pm18": {
        "name": "Pro Mazda",
        "class": "open_wheel",
        "power": "medium",
        "braking": "medium",
        "aero": "medium",
        "tire_deg": "low",
        "traits": ["balanced", "indy_ladder", "forgiving", "good_stepping_stone"],
        "advice_style": "balance of grip and speed, prep for higher formulas"
    },

    # =========================================================================
    # FORMULA / OPEN WHEEL - ENTRY LEVEL
    # =========================================================================
    "skip_barber": {
        "name": "Skip Barber Formula 2000",
        "class": "open_wheel",
        "power": "low",
        "braking": "light",
        "aero": "low",
        "tire_deg": "low",
        "traits": ["momentum_car", "no_downforce", "forgiving", "equal_racing"],
        "advice_style": "pure momentum, no aero to rely on, smooth is fast"
    },
    "formula_vee": {
        "name": "Formula Vee",
        "class": "open_wheel",
        "power": "very_low",
        "braking": "very_light",
        "aero": "none",
        "tire_deg": "very_low",
        "traits": ["momentum_car", "no_downforce", "draft_dependent", "equal_racing"],
        "advice_style": "pure momentum and drafting, no aero at all"
    },
    "ray_ff1600": {
        "name": "Ray FF1600",
        "class": "open_wheel",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "low",
        "traits": ["momentum_car", "rear_wheel_drive", "lightweight", "responsive"],
        "advice_style": "classic formula feel, momentum is key"
    },

    # =========================================================================
    # HISTORIC FORMULA
    # =========================================================================
    "lotus_49": {
        "name": "Lotus 49",
        "class": "historic_f1",
        "power": "high",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "medium",
        "traits": ["no_downforce", "high_skill", "tail_happy", "historic"],
        "advice_style": "no aero, pure car control, smooth throttle essential"
    },
    "lotus_79": {
        "name": "Lotus 79",
        "class": "historic_f1",
        "power": "high",
        "braking": "light",
        "aero": "ground_effect",
        "tire_deg": "medium",
        "traits": ["ground_effect", "aero_sensitive", "challenging", "historic"],
        "advice_style": "first ground effect car, keep it planted, respect the aero"
    },

    # =========================================================================
    # RALLYCROSS (Road-capable)
    # =========================================================================
    "fia_cross_car": {
        "name": "FIA Cross Car",
        "class": "rallycross",
        "power": "low",
        "braking": "light",
        "aero": "minimal",
        "tire_deg": "medium",
        "traits": ["rear_wheel_drive", "lightweight", "agile", "mixed_surface"],
        "advice_style": "balance throttle on mixed surfaces, commit to jumps"
    },
}

# Mapping from iRacing car names to metadata keys
_CAR_NAME_MAP: Dict[str, str] = {
    # =========================================================================
    # PRODUCTION / ENTRY LEVEL
    # =========================================================================
    "mazda mx-5 cup": "mx5_cup",
    "mx-5 cup": "mx5_cup",
    "global mazda mx-5 cup": "mx5_cup",
    "bmw m2 cs racing": "bmw_m2_csr",
    "bmw m2 csr": "bmw_m2_csr",
    "ford mustang fr500s": "ford_mustang_fr500s",
    "mustang fr500s": "ford_mustang_fr500s",
    "pontiac solstice": "pontiac_solstice",
    "cadillac cts-v racecar": "cadillac_cts_v",
    "cadillac cts-v": "cadillac_cts_v",
    "cts-v": "cadillac_cts_v",
    "scca spec racer ford": "spec_racer_ford",
    "spec racer ford": "spec_racer_ford",
    "srf": "spec_racer_ford",
    "ruf rt 12r": "ruf_rt12r",
    "ruf rt12r": "ruf_rt12r",
    "ruf rt 12r track": "ruf_rt12r_track",
    "ruf rt12r track": "ruf_rt12r_track",
    "volkswagen jetta tdi cup": "vw_jetta_tdi",
    "vw jetta tdi": "vw_jetta_tdi",
    "jetta tdi cup": "vw_jetta_tdi",
    "kia optima": "kia_optima",

    # =========================================================================
    # TCR / TOURING CARS
    # =========================================================================
    "audi rs 3 lms": "audi_rs3_lms",
    "audi rs3 lms": "audi_rs3_lms",
    "audi rs 3 lms tcr": "audi_rs3_lms",
    "honda civic type r tcr": "honda_civic_type_r_tcr",
    "civic type r tcr": "honda_civic_type_r_tcr",
    "hyundai elantra n tcr": "hyundai_elantra_n_tcr",
    "elantra n tcr": "hyundai_elantra_n_tcr",
    "hyundai veloster n tcr": "hyundai_veloster_n_tcr",
    "veloster n tcr": "hyundai_veloster_n_tcr",

    # =========================================================================
    # SUPERCARS
    # =========================================================================
    "supercars ford mustang gt": "supercars_ford_mustang",
    "ford mustang gt supercars": "supercars_ford_mustang",
    "supercars chevrolet camaro zl1": "supercars_chevy_camaro",
    "chevrolet camaro zl1 supercars": "supercars_chevy_camaro",
    "camaro zl1": "supercars_chevy_camaro",

    # =========================================================================
    # GT4 CARS
    # =========================================================================
    "aston martin vantage gt4": "aston_martin_vantage_gt4",
    "vantage gt4": "aston_martin_vantage_gt4",
    "bmw m4 gt4": "bmw_m4_gt4",
    "bmw m4 g82 gt4 evo": "bmw_m4_gt4",
    "bmw m4 f82 gt4": "bmw_m4_gt4",
    "ford mustang gt4": "ford_mustang_gt4",
    "mustang gt4": "ford_mustang_gt4",
    "mclaren 570s gt4": "mclaren_570s_gt4",
    "570s gt4": "mclaren_570s_gt4",
    "mercedes-amg gt4": "mercedes_amg_gt4",
    "mercedes amg gt4": "mercedes_amg_gt4",
    "porsche 718 cayman gt4 clubsport mr": "porsche_718_cayman_gt4",
    "porsche 718 cayman gt4 clubsport": "porsche_718_cayman_gt4",
    "porsche cayman gt4": "porsche_718_cayman_gt4",
    "718 cayman gt4": "porsche_718_cayman_gt4",

    # =========================================================================
    # GT3 CARS
    # =========================================================================
    "acura nsx gt3 evo 22": "acura_nsx_gt3",
    "acura nsx gt3": "acura_nsx_gt3",
    "nsx gt3": "acura_nsx_gt3",
    "aston martin vantage gt3 evo": "aston_martin_vantage_gt3",
    "aston martin vantage gt3": "aston_martin_vantage_gt3",
    "vantage gt3": "aston_martin_vantage_gt3",
    "audi r8 lms evo ii gt3": "audi_r8_lms_gt3",
    "audi r8 lms gt3": "audi_r8_lms_gt3",
    "audi r8 gt3": "audi_r8_lms_gt3",
    "r8 lms gt3": "audi_r8_lms_gt3",
    "bmw m4 gt3": "bmw_m4_gt3",
    "bmw m4 gt3 evo": "bmw_m4_gt3",
    "chevrolet corvette z06 gt3.r": "corvette_z06_gt3r",
    "corvette z06 gt3.r": "corvette_z06_gt3r",
    "corvette gt3": "corvette_z06_gt3r",
    "z06 gt3.r": "corvette_z06_gt3r",
    "ferrari 296 gt3": "ferrari_296_gt3",
    "296 gt3": "ferrari_296_gt3",
    "ferrari 488 gt3 evo 2020": "ferrari_488_gt3",
    "ferrari 488 gt3 evo": "ferrari_488_gt3",
    "ferrari 488 gt3": "ferrari_488_gt3",
    "488 gt3": "ferrari_488_gt3",
    "ford mustang gt3": "ford_mustang_gt3",
    "mustang gt3": "ford_mustang_gt3",
    "lamborghini huracán gt3 evo": "lamborghini_huracan_gt3",
    "lamborghini huracan gt3 evo": "lamborghini_huracan_gt3",
    "lamborghini huracan gt3": "lamborghini_huracan_gt3",
    "huracan gt3": "lamborghini_huracan_gt3",
    "mclaren 720s gt3 evo": "mclaren_720s_gt3",
    "mclaren 720s gt3": "mclaren_720s_gt3",
    "720s gt3": "mclaren_720s_gt3",
    "mercedes-amg gt3 2020": "mercedes_amg_gt3",
    "mercedes-amg gt3": "mercedes_amg_gt3",
    "mercedes amg gt3 2020": "mercedes_amg_gt3",
    "mercedes amg gt3": "mercedes_amg_gt3",
    "amg gt3": "mercedes_amg_gt3",
    "porsche 911 gt3 r (992)": "porsche_911_gt3_r",
    "porsche 911 gt3 r": "porsche_911_gt3_r",
    "911 gt3 r": "porsche_911_gt3_r",

    # =========================================================================
    # CUP / ONE-MAKE SERIES
    # =========================================================================
    "porsche 911 gt3 cup (992)": "porsche_911_gt3_cup_992",
    "porsche 911 gt3 cup": "porsche_911_gt3_cup_992",
    "911 gt3 cup": "porsche_911_gt3_cup_992",
    "porsche 911 cup (992.2)": "porsche_911_gt3_cup_992_2",
    "porsche 992.2 cup": "porsche_911_gt3_cup_992_2",
    "911 cup 992.2": "porsche_911_gt3_cup_992_2",

    # =========================================================================
    # GTE CARS
    # =========================================================================
    "bmw m8 gte": "bmw_m8_gte",
    "m8 gte": "bmw_m8_gte",
    "chevrolet corvette c8.r": "corvette_c8r",
    "corvette c8.r": "corvette_c8r",
    "c8.r": "corvette_c8r",
    "ferrari 488 gte": "ferrari_488_gte",
    "488 gte": "ferrari_488_gte",
    "ford gt gte": "ford_gt_gte",
    "ford gt": "ford_gt_gte",
    "porsche 911 rsr": "porsche_911_rsr",
    "911 rsr": "porsche_911_rsr",

    # =========================================================================
    # GTP / LMDH / HYPERCAR
    # =========================================================================
    "acura arx-06 gtp": "acura_arx_06",
    "acura arx-06": "acura_arx_06",
    "arx-06": "acura_arx_06",
    "bmw m hybrid v8": "bmw_m_hybrid_v8",
    "m hybrid v8": "bmw_m_hybrid_v8",
    "cadillac v-series.r gtp": "cadillac_v_series_r",
    "cadillac v-series.r": "cadillac_v_series_r",
    "v-series.r": "cadillac_v_series_r",
    "ferrari 499p": "ferrari_499p",
    "499p": "ferrari_499p",
    "porsche 963 gtp": "porsche_963",
    "porsche 963": "porsche_963",

    # =========================================================================
    # LMP2 / LMP3 / PROTOTYPE
    # =========================================================================
    "dallara p217 lmp2": "dallara_p217",
    "dallara p217": "dallara_p217",
    "p217": "dallara_p217",
    "ligier js p320": "ligier_js_p320",
    "js p320": "ligier_js_p320",
    "radical sr10": "radical_sr10",
    "sr10": "radical_sr10",
    "radical sr8": "radical_sr8",
    "sr8": "radical_sr8",

    # =========================================================================
    # FORMULA / OPEN WHEEL - F1 LEVEL
    # =========================================================================
    "mercedes-amg w13 e performance": "mercedes_w13",
    "mercedes-amg f1 w13 e performance": "mercedes_w13",
    "mercedes w13": "mercedes_w13",
    "w13": "mercedes_w13",
    "mercedes-amg w12 e performance": "mercedes_w12",
    "mercedes-amg f1 w12 e performance": "mercedes_w12",
    "mercedes w12": "mercedes_w12",
    "w12": "mercedes_w12",
    "mclaren mp4-30": "mclaren_mp4_30",
    "mp4-30": "mclaren_mp4_30",
    "williams-toyota fw31": "williams_fw31",
    "williams fw31": "williams_fw31",
    "fw31": "williams_fw31",

    # =========================================================================
    # FORMULA / OPEN WHEEL - INDYCAR
    # =========================================================================
    "dallara ir18": "dallara_ir18",
    "ir18": "dallara_ir18",
    "indycar": "dallara_ir18",
    "dallara ir-01": "dallara_ir01",
    "dallara ir01": "dallara_ir01",
    "ir-01": "dallara_ir01",
    "ir01": "dallara_ir01",

    # =========================================================================
    # FORMULA / OPEN WHEEL - SUPER FORMULA
    # =========================================================================
    "super formula sf23 - honda": "super_formula_sf23_honda",
    "super formula sf23 honda": "super_formula_sf23_honda",
    "sf23 honda": "super_formula_sf23_honda",
    "super formula sf23 - toyota": "super_formula_sf23_toyota",
    "super formula sf23 toyota": "super_formula_sf23_toyota",
    "sf23 toyota": "super_formula_sf23_toyota",
    "super formula lights": "super_formula_lights",

    # =========================================================================
    # FORMULA / OPEN WHEEL - JUNIOR FORMULA
    # =========================================================================
    "dallara f3": "dallara_f3",
    "f3": "dallara_f3",
    "fia f4": "fia_f4",
    "f4": "fia_f4",
    "formula renault 3.5": "formula_renault_3_5",
    "fr 3.5": "formula_renault_3_5",
    "formula renault 2.0": "formula_renault_2_0",
    "fr 2.0": "formula_renault_2_0",

    # =========================================================================
    # FORMULA / OPEN WHEEL - INDY LADDER
    # =========================================================================
    "indy pro 2000": "indy_pro_2000",
    "ip2000": "indy_pro_2000",
    "dallara il-15": "dallara_il15",
    "dallara il15": "dallara_il15",
    "il-15": "dallara_il15",
    "indy lights": "dallara_il15",
    "usf 2000": "usf2000",
    "usf2000": "usf2000",
    "pro mazda": "pm18",
    "pm-18": "pm18",

    # =========================================================================
    # FORMULA / OPEN WHEEL - ENTRY LEVEL
    # =========================================================================
    "skip barber formula 2000": "skip_barber",
    "skip barber": "skip_barber",
    "skippy": "skip_barber",
    "formula vee": "formula_vee",
    "vee": "formula_vee",
    "ray ff1600": "ray_ff1600",
    "ff1600": "ray_ff1600",

    # =========================================================================
    # HISTORIC FORMULA
    # =========================================================================
    "lotus 49": "lotus_49",
    "lotus 79": "lotus_79",

    # =========================================================================
    # RALLYCROSS
    # =========================================================================
    "fia cross car": "fia_cross_car",
    "cross car": "fia_cross_car",
}


# =============================================================================
# TRACK METADATA
# =============================================================================

TRACKS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # EUROPEAN ROAD COURSES
    # =========================================================================
    "monza": {
        "name": "Monza",
        "full_name": "Autodromo Nazionale Monza",
        "type": "high_speed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.15: "Variante del Rettifilo",
            0.35: "Curva Grande",
            0.48: "Variante della Roggia",
            0.65: "Lesmo 1",
            0.70: "Lesmo 2",
            0.85: "Ascari",
            0.95: "Parabolica"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "long straights, hard braking, rear tire stress in Lesmos"
    },
    "spa": {
        "name": "Spa",
        "full_name": "Circuit de Spa-Francorchamps",
        "type": "mixed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.05: "La Source",
            0.15: "Eau Rouge",
            0.30: "Les Combes",
            0.55: "Pouhon",
            0.70: "Fagnes",
            0.85: "Blanchimont",
            0.95: "Bus Stop"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "elevation changes, high speed, weather variable"
    },
    "redbull_ring": {
        "name": "Red Bull Ring",
        "full_name": "Red Bull Ring",
        "type": "high_speed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.10: "Turn 1",
            0.20: "Turn 2",
            0.35: "Turn 3",
            0.55: "Turn 4",
            0.70: "Turn 5-6",
            0.85: "Turn 7",
            0.95: "Turn 8-9"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "short track, heavy braking, long straights between corners"
    },
    "mugello": {
        "name": "Mugello",
        "full_name": "Autodromo del Mugello",
        "type": "mixed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "San Donato",
            0.18: "Luco",
            0.28: "Poggio Secco",
            0.40: "Materassi",
            0.50: "Borgo San Lorenzo",
            0.62: "Casanova",
            0.72: "Savelli",
            0.82: "Arrabbiata 1",
            0.88: "Arrabbiata 2",
            0.95: "Scarperia"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "flowing corners, elevation changes, high commitment"
    },
    "silverstone": {
        "name": "Silverstone",
        "full_name": "Silverstone Circuit",
        "type": "high_speed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.05: "Abbey",
            0.12: "Farm",
            0.18: "Village",
            0.25: "The Loop",
            0.35: "Aintree",
            0.42: "Wellington Straight",
            0.50: "Brooklands",
            0.55: "Luffield",
            0.62: "Copse",
            0.72: "Maggotts",
            0.78: "Becketts",
            0.82: "Chapel",
            0.90: "Stowe",
            0.97: "Club"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "fast flowing corners, Maggotts-Becketts complex, high commitment"
    },
    "nurburgring_gp": {
        "name": "Nürburgring GP",
        "full_name": "Nürburgring Grand-Prix-Strecke",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "Turn 2-3",
            0.28: "Ford Kurve",
            0.38: "Dunlop Kehre",
            0.48: "Michael Schumacher S",
            0.60: "Bit Kurve",
            0.72: "RTL Kurve",
            0.85: "Coca-Cola Kurve",
            0.95: "Veedol Schikane"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "modern F1 circuit, technical sections, elevation changes"
    },
    "nurburgring_nordschleife": {
        "name": "Nordschleife",
        "full_name": "Nürburgring Nordschleife",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.03: "Hatzenbach",
            0.08: "Flugplatz",
            0.12: "Schwedenkreuz",
            0.18: "Fuchsröhre",
            0.25: "Adenauer Forst",
            0.32: "Metzgesfeld",
            0.38: "Kallenhard",
            0.45: "Wehrseifen",
            0.52: "Ex-Mühle",
            0.58: "Bergwerk",
            0.65: "Kesselchen",
            0.72: "Klostertal",
            0.78: "Caracciola Karussell",
            0.85: "Hohe Acht",
            0.90: "Brünnchen",
            0.95: "Pflanzgarten",
            0.98: "Schwalbenschwanz"
        },
        "braking_severity": "medium",
        "fuel_consumption": "high",
        "characteristics": "the Green Hell, 20.8km, 154 corners, extreme elevation changes"
    },
    "nurburgring_combined": {
        "name": "Nürburgring Combined",
        "full_name": "Nürburgring Combined",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.02: "GP Turn 1",
            0.05: "Hatzenbach",
            0.10: "Flugplatz",
            0.50: "Caracciola Karussell",
            0.70: "Pflanzgarten",
            0.90: "Schwalbenschwanz",
            0.95: "GP Section"
        },
        "braking_severity": "medium",
        "fuel_consumption": "high",
        "characteristics": "full 24h layout, Nordschleife + GP circuit combined"
    },
    "imola": {
        "name": "Imola",
        "full_name": "Autodromo Enzo e Dino Ferrari",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Tamburello",
            0.18: "Villeneuve",
            0.28: "Tosa",
            0.40: "Piratella",
            0.52: "Acque Minerali",
            0.65: "Variante Alta",
            0.80: "Rivazza 1",
            0.88: "Rivazza 2"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "old school layout, heavy braking zones, elevation changes"
    },
    "barcelona": {
        "name": "Barcelona",
        "full_name": "Circuit de Barcelona-Catalunya",
        "type": "mixed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.05: "Elf",
            0.15: "Renault",
            0.22: "Repsol",
            0.32: "Seat",
            0.42: "Campsa",
            0.52: "La Caixa",
            0.62: "Banc Sabadell",
            0.75: "New Holland",
            0.88: "Europcar"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "tire degradation track, Turn 3 high speed, technical final sector"
    },
    "hungaroring": {
        "name": "Hungaroring",
        "full_name": "Hungaroring",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "Turn 2",
            0.28: "Turn 3",
            0.38: "Turn 4",
            0.48: "Turn 5-6",
            0.58: "Turn 7-8",
            0.70: "Turn 9-10",
            0.82: "Turn 11",
            0.92: "Turn 12-14"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "twisty, low speed, overtaking difficult, hot conditions"
    },
    "zandvoort": {
        "name": "Zandvoort",
        "full_name": "Circuit Park Zandvoort",
        "type": "technical",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "Tarzanbocht",
            0.18: "Gerlachbocht",
            0.30: "Hugenholzbocht",
            0.42: "Hunserug",
            0.55: "Rob Slotemaker",
            0.68: "Scheivlak",
            0.78: "Mastersbocht",
            0.88: "Hans Ernst",
            0.95: "Arie Luyendyk"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "banked corners, narrow, sand traps, old school character"
    },
    "portimao": {
        "name": "Portimão",
        "full_name": "Autódromo Internacional do Algarve",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "Turn 3",
            0.28: "Turn 4-5",
            0.42: "Turn 6-7",
            0.55: "Turn 8",
            0.68: "Turn 9-10",
            0.78: "Turn 11",
            0.88: "Turn 13",
            0.95: "Turn 14-15"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "roller coaster elevation, blind crests, tricky braking zones"
    },
    "hockenheim": {
        "name": "Hockenheim",
        "full_name": "Hockenheimring Baden-Württemberg",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Nordkurve",
            0.22: "Einfahrt Parabolika",
            0.35: "Parabolika",
            0.48: "Spitzkehre",
            0.62: "Mobil 1 Kurve",
            0.78: "Sachs Kurve",
            0.92: "Südkurve"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "stadium section, heavy braking into hairpin, high speed straights"
    },
    "misano": {
        "name": "Misano",
        "full_name": "Misano World Circuit Marco Simoncelli",
        "type": "technical",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "Variante del Parco",
            0.20: "Curva del Rio",
            0.32: "Variante del Carro",
            0.45: "Curva della Quercia",
            0.58: "Curvone",
            0.72: "Tramonto",
            0.85: "Carro 2"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "MotoGP circuit, tight and technical, anticlockwise"
    },
    "le_mans": {
        "name": "Le Mans",
        "full_name": "Circuit des 24 Heures du Mans",
        "type": "high_speed",
        "tire_stress": "balanced",
        "key_corners": {
            0.03: "Dunlop Chicane",
            0.08: "Dunlop Curve",
            0.15: "Esses",
            0.22: "Tertre Rouge",
            0.55: "Mulsanne Corner",
            0.70: "Indianapolis",
            0.80: "Arnage",
            0.88: "Porsche Curves",
            0.95: "Ford Chicane"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "13.6km circuit, Mulsanne Straight, endurance classic"
    },
    "donington": {
        "name": "Donington Park",
        "full_name": "Donington Park Racing Circuit",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Redgate",
            0.18: "Craner Curves",
            0.30: "Old Hairpin",
            0.45: "Starkeys",
            0.58: "Schwantz Curve",
            0.70: "McLeans",
            0.82: "Coppice",
            0.92: "Goddards"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "historic circuit, Craner Curves iconic, elevation changes"
    },
    "brands_hatch": {
        "name": "Brands Hatch",
        "full_name": "Brands Hatch Circuit",
        "type": "technical",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "Paddock Hill Bend",
            0.18: "Druids",
            0.32: "Graham Hill Bend",
            0.45: "Surtees",
            0.58: "Hawthorn Bend",
            0.70: "Westfield",
            0.82: "Sheene Curve",
            0.92: "Clearways"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "amphitheater layout, Paddock Hill iconic, short and intense"
    },
    "oulton_park": {
        "name": "Oulton Park",
        "full_name": "Oulton Park Circuit",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Old Hall",
            0.18: "Dentons",
            0.30: "Cascades",
            0.42: "Island",
            0.55: "Shell Oils Corner",
            0.68: "Knickerbrook",
            0.80: "Clay Hill",
            0.92: "Druids"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "undulating, natural terrain, classic British circuit"
    },
    "snetterton": {
        "name": "Snetterton",
        "full_name": "Snetterton Circuit",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.10: "Riches",
            0.25: "Montreal",
            0.38: "Palmer",
            0.52: "Agostini",
            0.65: "Hamilton",
            0.78: "Oggies",
            0.90: "Coram"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "former airfield, long straights, challenging corners"
    },
    "knockhill": {
        "name": "Knockhill",
        "full_name": "Knockhill Racing Circuit",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.10: "Duffus Dip",
            0.25: "Leslie's",
            0.40: "Clark's",
            0.55: "Railway",
            0.70: "McIntyres",
            0.85: "Scotsman"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "Scottish circuit, short and tight, significant elevation"
    },

    # =========================================================================
    # NORTH AMERICAN ROAD COURSES
    # =========================================================================
    "lime_rock": {
        "name": "Lime Rock Park",
        "full_name": "Lime Rock Park",
        "type": "technical",
        "tire_stress": "front_limited",
        "key_corners": {
            0.12: "Big Bend",
            0.30: "Left Hander",
            0.45: "The Diving Turn",
            0.60: "The Uphill",
            0.80: "West Bend",
            0.95: "The Downhill"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "short track, no straights, high commitment corners"
    },
    "road_america": {
        "name": "Road America",
        "full_name": "Road America",
        "type": "high_speed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.15: "Turn 3",
            0.25: "Turn 5",
            0.35: "The Kink",
            0.48: "Turn 8 (Canada Corner)",
            0.60: "Carousel",
            0.75: "Turn 12",
            0.85: "Turn 13",
            0.95: "Turn 14"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "4 miles long, The Kink flat out, high speed sweepers"
    },
    "road_atlanta": {
        "name": "Road Atlanta",
        "full_name": "Road Atlanta",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.20: "The Esses",
            0.35: "Turn 5",
            0.50: "Turn 6",
            0.62: "Turn 7",
            0.75: "Turn 10a",
            0.85: "Turn 10b",
            0.95: "Turn 12"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "elevation changes, tricky Esses, blind corners"
    },
    "watkins_glen": {
        "name": "Watkins Glen",
        "full_name": "Watkins Glen International",
        "type": "high_speed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "The Esses",
            0.30: "The Back Straight",
            0.45: "The Inner Loop",
            0.58: "Toe of the Boot",
            0.70: "The Boot",
            0.85: "The Bus Stop",
            0.95: "Turn 11"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "classic American road course, Boot section iconic, flowing"
    },
    "laguna_seca": {
        "name": "Laguna Seca",
        "full_name": "WeatherTech Raceway Laguna Seca",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Andretti Hairpin",
            0.20: "Turn 2",
            0.32: "Turn 3",
            0.45: "Turn 4",
            0.55: "Turn 5",
            0.65: "Turn 6",
            0.75: "The Corkscrew",
            0.85: "Rainey Curve",
            0.95: "Turn 11"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "The Corkscrew iconic, elevation changes, technical"
    },
    "cota": {
        "name": "COTA",
        "full_name": "Circuit of the Americas",
        "type": "mixed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.05: "Turn 1",
            0.12: "Turn 2-3-4",
            0.20: "Turn 5-6",
            0.30: "Turn 7-8-9",
            0.42: "Turn 10",
            0.50: "Turn 11",
            0.60: "Turn 12-13-14-15",
            0.75: "Turn 16-17-18",
            0.90: "Turn 19",
            0.95: "Turn 20"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "Turn 1 uphill braking, flowing esses, Austin Texas"
    },
    "mid_ohio": {
        "name": "Mid-Ohio",
        "full_name": "Mid-Ohio Sports Car Course",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.20: "Turn 2-3",
            0.32: "Turn 4 (Keyhole)",
            0.45: "Turn 5",
            0.55: "Turn 6",
            0.65: "Thunder Valley",
            0.78: "Turn 9",
            0.88: "Turn 10-11",
            0.95: "Turn 12-13"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "tight and technical, Keyhole challenging, natural terrain"
    },
    "vir": {
        "name": "VIR",
        "full_name": "Virginia International Raceway",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.15: "Turn 2-3",
            0.28: "Oak Tree",
            0.42: "Climbing Esses",
            0.55: "Hog Pen",
            0.68: "Roller Coaster",
            0.82: "Oak Tree 2",
            0.92: "Turn 17"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "beautiful natural terrain, multiple configurations, challenging"
    },
    "sebring": {
        "name": "Sebring",
        "full_name": "Sebring International Raceway",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "Turn 3",
            0.30: "Hairpin",
            0.42: "Sunset Bend",
            0.55: "Turn 7",
            0.65: "Fangio Chicane",
            0.78: "Turn 13",
            0.88: "Turn 15-16",
            0.95: "Turn 17"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "bumpy concrete surface, 12 Hours endurance classic"
    },
    "long_beach": {
        "name": "Long Beach",
        "full_name": "Long Beach Street Circuit",
        "type": "street",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.20: "Turn 2",
            0.35: "Turn 3-4",
            0.50: "Fountain Turn",
            0.65: "Turn 8",
            0.78: "Turn 9",
            0.90: "Turn 11"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "street circuit, no runoff, IndyCar classic venue"
    },
    "sonoma": {
        "name": "Sonoma",
        "full_name": "Sonoma Raceway",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 2",
            0.20: "Turn 3",
            0.35: "Turn 4",
            0.48: "Turn 5",
            0.60: "Turn 6",
            0.72: "Turn 7",
            0.82: "The Carousel",
            0.92: "Turn 10-11"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "Wine Country, elevation changes, NASCAR road course"
    },
    "willow_springs": {
        "name": "Willow Springs",
        "full_name": "Willow Springs International Motorsports Park",
        "type": "high_speed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.10: "Turn 1",
            0.25: "Turn 2",
            0.40: "Turn 3",
            0.55: "Turn 4",
            0.65: "Turn 5",
            0.75: "Turn 6",
            0.85: "Turn 7",
            0.95: "Turn 8-9"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "high speed, flowing layout, desert location"
    },
    "montreal": {
        "name": "Montreal",
        "full_name": "Circuit Gilles Villeneuve",
        "type": "street",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Turn 1-2",
            0.20: "Turn 3",
            0.35: "Turn 4-5",
            0.48: "Turn 6-7",
            0.60: "Turn 8-9",
            0.75: "Turn 10",
            0.85: "Wall of Champions",
            0.95: "Turn 13-14"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "semi-permanent street circuit, Wall of Champions infamous"
    },

    # =========================================================================
    # JAPANESE / ASIAN ROAD COURSES
    # =========================================================================
    "suzuka": {
        "name": "Suzuka",
        "full_name": "Suzuka International Racing Course",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.05: "First Curve",
            0.12: "S Curves",
            0.22: "Dunlop Curve",
            0.30: "Degner 1",
            0.35: "Degner 2",
            0.45: "Hairpin",
            0.55: "Spoon Curve",
            0.70: "130R",
            0.85: "Casio Triangle",
            0.95: "Chicane"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "figure-8 layout, 130R legendary, Esses technical"
    },
    "fuji": {
        "name": "Fuji",
        "full_name": "Fuji Speedway",
        "type": "high_speed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.20: "Coca-Cola Corner",
            0.32: "100R",
            0.45: "Hairpin",
            0.58: "300R",
            0.70: "Dunlop Corner",
            0.82: "Turn 13",
            0.92: "Final Corner"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "long main straight, heavy braking Turn 1, Mount Fuji views"
    },
    "okayama": {
        "name": "Okayama",
        "full_name": "Okayama International Circuit",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.10: "First Corner",
            0.22: "Moss Corner",
            0.35: "Williams Corner",
            0.48: "Attwood Curve",
            0.60: "Revolver Corner",
            0.75: "Piper Corner",
            0.90: "Redman Corner"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "tight and technical, elevation changes, tricky surface"
    },
    "tsukuba": {
        "name": "Tsukuba",
        "full_name": "Tsukuba Circuit",
        "type": "technical",
        "tire_stress": "balanced",
        "key_corners": {
            0.10: "First Corner",
            0.25: "Turn 2",
            0.40: "Turn 3-4",
            0.55: "Turn 5",
            0.70: "Dunlop Corner",
            0.85: "Final Corner"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "short technical circuit, time attack favorite"
    },
    "motegi": {
        "name": "Motegi",
        "full_name": "Twin Ring Motegi",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1-2",
            0.20: "Turn 3",
            0.35: "Victory Corner",
            0.50: "Turn 5-6",
            0.65: "90 Degree Corner",
            0.78: "V Corner",
            0.90: "Downhill Turn"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "Honda test facility, road course and oval combination"
    },

    # =========================================================================
    # AUSTRALIAN ROAD COURSES
    # =========================================================================
    "bathurst": {
        "name": "Bathurst",
        "full_name": "Mount Panorama Circuit",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.05: "Hell Corner",
            0.15: "Griffins Bend",
            0.25: "The Cutting",
            0.35: "Sulman Park",
            0.45: "McPhillamy Park",
            0.55: "Skyline",
            0.65: "The Dipper",
            0.75: "Forrest Elbow",
            0.85: "Conrod Straight",
            0.95: "The Chase"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "iconic mountain circuit, extreme elevation, The Mountain"
    },
    "phillip_island": {
        "name": "Phillip Island",
        "full_name": "Phillip Island Grand Prix Circuit",
        "type": "high_speed",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "Doohan Corner",
            0.20: "Southern Loop",
            0.35: "Stoner Corner",
            0.48: "Honda Corner",
            0.60: "Siberia",
            0.75: "Lukey Heights",
            0.88: "MG Corner"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "fast and flowing, coastal winds, MotoGP venue"
    },
    "adelaide": {
        "name": "Adelaide",
        "full_name": "Adelaide Street Circuit",
        "type": "street",
        "tire_stress": "front_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "Turn 2-3",
            0.30: "Senna Chicane",
            0.42: "Turn 6",
            0.55: "Turn 8",
            0.68: "Turn 9",
            0.80: "Turn 11",
            0.92: "Turn 14"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "Supercars street circuit, tight and technical"
    },

    # =========================================================================
    # SOUTH AMERICAN ROAD COURSES
    # =========================================================================
    "interlagos": {
        "name": "Interlagos",
        "full_name": "Autódromo José Carlos Pace",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.05: "Senna S",
            0.15: "Curva do Sol",
            0.28: "Reta Oposta",
            0.40: "Descida do Lago",
            0.55: "Ferradura",
            0.70: "Laranjinha",
            0.82: "Pinheirinho",
            0.92: "Mergulho",
            0.98: "Juncao"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "anticlockwise, elevation changes, Brazilian GP venue"
    },

    # =========================================================================
    # MIDDLE EAST ROAD COURSES
    # =========================================================================
    "miami": {
        "name": "Miami",
        "full_name": "Miami International Autodrome",
        "type": "street",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.05: "Turn 1",
            0.12: "Turn 2-3",
            0.22: "Turn 4-5",
            0.32: "Turn 6",
            0.42: "Turn 7",
            0.52: "Turn 8-9-10",
            0.65: "Turn 11",
            0.75: "Turn 12-13",
            0.85: "Turn 14-15",
            0.95: "Turn 16-17"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "F1 street circuit, Hard Rock Stadium venue, 19 turns"
    },

    # =========================================================================
    # NASCAR / INDYCAR ROAD COURSE CONFIGURATIONS
    # =========================================================================
    "daytona_road": {
        "name": "Daytona Road Course",
        "full_name": "Daytona International Speedway - Road Course",
        "type": "mixed",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.08: "Turn 1",
            0.18: "International Horseshoe",
            0.32: "Turn 3",
            0.45: "Infield Section",
            0.60: "Bus Stop",
            0.75: "NASCAR Turn 1",
            0.90: "West Banking"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "infield road course with banking, 24 Hours of Daytona"
    },
    "charlotte_roval": {
        "name": "Charlotte Roval",
        "full_name": "Charlotte Motor Speedway - Roval",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.20: "Infield Turn 2-3",
            0.35: "Turn 4-5",
            0.50: "Turn 6",
            0.65: "Backstretch Chicane",
            0.80: "Turn 9-10",
            0.92: "Turn 11-12"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "NASCAR playoff track, combines oval and infield"
    },
    "indianapolis_road": {
        "name": "Indianapolis Road Course",
        "full_name": "Indianapolis Motor Speedway - Road Course",
        "type": "mixed",
        "tire_stress": "balanced",
        "key_corners": {
            0.08: "Turn 1",
            0.20: "Turn 2-3-4",
            0.35: "Turn 5-6",
            0.50: "Turn 7",
            0.65: "Turn 8-9-10",
            0.80: "Turn 11-12",
            0.92: "Turn 13-14"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "F1 US GP venue, uses part of famous oval"
    },

    # =========================================================================
    # SUPERSPEEDWAYS (OVAL)
    # =========================================================================
    "daytona_oval": {
        "name": "Daytona",
        "full_name": "Daytona International Speedway",
        "type": "superspeedway",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "none",
        "fuel_consumption": "high",
        "characteristics": "2.5 miles, 31 degree banking, restrictor plate racing, drafting critical"
    },
    "talladega": {
        "name": "Talladega",
        "full_name": "Talladega Superspeedway",
        "type": "superspeedway",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "none",
        "fuel_consumption": "high",
        "characteristics": "2.66 miles, 33 degree banking, fastest NASCAR track, pack racing"
    },

    # =========================================================================
    # INTERMEDIATE SPEEDWAYS (OVAL)
    # =========================================================================
    "charlotte_oval": {
        "name": "Charlotte",
        "full_name": "Charlotte Motor Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, 24 degree banking, Coca-Cola 600"
    },
    "atlanta_oval": {
        "name": "Atlanta",
        "full_name": "Atlanta Motor Speedway",
        "type": "superspeedway",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "none",
        "fuel_consumption": "high",
        "characteristics": "1.54 miles, 28 degree banking, repaved for pack racing"
    },
    "texas_oval": {
        "name": "Texas",
        "full_name": "Texas Motor Speedway",
        "type": "intermediate",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, variable banking, high speeds"
    },
    "las_vegas_oval": {
        "name": "Las Vegas",
        "full_name": "Las Vegas Motor Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, 20 degree banking, desert location"
    },
    "kansas_oval": {
        "name": "Kansas",
        "full_name": "Kansas Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, progressive banking, wide racing groove"
    },
    "michigan_oval": {
        "name": "Michigan",
        "full_name": "Michigan International Speedway",
        "type": "intermediate",
        "tire_stress": "front_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "2 miles, 18 degree banking, wide and fast"
    },
    "auto_club_oval": {
        "name": "Auto Club",
        "full_name": "Auto Club Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "2 miles, D-shaped, California"
    },
    "homestead_oval": {
        "name": "Homestead-Miami",
        "full_name": "Homestead-Miami Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, variable banking, championship finale venue"
    },
    "chicagoland_oval": {
        "name": "Chicagoland",
        "full_name": "Chicagoland Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, 18 degree banking"
    },
    "kentucky_oval": {
        "name": "Kentucky",
        "full_name": "Kentucky Speedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.5 miles, progressive banking"
    },
    "nashville_oval": {
        "name": "Nashville Superspeedway",
        "full_name": "Nashville Superspeedway",
        "type": "intermediate",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "1.33 miles, concrete surface"
    },

    # =========================================================================
    # SHORT TRACKS (OVAL)
    # =========================================================================
    "bristol_oval": {
        "name": "Bristol",
        "full_name": "Bristol Motor Speedway",
        "type": "short_track",
        "tire_stress": "front_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "low",
        "characteristics": "0.533 miles, 26-30 degree banking, Thunder Valley, contact racing"
    },
    "martinsville_oval": {
        "name": "Martinsville",
        "full_name": "Martinsville Speedway",
        "type": "short_track",
        "tire_stress": "front_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "very_heavy",
        "fuel_consumption": "low",
        "characteristics": "0.526 miles, paperclip shape, flat corners, heavy braking"
    },
    "richmond_oval": {
        "name": "Richmond",
        "full_name": "Richmond Raceway",
        "type": "short_track",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "0.75 miles, D-shaped, night racing"
    },
    "phoenix_oval": {
        "name": "Phoenix",
        "full_name": "Phoenix Raceway",
        "type": "short_track",
        "tire_stress": "balanced",
        "key_corners": {
            0.20: "Turn 1-2",
            0.70: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "1 mile, variable banking, dogleg on backstretch"
    },
    "new_hampshire_oval": {
        "name": "New Hampshire",
        "full_name": "New Hampshire Motor Speedway",
        "type": "short_track",
        "tire_stress": "front_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "1.058 miles, flat, The Magic Mile"
    },
    "dover_oval": {
        "name": "Dover",
        "full_name": "Dover Motor Speedway",
        "type": "short_track",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "1 mile, concrete, 24 degree banking, Monster Mile"
    },
    "iowa_oval": {
        "name": "Iowa",
        "full_name": "Iowa Speedway",
        "type": "short_track",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "0.875 miles, progressive banking, IndyCar venue"
    },
    "gateway_oval": {
        "name": "Gateway",
        "full_name": "World Wide Technology Raceway",
        "type": "short_track",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "low",
        "characteristics": "1.25 miles, egg-shaped, variable banking"
    },
    "rockingham_oval": {
        "name": "Rockingham",
        "full_name": "Rockingham Speedway",
        "type": "short_track",
        "tire_stress": "rear_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "1.017 miles, historic NASCAR venue, The Rock"
    },

    # =========================================================================
    # UNIQUE OVALS
    # =========================================================================
    "indianapolis_oval": {
        "name": "Indianapolis",
        "full_name": "Indianapolis Motor Speedway",
        "type": "superspeedway",
        "tire_stress": "balanced",
        "key_corners": {
            0.25: "Turn 1",
            0.50: "Turn 2",
            0.75: "Turn 3",
            1.00: "Turn 4"
        },
        "braking_severity": "light",
        "fuel_consumption": "medium",
        "characteristics": "2.5 miles, The Brickyard, Indy 500, 9 degree banking, rectangle shape"
    },
    "pocono_oval": {
        "name": "Pocono",
        "full_name": "Pocono Raceway",
        "type": "superspeedway",
        "tire_stress": "balanced",
        "key_corners": {
            0.15: "Turn 1 (Tunnel Turn)",
            0.50: "Turn 2 (North Turn)",
            0.85: "Turn 3 (The Groove)"
        },
        "braking_severity": "heavy",
        "fuel_consumption": "medium",
        "characteristics": "2.5 miles, triangular, three unique corners, Tricky Triangle"
    },
    "darlington_oval": {
        "name": "Darlington",
        "full_name": "Darlington Raceway",
        "type": "intermediate",
        "tire_stress": "front_limited",
        "key_corners": {
            0.25: "Turn 1-2",
            0.75: "Turn 3-4"
        },
        "braking_severity": "medium",
        "fuel_consumption": "medium",
        "characteristics": "1.366 miles, egg-shaped, The Lady in Black, wall riding"
    },
}

# Mapping from iRacing track names to metadata keys
_TRACK_NAME_MAP: Dict[str, str] = {
    # =========================================================================
    # EUROPEAN ROAD COURSES
    # =========================================================================
    "autodromo nazionale monza": "monza",
    "monza": "monza",
    "circuit de spa-francorchamps": "spa",
    "spa": "spa",
    "spa-francorchamps": "spa",
    "red bull ring": "redbull_ring",
    "redbull ring": "redbull_ring",
    "autodromo del mugello": "mugello",
    "mugello": "mugello",
    "silverstone circuit": "silverstone",
    "silverstone": "silverstone",
    "nürburgring grand-prix-strecke": "nurburgring_gp",
    "nurburgring grand-prix-strecke": "nurburgring_gp",
    "nurburgring gp": "nurburgring_gp",
    "nürburgring nordschleife": "nurburgring_nordschleife",
    "nurburgring nordschleife": "nurburgring_nordschleife",
    "nordschleife": "nurburgring_nordschleife",
    "nürburgring combined": "nurburgring_combined",
    "nurburgring combined": "nurburgring_combined",
    "autodromo enzo e dino ferrari": "imola",
    "imola": "imola",
    "circuit de barcelona-catalunya": "barcelona",
    "barcelona": "barcelona",
    "barcelona-catalunya": "barcelona",
    "hungaroring": "hungaroring",
    "circuit park zandvoort": "zandvoort",
    "zandvoort": "zandvoort",
    "autódromo internacional do algarve": "portimao",
    "autodromo internacional do algarve": "portimao",
    "portimao": "portimao",
    "portimão": "portimao",
    "algarve": "portimao",
    "hockenheimring baden-württemberg": "hockenheim",
    "hockenheimring baden-wurttemberg": "hockenheim",
    "hockenheim": "hockenheim",
    "hockenheimring": "hockenheim",
    "misano world circuit marco simoncelli": "misano",
    "misano world circuit": "misano",
    "misano": "misano",
    "circuit des 24 heures du mans": "le_mans",
    "le mans": "le_mans",
    "lemans": "le_mans",
    "donington park racing circuit": "donington",
    "donington park": "donington",
    "donington": "donington",
    "brands hatch circuit": "brands_hatch",
    "brands hatch": "brands_hatch",
    "oulton park circuit": "oulton_park",
    "oulton park": "oulton_park",
    "snetterton circuit": "snetterton",
    "snetterton": "snetterton",
    "knockhill racing circuit": "knockhill",
    "knockhill": "knockhill",

    # =========================================================================
    # NORTH AMERICAN ROAD COURSES
    # =========================================================================
    "lime rock park": "lime_rock",
    "lime rock": "lime_rock",
    "road america": "road_america",
    "road atlanta": "road_atlanta",
    "watkins glen international": "watkins_glen",
    "watkins glen": "watkins_glen",
    "the glen": "watkins_glen",
    "weathertech raceway laguna seca": "laguna_seca",
    "laguna seca": "laguna_seca",
    "mazda raceway laguna seca": "laguna_seca",
    "circuit of the americas": "cota",
    "cota": "cota",
    "austin": "cota",
    "mid-ohio sports car course": "mid_ohio",
    "mid-ohio": "mid_ohio",
    "mid ohio": "mid_ohio",
    "virginia international raceway": "vir",
    "vir": "vir",
    "sebring international raceway": "sebring",
    "sebring": "sebring",
    "long beach street circuit": "long_beach",
    "long beach": "long_beach",
    "sonoma raceway": "sonoma",
    "sonoma": "sonoma",
    "sears point": "sonoma",
    "willow springs international motorsports park": "willow_springs",
    "willow springs": "willow_springs",
    "circuit gilles villeneuve": "montreal",
    "montreal": "montreal",
    "gilles villeneuve": "montreal",

    # =========================================================================
    # JAPANESE / ASIAN ROAD COURSES
    # =========================================================================
    "suzuka international racing course": "suzuka",
    "suzuka": "suzuka",
    "fuji speedway": "fuji",
    "fuji": "fuji",
    "okayama international circuit": "okayama",
    "okayama": "okayama",
    "tsukuba circuit": "tsukuba",
    "tsukuba": "tsukuba",
    "twin ring motegi": "motegi",
    "motegi": "motegi",

    # =========================================================================
    # AUSTRALIAN ROAD COURSES
    # =========================================================================
    "mount panorama circuit": "bathurst",
    "bathurst": "bathurst",
    "mount panorama": "bathurst",
    "phillip island grand prix circuit": "phillip_island",
    "phillip island": "phillip_island",
    "adelaide street circuit": "adelaide",
    "adelaide": "adelaide",

    # =========================================================================
    # SOUTH AMERICAN ROAD COURSES
    # =========================================================================
    "autódromo josé carlos pace": "interlagos",
    "autodromo jose carlos pace": "interlagos",
    "interlagos": "interlagos",

    # =========================================================================
    # STREET CIRCUITS
    # =========================================================================
    "miami international autodrome": "miami",
    "miami": "miami",

    # =========================================================================
    # NASCAR / INDYCAR ROAD COURSE CONFIGURATIONS
    # =========================================================================
    "daytona international speedway - road course": "daytona_road",
    "daytona road course": "daytona_road",
    "daytona road": "daytona_road",
    "charlotte motor speedway - roval": "charlotte_roval",
    "charlotte roval": "charlotte_roval",
    "roval": "charlotte_roval",
    "indianapolis motor speedway - road course": "indianapolis_road",
    "indianapolis road course": "indianapolis_road",
    "indy road": "indianapolis_road",

    # =========================================================================
    # SUPERSPEEDWAYS (OVAL)
    # =========================================================================
    "daytona international speedway": "daytona_oval",
    "daytona oval": "daytona_oval",
    "daytona": "daytona_oval",
    "talladega superspeedway": "talladega",
    "talladega": "talladega",

    # =========================================================================
    # INTERMEDIATE SPEEDWAYS (OVAL)
    # =========================================================================
    "charlotte motor speedway": "charlotte_oval",
    "charlotte": "charlotte_oval",
    "atlanta motor speedway": "atlanta_oval",
    "atlanta": "atlanta_oval",
    "texas motor speedway": "texas_oval",
    "texas": "texas_oval",
    "las vegas motor speedway": "las_vegas_oval",
    "las vegas": "las_vegas_oval",
    "kansas speedway": "kansas_oval",
    "kansas": "kansas_oval",
    "michigan international speedway": "michigan_oval",
    "michigan": "michigan_oval",
    "auto club speedway": "auto_club_oval",
    "auto club": "auto_club_oval",
    "fontana": "auto_club_oval",
    "homestead-miami speedway": "homestead_oval",
    "homestead miami speedway": "homestead_oval",
    "homestead": "homestead_oval",
    "chicagoland speedway": "chicagoland_oval",
    "chicagoland": "chicagoland_oval",
    "kentucky speedway": "kentucky_oval",
    "kentucky": "kentucky_oval",
    "nashville superspeedway": "nashville_oval",
    "nashville": "nashville_oval",

    # =========================================================================
    # SHORT TRACKS (OVAL)
    # =========================================================================
    "bristol motor speedway": "bristol_oval",
    "bristol": "bristol_oval",
    "martinsville speedway": "martinsville_oval",
    "martinsville": "martinsville_oval",
    "richmond raceway": "richmond_oval",
    "richmond": "richmond_oval",
    "phoenix raceway": "phoenix_oval",
    "phoenix": "phoenix_oval",
    "new hampshire motor speedway": "new_hampshire_oval",
    "new hampshire": "new_hampshire_oval",
    "loudon": "new_hampshire_oval",
    "dover motor speedway": "dover_oval",
    "dover": "dover_oval",
    "iowa speedway": "iowa_oval",
    "iowa": "iowa_oval",
    "world wide technology raceway": "gateway_oval",
    "gateway": "gateway_oval",
    "wwt raceway": "gateway_oval",
    "rockingham speedway": "rockingham_oval",
    "rockingham": "rockingham_oval",
    "the rock": "rockingham_oval",

    # =========================================================================
    # UNIQUE OVALS
    # =========================================================================
    "indianapolis motor speedway": "indianapolis_oval",
    "indianapolis": "indianapolis_oval",
    "indy": "indianapolis_oval",
    "the brickyard": "indianapolis_oval",
    "pocono raceway": "pocono_oval",
    "pocono": "pocono_oval",
    "darlington raceway": "darlington_oval",
    "darlington": "darlington_oval",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_car_key(iracing_car_name: str) -> Optional[str]:
    """
    Map iRacing car name to metadata key.

    Args:
        iracing_car_name: Car name from iRacing (e.g., "BMW M4 GT3")

    Returns:
        Metadata key (e.g., "bmw_m4_gt3") or None if not found.
    """
    if not iracing_car_name:
        return None

    name_lower = iracing_car_name.lower().strip()

    # Try exact match first
    if name_lower in _CAR_NAME_MAP:
        return _CAR_NAME_MAP[name_lower]

    # Try partial match
    for pattern, key in _CAR_NAME_MAP.items():
        if pattern in name_lower or name_lower in pattern:
            return key

    return None


def get_track_key(iracing_track_name: str) -> Optional[str]:
    """
    Map iRacing track name to metadata key.

    Args:
        iracing_track_name: Track name from iRacing (e.g., "Autodromo Nazionale Monza")

    Returns:
        Metadata key (e.g., "monza") or None if not found.
    """
    if not iracing_track_name:
        return None

    name_lower = iracing_track_name.lower().strip()

    # Try exact match first
    if name_lower in _TRACK_NAME_MAP:
        return _TRACK_NAME_MAP[name_lower]

    # Try partial match
    for pattern, key in _TRACK_NAME_MAP.items():
        if pattern in name_lower or name_lower in pattern:
            return key

    return None


def get_car_metadata(iracing_car_name: str) -> Optional[Dict[str, Any]]:
    """
    Get full car metadata from iRacing car name.

    Args:
        iracing_car_name: Car name from iRacing

    Returns:
        Car metadata dict or None if not found.
    """
    key = get_car_key(iracing_car_name)
    if key and key in CARS:
        return CARS[key]
    return None


def get_track_metadata(iracing_track_name: str) -> Optional[Dict[str, Any]]:
    """
    Get full track metadata from iRacing track name.

    Args:
        iracing_track_name: Track name from iRacing

    Returns:
        Track metadata dict or None if not found.
    """
    key = get_track_key(iracing_track_name)
    if key and key in TRACKS:
        return TRACKS[key]
    return None


def get_upcoming_corner(track_key: str, lap_pct: float) -> Optional[str]:
    """
    Get the upcoming corner name based on lap percentage.

    Finds the next corner ahead of the current position.

    Args:
        track_key: Track metadata key (e.g., "monza")
        lap_pct: Current lap percentage (0.0 to 1.0)

    Returns:
        Corner name or None if track not found.
    """
    if track_key not in TRACKS:
        return None

    key_corners = TRACKS[track_key].get("key_corners", {})
    if not key_corners:
        return None

    # Find the next corner ahead of current position
    upcoming_corners = [
        (pct, name) for pct, name in key_corners.items()
        if pct >= lap_pct
    ]

    if upcoming_corners:
        # Return the nearest upcoming corner
        upcoming_corners.sort(key=lambda x: x[0])
        return upcoming_corners[0][1]

    # If we're past all corners, wrap around to first corner
    first_corner_pct = min(key_corners.keys())
    return key_corners[first_corner_pct]
