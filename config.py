"""
Configuration for iRacing AI Race Strategist.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class Config:
    """Configuration with environment variable support."""

    # Paths - loaded from .env file
    piper_path: str = field(
        default_factory=lambda: os.getenv("PIPER_PATH", "piper.exe")
    )
    tts_model_path: str = field(
        default_factory=lambda: os.getenv("TTS_MODEL_PATH", "en_US-lessac-medium.onnx")
    )

    # LM Studio
    lm_studio_url: str = field(
        default_factory=lambda: os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    )
    llm_timeout_sec: float = 10.0

    # Strategy thresholds
    fuel_warning_laps: float = 5.0
    fuel_critical_laps: float = 2.0
    tire_warning_pct: float = 70.0
    tire_critical_pct: float = 85.0

    # Update behavior
    periodic_update_laps: int = 5  # Routine updates every N laps
    llm_cooldown_sec: float = 10.0

    # === EVENT DETECTION ===

    # Position changes (batched to avoid lap 1 spam)
    position_change_enabled: bool = True
    position_settle_time_sec: float = 5.0  # Wait for positions to settle
    position_min_lap: int = 2  # Skip position callouts before this lap

    # Gap/Battle detection (seconds)
    gap_close_threshold_sec: float = 1.5  # "Car closing" when under this
    gap_battle_threshold_sec: float = 0.8  # "Defend!" when under this
    gap_safe_threshold_sec: float = 3.0  # "Clear" when over this (after being close)
    gap_ahead_dirty_air_sec: float = 1.5  # "Dirty air" when following closer
    gap_ahead_clean_air_sec: float = 2.5  # "Clean air" when gap opens

    # iRacing only updates tire temp AND wear telemetry when the pit crew
    # measures the tires (at spawn and after each pit stop) -- the values are
    # frozen during a stint. There is no live tire-temp channel in the SDK.
    # With this False, live tire-temp callouts are disabled and tire_temps are
    # omitted from the model prompt (otherwise the frozen cold spawn value makes
    # the model report "tires cold" forever). Flip to True only on a sim that
    # actually streams live tire temps.
    tire_temp_telemetry_live: bool = False

    # Since iRacing has no live tire-temp channel, estimate per-corner tire
    # temps/wear from live load telemetry (G, throttle, brake, speed), anchored
    # to the real values measured at each pit stop. Estimated values feed the
    # model prompt and the tire events, and are labeled "estimated" in the
    # overlay. See src/tire_model.py.
    tire_temp_estimation_enabled: bool = True

    # Tire-state estimator tunables (dimensionless gains unless noted).
    # Tuned so a cold tire reaches the ~80-100C window after 1-2 pushing laps,
    # fronts spike under braking, the outer pair runs hotter in long corners,
    # and long straights cool toward the live track/air ambient.
    tire_est_heat_gain: float = 0.024  # heat-in scale (load x speed)
    tire_est_cool_rate: float = 0.012  # cooling scale toward ambient (x airflow)
    tire_est_ambient_offset_c: float = 8.0  # resting tire temp above track temp
    tire_est_wear_gain: float = 0.008  # wear %/tick scale (load x slip), x tire_deg; ~0.4%/lap GT3
    tire_est_max_temp_c: float = 140.0  # clamp ceiling

    # Tire temperature thresholds (Celsius) - used for live OR estimated temps
    tire_temp_cold_c: float = 40.0  # Below = cold warning
    tire_temp_optimal_low_c: float = 80.0  # Optimal range start
    tire_temp_optimal_high_c: float = 100.0  # Optimal range end
    tire_temp_hot_c: float = 110.0  # Above = overheating
    tire_temp_spike_delta_c: float = 15.0  # Sudden spike = lockup/wheelspin

    # Pace tracking
    pace_trend_laps: int = 3  # Compare last N laps for trend
    pace_drop_threshold_sec: float = 0.5  # Pace dropping alert
    pace_gain_threshold_sec: float = 0.3  # Found pace alert

    # Race progress callouts
    race_progress_enabled: bool = True
    race_halfway_callout: bool = True
    race_laps_remaining_callouts: tuple = (5, 3, 1)  # Callout at these laps remaining

    # Other event callouts
    incident_callout_enabled: bool = True
    flag_callout_enabled: bool = True
    pit_entry_callout: bool = True
    pit_exit_callout: bool = True
    personal_best_callout: bool = True

    # Event cooldowns (seconds) - per event type
    event_cooldown_position: float = 10.0
    event_cooldown_gap: float = 15.0
    event_cooldown_tire_temp: float = 30.0
    event_cooldown_pace: float = 60.0

    # TTS
    tts_queue_size: int = 10

    # Logging
    log_sessions: bool = True
    session_log_dir: Path = field(default_factory=lambda: Path("./data/sessions"))

    # Overlay
    overlay_enabled: bool = True
    overlay_host: str = "localhost"
    overlay_port: int = 8080
    overlay_model_label: str = "Race-Engineer FT v1"  # Shown as badge in overlay
    overlay_fine_tuned: bool = True  # Flag toggles "FINE-TUNED" vs "BASE" badge styling

    def __post_init__(self):
        """Ensure directories exist."""
        if self.log_sessions:
            self.session_log_dir.mkdir(parents=True, exist_ok=True)