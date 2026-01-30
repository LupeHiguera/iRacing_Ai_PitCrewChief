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

    # Tire temperature thresholds (Celsius)
    tire_temp_cold_c: float = 60.0  # Below = cold warning
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

    def __post_init__(self):
        """Ensure directories exist."""
        if self.log_sessions:
            self.session_log_dir.mkdir(parents=True, exist_ok=True)