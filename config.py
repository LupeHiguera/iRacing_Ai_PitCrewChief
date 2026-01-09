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
    llm_timeout_sec: float = 3.0

    # Strategy thresholds
    fuel_warning_laps: float = 5.0
    fuel_critical_laps: float = 2.0
    tire_warning_pct: float = 70.0
    tire_critical_pct: float = 85.0

    # Update behavior
    periodic_update_laps: int = 1  # Changed from 5 for testing
    llm_cooldown_sec: float = 10.0

    # TTS
    tts_queue_size: int = 10

    # Logging
    log_sessions: bool = True
    session_log_dir: Path = field(default_factory=lambda: Path("./data/sessions"))

    def __post_init__(self):
        """Ensure directories exist."""
        if self.log_sessions:
            self.session_log_dir.mkdir(parents=True, exist_ok=True)