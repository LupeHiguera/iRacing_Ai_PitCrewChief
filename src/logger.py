"""
Session logger for collecting fine-tuning data.
"""

import gzip
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .telemetry import TelemetrySnapshot
from .strategy import StrategyState, Urgency


class SessionLogger:
    """Logs race sessions for fine-tuning data collection."""

    def __init__(self, log_dir: str = "./data/sessions"):
        self.log_dir = log_dir
        self._session_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._track: Optional[str] = None
        self._car: Optional[str] = None
        self._events: List[Dict[str, Any]] = []
        self._current_lap: int = 0

    def start_session(self, track: str, car: str) -> str:
        """
        Start a new logging session.

        Args:
            track: Track name.
            car: Car name.

        Returns:
            Session ID.
        """
        # Create log directory if needed
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize session
        self._session_id = str(uuid.uuid4())[:8]
        self._start_time = datetime.now()
        self._track = track
        self._car = car
        self._events = []
        self._current_lap = 0

        return self._session_id

    def log_telemetry(
        self,
        snapshot: TelemetrySnapshot,
        strategy_state: StrategyState,
    ) -> None:
        """
        Log a telemetry snapshot and strategy state.

        Args:
            snapshot: Current telemetry snapshot.
            strategy_state: Current strategy state.
        """
        if self._session_id is None:
            return

        self._current_lap = snapshot.lap

        event = {
            "timestamp": datetime.now().isoformat(),
            "lap": snapshot.lap,
            "event_type": "telemetry",
            "data": {
                "snapshot": self._snapshot_to_dict(snapshot),
                "strategy": self._strategy_to_dict(strategy_state),
            },
        }
        self._events.append(event)

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        latency_ms: float,
    ) -> None:
        """
        Log an LLM call.

        Args:
            prompt: The prompt sent to the LLM.
            response: The response from the LLM.
            latency_ms: Response latency in milliseconds.
        """
        if self._session_id is None:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "lap": self._current_lap,
            "event_type": "llm",
            "data": {
                "prompt": prompt,
                "response": response,
                "latency_ms": latency_ms,
            },
        }
        self._events.append(event)

    def end_session(self) -> Optional[str]:
        """
        End the session and save to gzipped JSON file.

        Returns:
            Path to the saved file, or None if no session was active.
        """
        if self._session_id is None:
            return None

        # Build session data
        session_data = {
            "metadata": {
                "session_id": self._session_id,
                "start_time": self._start_time.isoformat(),
                "track": self._track,
                "car": self._car,
            },
            "events": self._events,
        }

        # Generate filename from timestamp with microseconds for uniqueness
        filename = self._start_time.strftime("%Y-%m-%d_%H-%M-%S-%f") + ".json.gz"
        filepath = os.path.join(self.log_dir, filename)

        # Write gzipped JSON
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        # Reset session
        self._session_id = None
        self._start_time = None
        self._track = None
        self._car = None
        self._events = []
        self._current_lap = 0

        return filepath

    def _snapshot_to_dict(self, snapshot: TelemetrySnapshot) -> Dict[str, Any]:
        """Convert TelemetrySnapshot to dict."""
        return asdict(snapshot)

    def _strategy_to_dict(self, state: StrategyState) -> Dict[str, Any]:
        """Convert StrategyState to dict with enum handling."""
        data = asdict(state)
        # Convert Urgency enum to string
        data["urgency"] = state.urgency.value
        return data