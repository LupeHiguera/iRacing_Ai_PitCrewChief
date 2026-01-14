"""
Async LLM client for LM Studio's OpenAI-compatible API.
"""

import json
import time
import asyncio
from dataclasses import dataclass
from typing import Optional

import aiohttp

from typing import TYPE_CHECKING

from .telemetry import TelemetrySnapshot
from .strategy import StrategyState, Urgency
from .metadata import get_car_metadata, get_track_metadata

if TYPE_CHECKING:
    from .event_detector import RaceEvent


@dataclass
class LLMResponse:
    """Response from LLM with text and latency."""
    text: str
    latency_ms: float


class LMStudioClient:
    """Async client for LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 3.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        self.system_prompt = (
            "You are a race engineer giving radio updates to your driver. "
            "Be concise - max 2 sentences. Use plain text suitable for voice/TTS. "
            "No bullets, no formatting. Be calm but urgent when needed. "
            "Give actionable advice."
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def generate(self, prompt: str) -> Optional[LLMResponse]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.

        Returns:
            LLMResponse with text and latency, or None on failure.
        """
        start_time = time.perf_counter()

        try:
            response = await self._make_request(prompt)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse response
            text = self._extract_text(response)
            if not text:
                return None

            return LLMResponse(text=text, latency_ms=latency_ms)

        except asyncio.TimeoutError:
            print(f"[LLM] Timeout after {self.timeout}s")
            return None
        except aiohttp.ClientError as e:
            print(f"[LLM] Connection error: {e}")
            return None
        except Exception as e:
            print(f"[LLM] Unexpected error: {e}")
            return None

    async def _make_request(self, prompt: str) -> dict:
        """Make the actual API request to LM Studio."""
        session = await self._ensure_session()

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        async with session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _extract_text(self, response: dict) -> Optional[str]:
        """Extract text content from API response."""
        try:
            text = response["choices"][0]["message"]["content"]
            if text and text.strip():
                return text.strip()
            return None
        except (KeyError, IndexError, TypeError):
            return None

    def format_telemetry_prompt(
        self,
        state: StrategyState,
        snapshot: TelemetrySnapshot,
        event: Optional["RaceEvent"] = None,
    ) -> str:
        """
        Format telemetry and strategy state into a prompt for the LLM.

        Args:
            state: Current strategy state.
            snapshot: Current telemetry snapshot.
            event: Optional event that triggered this call.

        Returns:
            Formatted prompt string.
        """
        lines = [
            f"Lap {snapshot.lap}, P{snapshot.position}",
            f"Fuel: {state.laps_of_fuel:.1f} laps remaining ({state.fuel_per_lap:.2f}/lap)",
            f"Tires: {state.worst_tire_corner} at {state.worst_tire_wear:.0f}% worn",
        ]

        # Add session info - handle time-based vs lap-based races
        if snapshot.session_laps_remain < 1000:
            lines.append(f"Session: {snapshot.session_laps_remain} laps to go")
        elif snapshot.session_time_remain > 0:
            mins_remain = int(snapshot.session_time_remain // 60)
            lines.append(f"Session: {mins_remain} minutes remaining")
        # Omit session line if neither is useful

        # Add gap info if available
        if snapshot.gap_behind_sec is not None:
            lines.append(f"Gap behind: {snapshot.gap_behind_sec:.1f}s")
        if snapshot.gap_ahead_sec is not None:
            lines.append(f"Gap ahead: {snapshot.gap_ahead_sec:.1f}s")

        # Add event context if provided
        if event:
            lines.append(f"EVENT: {event.message}")
            # Add event-specific context
            if event.data:
                for key, value in event.data.items():
                    if key not in ["gap_behind", "gap_ahead"]:  # Avoid duplicates
                        lines.append(f"  {key}: {value}")

        # Add urgency context
        if state.urgency == Urgency.CRITICAL:
            lines.append(f"CRITICAL: {state.pit_reason}")
        elif state.urgency == Urgency.WARNING:
            lines.append(f"Warning: {state.pit_reason}")
        elif state.needs_pit:
            lines.append(f"Pit recommended: {state.pit_reason}")

        return "\n".join(lines)

    def format_telemetry_prompt_json(
        self,
        state: StrategyState,
        snapshot: TelemetrySnapshot,
        car_name: str,
        track_name: str,
        event: Optional["RaceEvent"] = None,
    ) -> str:
        """
        Format telemetry into JSON format matching training data schema.

        This format is used for the fine-tuned model which expects structured
        JSON input with car/track metadata.

        Args:
            state: Current strategy state.
            snapshot: Current telemetry snapshot.
            car_name: Car name from iRacing.
            track_name: Track name from iRacing.
            event: Optional event that triggered this call.

        Returns:
            JSON string with all telemetry fields.
        """
        # Get car metadata
        car_meta = get_car_metadata(car_name)
        if car_meta:
            car_class = car_meta.get("class", "unknown")
            car_traits = car_meta.get("traits", [])
        else:
            car_class = "unknown"
            car_traits = []

        # Get track metadata
        track_meta = get_track_metadata(track_name)
        if track_meta:
            track_short_name = track_meta.get("name", track_name)
            track_type = track_meta.get("type", "unknown")
        else:
            track_short_name = track_name
            track_type = "unknown"

        # Calculate averaged tire temps (L/M/R -> single value per corner)
        tire_temps = {
            "fl": round((snapshot.tire_temp_lf_l + snapshot.tire_temp_lf_m + snapshot.tire_temp_lf_r) / 3, 1),
            "fr": round((snapshot.tire_temp_rf_l + snapshot.tire_temp_rf_m + snapshot.tire_temp_rf_r) / 3, 1),
            "rl": round((snapshot.tire_temp_lr_l + snapshot.tire_temp_lr_m + snapshot.tire_temp_lr_r) / 3, 1),
            "rr": round((snapshot.tire_temp_rr_l + snapshot.tire_temp_rr_m + snapshot.tire_temp_rr_r) / 3, 1),
        }

        # Build the data structure
        data = {
            # Car info
            "car": car_name,
            "car_class": car_class,
            "car_traits": car_traits,

            # Track info
            "track": track_short_name,
            "track_type": track_type,

            # Lap info
            "lap": snapshot.lap,
            "lap_pct": snapshot.lap_pct,
            "position": snapshot.position,

            # Fuel
            "fuel_laps_remaining": state.laps_of_fuel,

            # Tire wear
            "tire_wear": {
                "fl": snapshot.tire_wear_lf,
                "fr": snapshot.tire_wear_rf,
                "rl": snapshot.tire_wear_lr,
                "rr": snapshot.tire_wear_rr,
            },

            # Tire temps (averaged)
            "tire_temps": tire_temps,

            # Gaps
            "gap_ahead": snapshot.gap_ahead_sec,
            "gap_behind": snapshot.gap_behind_sec,

            # Lap times
            "last_lap_time": snapshot.last_lap_time,
            "best_lap_time": snapshot.best_lap_time,

            # Session
            "session_laps_remain": snapshot.session_laps_remain,

            # Other
            "incident_count": snapshot.incident_count,
            "track_temp_c": snapshot.track_temp_c,
        }

        return json.dumps(data)

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "LMStudioClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()