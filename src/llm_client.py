"""
Async LLM client for LM Studio's OpenAI-compatible API.
"""

import time
import asyncio
from dataclasses import dataclass
from typing import Optional

import aiohttp

from .telemetry import TelemetrySnapshot
from .strategy import StrategyState, Urgency


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
            return None
        except aiohttp.ClientError:
            return None
        except Exception:
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
    ) -> str:
        """
        Format telemetry and strategy state into a prompt for the LLM.

        Args:
            state: Current strategy state.
            snapshot: Current telemetry snapshot.

        Returns:
            Formatted prompt string.
        """
        lines = [
            f"Lap {snapshot.lap}, P{snapshot.position}",
            f"Fuel: {state.laps_of_fuel:.1f} laps remaining ({state.fuel_per_lap:.2f}/lap)",
            f"Tires: {state.worst_tire_corner} at {state.worst_tire_wear:.0f}% worn",
            f"Session: {snapshot.session_laps_remain} laps to go",
        ]

        # Add urgency context
        if state.urgency == Urgency.CRITICAL:
            lines.append(f"CRITICAL: {state.pit_reason}")
        elif state.urgency == Urgency.WARNING:
            lines.append(f"Warning: {state.pit_reason}")
        elif state.needs_pit:
            lines.append(f"Pit recommended: {state.pit_reason}")

        return "\n".join(lines)

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