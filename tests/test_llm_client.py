"""
Tests for LMStudioClient class.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import aiohttp

from src.llm_client import LMStudioClient, LLMResponse
from src.telemetry import TelemetrySnapshot
from src.strategy import StrategyState, Urgency


def make_snapshot(
    lap: int = 10,
    lap_pct: float = 0.5,
    position: int = 5,
    fuel_level: float = 15.0,
    fuel_level_pct: float = 0.5,
    fuel_use_per_hour: float = 2.5,
    tire_wear_lf: float = 30.0,
    tire_wear_rf: float = 35.0,
    tire_wear_lr: float = 25.0,
    tire_wear_rr: float = 28.0,
    session_time_remain: float = 1800.0,
    session_laps_remain: int = 20,
    last_lap_time: float = 90.0,
    best_lap_time: float = 88.0,
    on_pit_road: bool = False,
    is_on_track: bool = True,
) -> TelemetrySnapshot:
    """Helper to create test snapshots."""
    return TelemetrySnapshot(
        lap=lap,
        lap_pct=lap_pct,
        position=position,
        fuel_level=fuel_level,
        fuel_level_pct=fuel_level_pct,
        fuel_use_per_hour=fuel_use_per_hour,
        tire_wear_lf=tire_wear_lf,
        tire_wear_rf=tire_wear_rf,
        tire_wear_lr=tire_wear_lr,
        tire_wear_rr=tire_wear_rr,
        session_time_remain=session_time_remain,
        session_laps_remain=session_laps_remain,
        last_lap_time=last_lap_time,
        best_lap_time=best_lap_time,
        on_pit_road=on_pit_road,
        is_on_track=is_on_track,
    )


def make_strategy_state(
    fuel_per_lap: float = 2.5,
    laps_of_fuel: float = 6.0,
    pit_window: int = 4,
    worst_tire_corner: str = "RF",
    worst_tire_wear: float = 35.0,
    needs_pit: bool = False,
    pit_reason: str = None,
    urgency: Urgency = Urgency.OK,
) -> StrategyState:
    """Helper to create test strategy states."""
    return StrategyState(
        fuel_per_lap=fuel_per_lap,
        laps_of_fuel=laps_of_fuel,
        pit_window=pit_window,
        worst_tire_corner=worst_tire_corner,
        worst_tire_wear=worst_tire_wear,
        needs_pit=needs_pit,
        pit_reason=pit_reason,
        urgency=urgency,
    )


class TestLMStudioClientInitialization:
    """Tests for client initialization."""

    def test_can_instantiate_with_defaults(self):
        """Test client can be created with default URL."""
        client = LMStudioClient()
        assert client is not None
        assert client.base_url == "http://localhost:1234/v1"

    def test_can_instantiate_with_custom_url(self):
        """Test client accepts custom base URL."""
        client = LMStudioClient(base_url="http://localhost:5000/v1")
        assert client.base_url == "http://localhost:5000/v1"

    def test_can_instantiate_with_custom_timeout(self):
        """Test client accepts custom timeout."""
        client = LMStudioClient(timeout=5.0)
        assert client.timeout == 5.0

    def test_default_timeout_is_3_seconds(self):
        """Test default timeout is 3 seconds per spec."""
        client = LMStudioClient()
        assert client.timeout == 3.0


class TestGenerateMethod:
    """Tests for the generate() method."""

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(self):
        """Test generate returns LLMResponse with text and latency."""
        client = LMStudioClient()

        mock_response = {
            "choices": [{"message": {"content": "Box this lap for fuel."}}]
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await client.generate("Test prompt")

        assert isinstance(result, LLMResponse)
        assert result.text == "Box this lap for fuel."
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_timeout(self):
        """Test generate returns None when request times out."""
        client = LMStudioClient(timeout=0.1)

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.side_effect = asyncio.TimeoutError()
            result = await client.generate("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_connection_error(self):
        """Test generate returns None when LM Studio is unreachable."""
        client = LMStudioClient()

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.side_effect = aiohttp.ClientError()
            result = await client.generate("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_invalid_response(self):
        """Test generate returns None when response format is invalid."""
        client = LMStudioClient()

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = {"invalid": "response"}
            result = await client.generate("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_empty_response(self):
        """Test generate returns None when response is empty."""
        client = LMStudioClient()

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = {"choices": [{"message": {"content": ""}}]}
            result = await client.generate("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_tracks_latency(self):
        """Test generate measures request latency."""
        client = LMStudioClient()

        mock_response = {
            "choices": [{"message": {"content": "Response text"}}]
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            # Simulate some latency
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.05)
                return mock_response

            mock.side_effect = delayed_response
            result = await client.generate("Test prompt")

        assert result.latency_ms >= 50  # At least 50ms


class TestFormatTelemetryPrompt:
    """Tests for format_telemetry_prompt() method."""

    def test_formats_basic_telemetry(self):
        """Test prompt includes key telemetry values."""
        client = LMStudioClient()
        snapshot = make_snapshot(lap=15, position=3)
        state = make_strategy_state(laps_of_fuel=8.0, worst_tire_wear=40.0)

        prompt = client.format_telemetry_prompt(state, snapshot)

        assert "15" in prompt  # lap
        assert "3" in prompt or "P3" in prompt  # position
        assert "8" in prompt  # laps of fuel

    def test_includes_urgency_in_prompt(self):
        """Test prompt reflects urgency level."""
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state(urgency=Urgency.CRITICAL, pit_reason="Low fuel")

        prompt = client.format_telemetry_prompt(state, snapshot)

        assert "critical" in prompt.lower() or "urgent" in prompt.lower()

    def test_includes_pit_reason_when_needed(self):
        """Test prompt includes pit reason when needs_pit is True."""
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state(
            needs_pit=True,
            pit_reason="Fuel critical",
            urgency=Urgency.CRITICAL,
        )

        prompt = client.format_telemetry_prompt(state, snapshot)

        assert "fuel" in prompt.lower()

    def test_includes_tire_info(self):
        """Test prompt includes tire wear information."""
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state(worst_tire_corner="RF", worst_tire_wear=75.0)

        prompt = client.format_telemetry_prompt(state, snapshot)

        assert "RF" in prompt or "right front" in prompt.lower()
        assert "75" in prompt

    def test_includes_fuel_info(self):
        """Test prompt includes fuel information."""
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state(fuel_per_lap=2.5, laps_of_fuel=4.5)

        prompt = client.format_telemetry_prompt(state, snapshot)

        assert "4" in prompt or "4.5" in prompt  # laps of fuel

    def test_includes_session_context(self):
        """Test prompt includes laps remaining context."""
        client = LMStudioClient()
        snapshot = make_snapshot(session_laps_remain=5)
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt(state, snapshot)

        assert "5" in prompt  # laps remaining


class TestSystemPrompt:
    """Tests for race engineer system prompt."""

    def test_has_system_prompt(self):
        """Test client has a system prompt defined."""
        client = LMStudioClient()
        assert client.system_prompt is not None
        assert len(client.system_prompt) > 0

    def test_system_prompt_mentions_conciseness(self):
        """Test system prompt instructs concise responses."""
        client = LMStudioClient()
        prompt_lower = client.system_prompt.lower()

        assert "concise" in prompt_lower or "brief" in prompt_lower or "short" in prompt_lower

    def test_system_prompt_suitable_for_tts(self):
        """Test system prompt mentions TTS/voice compatibility."""
        client = LMStudioClient()
        prompt_lower = client.system_prompt.lower()

        # Should mention voice/speech/TTS or no formatting
        has_voice_mention = any(
            word in prompt_lower
            for word in ["voice", "speech", "tts", "spoken", "radio"]
        )
        has_format_restriction = any(
            word in prompt_lower
            for word in ["no bullet", "no formatting", "plain text"]
        )

        assert has_voice_mention or has_format_restriction


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_has_required_fields(self):
        """Test LLMResponse has text and latency fields."""
        response = LLMResponse(text="Box now", latency_ms=150.5)

        assert response.text == "Box now"
        assert response.latency_ms == 150.5

    def test_llm_response_text_is_string(self):
        """Test LLMResponse text is a string."""
        response = LLMResponse(text="Test", latency_ms=100)
        assert isinstance(response.text, str)

    def test_llm_response_latency_is_numeric(self):
        """Test LLMResponse latency is numeric."""
        response = LLMResponse(text="Test", latency_ms=100.5)
        assert isinstance(response.latency_ms, (int, float))


class TestClientCleanup:
    """Tests for client resource management."""

    @pytest.mark.asyncio
    async def test_client_can_be_used_as_context_manager(self):
        """Test client supports async context manager for cleanup."""
        async with LMStudioClient() as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_close_releases_resources(self):
        """Test close() properly releases aiohttp session."""
        client = LMStudioClient()
        # Force session creation
        await client._ensure_session()
        assert client._session is not None

        await client.close()
        assert client._session is None or client._session.closed


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_very_long_response(self):
        """Test client handles unexpectedly long responses."""
        client = LMStudioClient()

        long_text = "word " * 1000  # Very long response
        mock_response = {
            "choices": [{"message": {"content": long_text}}]
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await client.generate("Test")

        # Should still return, possibly truncated
        assert result is not None

    @pytest.mark.asyncio
    async def test_handles_special_characters_in_response(self):
        """Test client handles special characters from LLM."""
        client = LMStudioClient()

        mock_response = {
            "choices": [{"message": {"content": "Box now! You're P3—push hard."}}]
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await client.generate("Test")

        assert result is not None
        assert "P3" in result.text

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self):
        """Test client handles multiple sequential generate calls."""
        client = LMStudioClient()

        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response

            result1 = await client.generate("Prompt 1")
            result2 = await client.generate("Prompt 2")
            result3 = await client.generate("Prompt 3")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert mock.call_count == 3
