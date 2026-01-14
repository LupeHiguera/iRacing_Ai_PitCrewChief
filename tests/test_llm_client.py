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
        """Test system prompt is suitable for race engineer role."""
        client = LMStudioClient()
        prompt_lower = client.system_prompt.lower()

        # Should mention race engineer role and brevity
        assert "race engineer" in prompt_lower
        assert "callout" in prompt_lower or "brief" in prompt_lower


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


# =============================================================================
# NEW TESTS: JSON Format for Fine-Tuned Model (TDD)
# =============================================================================

class TestFormatTelemetryPromptJSON:
    """Tests for JSON format output matching training data format."""

    def test_format_returns_valid_json(self):
        """Test format_telemetry_prompt_json returns valid JSON string."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )

        # Should be valid JSON
        data = json.loads(prompt)
        assert isinstance(data, dict)

    def test_json_includes_car_info(self):
        """Test JSON includes car name, class, and traits."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert "car" in data
        assert "car_class" in data
        assert "car_traits" in data
        assert data["car"] == "BMW M4 GT3"
        assert data["car_class"] == "GT3"
        assert isinstance(data["car_traits"], list)

    def test_json_includes_track_info(self):
        """Test JSON includes track name and type."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert "track" in data
        assert "track_type" in data
        assert data["track"] == "Monza"  # Short name
        assert data["track_type"] == "high_speed"

    def test_json_includes_lap_info(self):
        """Test JSON includes lap number and lap_pct."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot(lap=12, lap_pct=0.65)
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["lap"] == 12
        assert data["lap_pct"] == 0.65

    def test_json_includes_position(self):
        """Test JSON includes position."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot(position=8)
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["position"] == 8

    def test_json_includes_fuel_laps_remaining(self):
        """Test JSON includes fuel_laps_remaining from strategy state."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state(laps_of_fuel=14.5)

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["fuel_laps_remaining"] == 14.5

    def test_json_includes_tire_wear_dict(self):
        """Test JSON includes tire_wear as dict with fl/fr/rl/rr keys."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot(
            tire_wear_lf=15.0, tire_wear_rf=22.0,
            tire_wear_lr=12.0, tire_wear_rr=14.0
        )
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert "tire_wear" in data
        assert isinstance(data["tire_wear"], dict)
        assert data["tire_wear"]["fl"] == 15.0
        assert data["tire_wear"]["fr"] == 22.0
        assert data["tire_wear"]["rl"] == 12.0
        assert data["tire_wear"]["rr"] == 14.0

    def test_json_includes_tire_temps_averaged(self):
        """Test JSON includes tire_temps as averaged L/M/R to single value per corner."""
        import json
        client = LMStudioClient()
        # Set tire temps: LF L=90, M=95, R=100 -> avg = 95
        snapshot = make_snapshot()
        # We need to set tire temps on the snapshot
        snapshot.tire_temp_lf_l = 90.0
        snapshot.tire_temp_lf_m = 95.0
        snapshot.tire_temp_lf_r = 100.0
        snapshot.tire_temp_rf_l = 100.0
        snapshot.tire_temp_rf_m = 102.0
        snapshot.tire_temp_rf_r = 104.0
        snapshot.tire_temp_lr_l = 85.0
        snapshot.tire_temp_lr_m = 88.0
        snapshot.tire_temp_lr_r = 91.0
        snapshot.tire_temp_rr_l = 88.0
        snapshot.tire_temp_rr_m = 91.0
        snapshot.tire_temp_rr_r = 94.0
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert "tire_temps" in data
        assert isinstance(data["tire_temps"], dict)
        # LF: (90+95+100)/3 = 95
        assert data["tire_temps"]["fl"] == 95.0
        # RF: (100+102+104)/3 = 102
        assert data["tire_temps"]["fr"] == 102.0
        # LR: (85+88+91)/3 = 88
        assert data["tire_temps"]["rl"] == 88.0
        # RR: (88+91+94)/3 = 91
        assert data["tire_temps"]["rr"] == 91.0

    def test_json_includes_gaps(self):
        """Test JSON includes gap_ahead and gap_behind."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        snapshot.gap_ahead_sec = 2.1
        snapshot.gap_behind_sec = 0.8
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["gap_ahead"] == 2.1
        assert data["gap_behind"] == 0.8

    def test_json_includes_lap_times(self):
        """Test JSON includes last_lap_time and best_lap_time."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot(last_lap_time=91.5, best_lap_time=90.8)
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["last_lap_time"] == 91.5
        assert data["best_lap_time"] == 90.8

    def test_json_includes_session_laps_remain(self):
        """Test JSON includes session_laps_remain."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot(session_laps_remain=18)
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["session_laps_remain"] == 18

    def test_json_includes_incident_count(self):
        """Test JSON includes incident_count."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        snapshot.incident_count = 2
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["incident_count"] == 2

    def test_json_includes_track_temp(self):
        """Test JSON includes track_temp_c."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        snapshot.track_temp_c = 35.0
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        assert data["track_temp_c"] == 35.0

    def test_json_handles_unknown_car_gracefully(self):
        """Test JSON format handles unknown car with defaults."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="Unknown Car XYZ", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        # Should still produce valid JSON with car name but unknown class/traits
        assert data["car"] == "Unknown Car XYZ"
        assert data["car_class"] == "unknown"
        assert data["car_traits"] == []

    def test_json_handles_unknown_track_gracefully(self):
        """Test JSON format handles unknown track with defaults."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Unknown Track 3000"
        )
        data = json.loads(prompt)

        # Should still produce valid JSON with track name but unknown type
        assert data["track"] == "Unknown Track 3000"
        assert data["track_type"] == "unknown"

    def test_json_handles_none_gaps(self):
        """Test JSON format handles None gap values."""
        import json
        client = LMStudioClient()
        snapshot = make_snapshot()
        snapshot.gap_ahead_sec = None
        snapshot.gap_behind_sec = None
        state = make_strategy_state()

        prompt = client.format_telemetry_prompt_json(
            state, snapshot, car_name="BMW M4 GT3", track_name="Autodromo Nazionale Monza"
        )
        data = json.loads(prompt)

        # None should become null in JSON or be omitted
        assert data.get("gap_ahead") is None
        assert data.get("gap_behind") is None


class TestUpdatedSystemPrompt:
    """Tests for updated system prompt matching training instruction."""

    def test_system_prompt_for_json_format(self):
        """Test system prompt matches training instruction format."""
        client = LMStudioClient()

        # The system prompt should match what we use in training
        expected_phrase = "race engineer"
        assert expected_phrase in client.system_prompt.lower()

    def test_system_prompt_mentions_car_track_telemetry(self):
        """Test system prompt mentions car, track, and telemetry."""
        client = LMStudioClient()
        prompt_lower = client.system_prompt.lower()

        # Should reference the key input types
        # Note: This test may need adjustment based on final prompt wording
        has_context = (
            "car" in prompt_lower or
            "telemetry" in prompt_lower or
            "driver" in prompt_lower
        )
        assert has_context
