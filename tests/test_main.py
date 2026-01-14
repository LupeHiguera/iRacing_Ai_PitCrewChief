"""
Tests for StrategyEngine main integration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from config import Config
from src.main import StrategyEngine
from src.telemetry import TelemetrySnapshot
from src.strategy import StrategyState, Urgency
from src.llm_client import LLMResponse


def make_snapshot(
    lap: int = 5,
    lap_pct: float = 0.5,
    position: int = 5,
    fuel_level: float = 15.0,
    fuel_level_pct: float = 0.5,
    fuel_use_per_hour: float = 2.5,
    tire_wear_lf: float = 20.0,
    tire_wear_rf: float = 25.0,
    tire_wear_lr: float = 15.0,
    tire_wear_rr: float = 18.0,
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


class TestStrategyEngineInitialization:
    """Tests for engine initialization."""

    def test_can_instantiate_with_config(self):
        """Test engine can be created with config."""
        config = Config()
        engine = StrategyEngine(config)
        assert engine is not None

    def test_uses_config_values(self):
        """Test engine uses config thresholds."""
        config = Config(
            fuel_warning_laps=4.0,
            fuel_critical_laps=1.5,
        )
        engine = StrategyEngine(config)
        assert engine._config.fuel_warning_laps == 4.0


class TestLLMTriggers:
    """Tests for LLM call triggering logic."""

    @pytest.mark.asyncio
    async def test_triggers_on_periodic_lap(self):
        """Test LLM triggered every N laps (periodic_update_laps)."""
        config = Config(periodic_update_laps=5)
        engine = StrategyEngine(config)

        # Mock dependencies
        engine._llm = AsyncMock()
        engine._llm.generate = AsyncMock(return_value=LLMResponse("Update", 100))
        engine._llm.format_telemetry_prompt = MagicMock(return_value="prompt")
        engine._tts = AsyncMock()
        engine._tts.speak = AsyncMock()

        state = make_strategy_state(urgency=Urgency.OK)

        # Lap 5 should trigger (returns reason string)
        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=5),
            state=state,
            last_lap=4,
        )
        assert trigger_reason is not None

        # Lap 6 should not trigger (returns None)
        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=6),
            state=state,
            last_lap=5,
        )
        assert trigger_reason is None

        # Lap 10 should trigger
        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=10),
            state=state,
            last_lap=9,
        )
        assert trigger_reason is not None

    @pytest.mark.asyncio
    async def test_triggers_on_urgency_change_to_warning(self):
        """Test LLM triggered when urgency escalates to WARNING."""
        config = Config()
        engine = StrategyEngine(config)
        engine._last_urgency = Urgency.OK

        state = make_strategy_state(urgency=Urgency.WARNING)

        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=3),
            state=state,
            last_lap=2,
        )
        assert trigger_reason is not None

    @pytest.mark.asyncio
    async def test_triggers_on_urgency_change_to_critical(self):
        """Test LLM triggered when urgency escalates to CRITICAL."""
        config = Config()
        engine = StrategyEngine(config)
        engine._last_urgency = Urgency.WARNING

        state = make_strategy_state(urgency=Urgency.CRITICAL)

        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=3),
            state=state,
            last_lap=2,
        )
        assert trigger_reason is not None

    @pytest.mark.asyncio
    async def test_critical_bypasses_cooldown(self):
        """Test CRITICAL urgency bypasses LLM cooldown."""
        config = Config(llm_cooldown_sec=10.0)
        engine = StrategyEngine(config)
        engine._last_urgency = Urgency.OK

        state = make_strategy_state(urgency=Urgency.CRITICAL)

        # Critical always triggers regardless of cooldown
        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=3),
            state=state,
            last_lap=2,
        )
        assert trigger_reason is not None
        assert "CRITICAL" in trigger_reason


class TestLLMCooldown:
    """Tests for LLM cooldown between calls."""

    @pytest.mark.asyncio
    async def test_respects_cooldown(self):
        """Test LLM calls respect cooldown period."""
        config = Config(llm_cooldown_sec=10.0, periodic_update_laps=1)
        engine = StrategyEngine(config)

        # Simulate recent LLM call
        import time
        engine._last_llm_call_time = time.time()

        state = make_strategy_state(urgency=Urgency.OK)

        # Should not trigger due to cooldown (non-critical)
        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=2),
            state=state,
            last_lap=1,
        )
        assert trigger_reason is None

    @pytest.mark.asyncio
    async def test_triggers_after_cooldown_expires(self):
        """Test LLM triggers after cooldown expires."""
        config = Config(llm_cooldown_sec=0.1, periodic_update_laps=1)
        engine = StrategyEngine(config)

        import time
        engine._last_llm_call_time = time.time() - 1.0  # 1 second ago

        state = make_strategy_state(urgency=Urgency.OK)

        # Should trigger, cooldown expired
        trigger_reason = engine._get_trigger_reason(
            snapshot=make_snapshot(lap=2),
            state=state,
            last_lap=1,
        )
        assert trigger_reason is not None


class TestFallbackMessages:
    """Tests for deterministic fallback messages."""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """Test fallback message used when LLM fails."""
        config = Config()
        engine = StrategyEngine(config)

        state = make_strategy_state(
            urgency=Urgency.CRITICAL,
            pit_reason="Fuel critical",
            laps_of_fuel=1.5,
        )
        snapshot = make_snapshot(lap=5, position=3)

        fallback = engine._get_fallback_message(state, snapshot)
        assert fallback is not None
        assert len(fallback) > 0

    @pytest.mark.asyncio
    async def test_fallback_mentions_fuel_when_fuel_critical(self):
        """Test fallback message mentions fuel for fuel issues."""
        config = Config()
        engine = StrategyEngine(config)

        state = make_strategy_state(
            urgency=Urgency.CRITICAL,
            pit_reason="Fuel critical - pit now",
        )
        snapshot = make_snapshot(lap=5, position=3)

        fallback = engine._get_fallback_message(state, snapshot)
        assert "fuel" in fallback.lower() or "pit" in fallback.lower()

    @pytest.mark.asyncio
    async def test_fallback_mentions_tires_when_tire_critical(self):
        """Test fallback message mentions tires for tire issues."""
        config = Config()
        engine = StrategyEngine(config)

        state = make_strategy_state(
            urgency=Urgency.CRITICAL,
            pit_reason="Tires critical - pit now",
        )
        snapshot = make_snapshot(lap=5, position=3)

        fallback = engine._get_fallback_message(state, snapshot)
        assert "tire" in fallback.lower() or "pit" in fallback.lower()


class TestTTSIntegration:
    """Tests for TTS integration."""

    @pytest.mark.asyncio
    async def test_speaks_llm_response(self):
        """Test LLM response is sent to TTS."""
        config = Config()
        engine = StrategyEngine(config)

        engine._tts = AsyncMock()
        engine._tts.speak = AsyncMock()

        await engine._speak("Box this lap for fuel.")

        engine._tts.speak.assert_called_once()
        call_args = engine._tts.speak.call_args
        assert "Box this lap" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_critical_uses_priority_speech(self):
        """Test critical messages use priority TTS."""
        config = Config()
        engine = StrategyEngine(config)

        engine._tts = AsyncMock()
        engine._tts.speak = AsyncMock()

        await engine._speak("Box now!", priority=True)

        engine._tts.speak.assert_called_once()
        call_args = engine._tts.speak.call_args
        assert call_args[1].get("priority") is True or call_args[0][1] is True


class TestSessionLogging:
    """Tests for session logging integration."""

    @pytest.mark.asyncio
    async def test_logs_llm_calls(self):
        """Test LLM calls are logged."""
        config = Config(log_sessions=True)
        engine = StrategyEngine(config)

        engine._logger = MagicMock()
        engine._logger.log_llm_call = MagicMock()

        await engine._log_llm_call("prompt", "response", 150.5)

        engine._logger.log_llm_call.assert_called_once_with(
            "prompt", "response", 150.5
        )

    @pytest.mark.asyncio
    async def test_logs_telemetry(self):
        """Test telemetry is logged."""
        config = Config(log_sessions=True)
        engine = StrategyEngine(config)

        engine._logger = MagicMock()
        engine._logger.log_telemetry = MagicMock()

        snapshot = make_snapshot()
        state = make_strategy_state()

        engine._log_telemetry(snapshot, state)

        engine._logger.log_telemetry.assert_called_once_with(snapshot, state)


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_stop_ends_session(self):
        """Test stopping engine ends logging session."""
        config = Config(log_sessions=True)
        engine = StrategyEngine(config)

        engine._logger = MagicMock()
        engine._logger.end_session = MagicMock()
        engine._tts = AsyncMock()
        engine._tts.stop = AsyncMock()
        engine._llm = AsyncMock()
        engine._llm.close = AsyncMock()
        engine._telemetry = MagicMock()
        engine._telemetry.disconnect = MagicMock()

        await engine.stop()

        engine._logger.end_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_disconnects_telemetry(self):
        """Test stopping engine disconnects from iRacing."""
        config = Config()
        engine = StrategyEngine(config)

        engine._telemetry = MagicMock()
        engine._telemetry.disconnect = MagicMock()
        engine._tts = AsyncMock()
        engine._tts.stop = AsyncMock()
        engine._llm = AsyncMock()
        engine._llm.close = AsyncMock()
        engine._logger = MagicMock()

        await engine.stop()

        engine._telemetry.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_closes_llm(self):
        """Test stopping engine closes LLM client."""
        config = Config()
        engine = StrategyEngine(config)

        engine._llm = AsyncMock()
        engine._llm.close = AsyncMock()
        engine._tts = AsyncMock()
        engine._tts.stop = AsyncMock()
        engine._telemetry = MagicMock()
        engine._telemetry.disconnect = MagicMock()
        engine._logger = MagicMock()

        await engine.stop()

        engine._llm.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_stops_tts(self):
        """Test stopping engine stops TTS."""
        config = Config()
        engine = StrategyEngine(config)

        engine._tts = AsyncMock()
        engine._tts.stop = AsyncMock()
        engine._llm = AsyncMock()
        engine._llm.close = AsyncMock()
        engine._telemetry = MagicMock()
        engine._telemetry.disconnect = MagicMock()
        engine._logger = MagicMock()

        await engine.stop()

        engine._tts.stop.assert_called_once()


class TestIRacingConnection:
    """Tests for iRacing connection handling."""

    @pytest.mark.asyncio
    async def test_waits_for_iracing_connection(self):
        """Test engine waits for iRacing to connect."""
        config = Config()
        engine = StrategyEngine(config)

        # Mock telemetry that connects on second try
        engine._telemetry = MagicMock()
        connect_calls = [False, True]
        engine._telemetry.connect = MagicMock(side_effect=lambda: connect_calls.pop(0))

        connected = await engine._wait_for_connection(max_attempts=2, delay=0.01)

        assert connected is True
        assert engine._telemetry.connect.call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self):
        """Test engine gives up after max connection attempts."""
        config = Config()
        engine = StrategyEngine(config)

        engine._telemetry = MagicMock()
        engine._telemetry.connect = MagicMock(return_value=False)

        connected = await engine._wait_for_connection(max_attempts=3, delay=0.01)

        assert connected is False
        assert engine._telemetry.connect.call_count == 3