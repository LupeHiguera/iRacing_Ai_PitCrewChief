"""
Tests for SessionLogger class.
"""

import gzip
import json
import os
import tempfile
from pathlib import Path

import pytest

from src.logger import SessionLogger
from src.telemetry import TelemetrySnapshot
from src.strategy import StrategyState, Urgency


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


class TestSessionLoggerInitialization:
    """Tests for logger initialization."""

    def test_can_instantiate_with_default_path(self):
        """Test logger can be created with default log directory."""
        logger = SessionLogger()
        assert logger is not None

    def test_can_instantiate_with_custom_path(self):
        """Test logger accepts custom log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            assert logger.log_dir == tmpdir

    def test_creates_log_directory_if_missing(self):
        """Test logger creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "sessions", "nested")
            logger = SessionLogger(log_dir=log_path)
            logger.start_session("Spa", "GT3")
            assert os.path.exists(log_path)


class TestStartSession:
    """Tests for start_session() method."""

    def test_start_session_returns_session_id(self):
        """Test start_session returns a session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            session_id = logger.start_session("Laguna Seca", "MX-5")
            assert session_id is not None
            assert len(session_id) > 0

    def test_start_session_stores_track_and_car(self):
        """Test start_session stores track and car in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Monza", "Ferrari 488")
            logger.end_session()

            # Find the log file
            files = list(Path(tmpdir).glob("*.json.gz"))
            assert len(files) == 1

            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["metadata"]["track"] == "Monza"
            assert data["metadata"]["car"] == "Ferrari 488"

    def test_start_session_records_start_time(self):
        """Test start_session records start timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "Porsche 911")
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert "start_time" in data["metadata"]
            assert data["metadata"]["start_time"] is not None

    def test_start_session_clears_previous_events(self):
        """Test starting new session clears events from previous session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)

            # First session
            logger.start_session("Track1", "Car1")
            logger.log_telemetry(make_snapshot(lap=1), make_strategy_state())
            logger.log_telemetry(make_snapshot(lap=2), make_strategy_state())
            logger.end_session()

            # Second session
            logger.start_session("Track2", "Car2")
            logger.log_telemetry(make_snapshot(lap=10), make_strategy_state())
            logger.end_session()

            files = sorted(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[1], "rt") as f:
                data = json.load(f)

            # Second session should only have 1 event
            assert len(data["events"]) == 1
            assert data["events"][0]["lap"] == 10


class TestLogTelemetry:
    """Tests for log_telemetry() method."""

    def test_log_telemetry_adds_event(self):
        """Test log_telemetry adds event to session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Silverstone", "AMG GT3")

            snapshot = make_snapshot(lap=5)
            state = make_strategy_state()
            logger.log_telemetry(snapshot, state)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert len(data["events"]) == 1
            assert data["events"][0]["event_type"] == "telemetry"

    def test_log_telemetry_includes_timestamp(self):
        """Test telemetry events include timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_telemetry(make_snapshot(), make_strategy_state())
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert "timestamp" in data["events"][0]

    def test_log_telemetry_includes_lap_number(self):
        """Test telemetry events include lap number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_telemetry(make_snapshot(lap=15), make_strategy_state())
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["events"][0]["lap"] == 15

    def test_log_telemetry_includes_snapshot_data(self):
        """Test telemetry events include snapshot data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            snapshot = make_snapshot(
                lap=10,
                position=3,
                fuel_level=12.5,
                tire_wear_rf=45.0,
            )
            logger.log_telemetry(snapshot, make_strategy_state())
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            event_data = data["events"][0]["data"]
            assert event_data["snapshot"]["position"] == 3
            assert event_data["snapshot"]["fuel_level"] == 12.5
            assert event_data["snapshot"]["tire_wear_rf"] == 45.0

    def test_log_telemetry_includes_strategy_state(self):
        """Test telemetry events include strategy state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            state = make_strategy_state(
                laps_of_fuel=8.5,
                worst_tire_corner="LF",
                urgency=Urgency.WARNING,
            )
            logger.log_telemetry(make_snapshot(), state)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            event_data = data["events"][0]["data"]
            assert event_data["strategy"]["laps_of_fuel"] == 8.5
            assert event_data["strategy"]["worst_tire_corner"] == "LF"
            assert event_data["strategy"]["urgency"] == "warning"


class TestLogLLMCall:
    """Tests for log_llm_call() method."""

    def test_log_llm_call_adds_event(self):
        """Test log_llm_call adds event to session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_llm_call("Test prompt", "Test response", 150.5)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert len(data["events"]) == 1
            assert data["events"][0]["event_type"] == "llm"

    def test_log_llm_call_includes_prompt(self):
        """Test LLM events include the prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_llm_call("Lap 10, P3, fuel critical", "Box now!", 100.0)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["events"][0]["data"]["prompt"] == "Lap 10, P3, fuel critical"

    def test_log_llm_call_includes_response(self):
        """Test LLM events include the response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_llm_call("Test prompt", "Box this lap for fuel.", 100.0)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["events"][0]["data"]["response"] == "Box this lap for fuel."

    def test_log_llm_call_includes_latency(self):
        """Test LLM events include latency in ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_llm_call("Prompt", "Response", 253.7)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["events"][0]["data"]["latency_ms"] == 253.7

    def test_log_llm_call_includes_timestamp(self):
        """Test LLM events include timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_llm_call("Prompt", "Response", 100.0)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert "timestamp" in data["events"][0]

    def test_log_llm_call_includes_current_lap(self):
        """Test LLM events include lap number from last telemetry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            # Log telemetry first to set current lap
            logger.log_telemetry(make_snapshot(lap=12), make_strategy_state())
            logger.log_llm_call("Prompt", "Response", 100.0)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            # LLM event should have lap from last telemetry
            llm_event = [e for e in data["events"] if e["event_type"] == "llm"][0]
            assert llm_event["lap"] == 12


class TestEndSession:
    """Tests for end_session() method."""

    def test_end_session_creates_gzip_file(self):
        """Test end_session creates a gzipped JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            assert len(files) == 1

    def test_end_session_file_has_timestamp_name(self):
        """Test output file is named with timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            filename = files[0].stem.replace(".json", "")
            # Should be something like 2024-01-15_14-30-00
            assert len(filename) > 10

    def test_end_session_file_is_valid_gzip(self):
        """Test output file is valid gzip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")
            logger.log_telemetry(make_snapshot(), make_strategy_state())
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            # Should not raise
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)
            assert data is not None

    def test_end_session_file_has_correct_structure(self):
        """Test output file has metadata and events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "events" in data
            assert "session_id" in data["metadata"]
            assert "start_time" in data["metadata"]
            assert "track" in data["metadata"]
            assert "car" in data["metadata"]

    def test_end_session_returns_file_path(self):
        """Test end_session returns path to created file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")
            path = logger.end_session()

            assert path is not None
            assert os.path.exists(path)
            assert path.endswith(".json.gz")


class TestMultipleEvents:
    """Tests for logging multiple events."""

    def test_events_logged_in_order(self):
        """Test events are logged in chronological order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            logger.log_telemetry(make_snapshot(lap=1), make_strategy_state())
            logger.log_telemetry(make_snapshot(lap=2), make_strategy_state())
            logger.log_llm_call("Prompt", "Response", 100.0)
            logger.log_telemetry(make_snapshot(lap=3), make_strategy_state())
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert len(data["events"]) == 4
            assert data["events"][0]["lap"] == 1
            assert data["events"][1]["lap"] == 2
            assert data["events"][2]["event_type"] == "llm"
            assert data["events"][3]["lap"] == 3

    def test_handles_many_events(self):
        """Test logger handles many events efficiently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")

            # Log 100 telemetry events
            for lap in range(1, 101):
                logger.log_telemetry(make_snapshot(lap=lap), make_strategy_state())

            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert len(data["events"]) == 100


class TestEdgeCases:
    """Tests for edge cases."""

    def test_log_without_start_session_is_ignored(self):
        """Test logging before start_session doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            # Should not raise
            logger.log_telemetry(make_snapshot(), make_strategy_state())
            logger.log_llm_call("Prompt", "Response", 100.0)

    def test_end_session_without_start_is_safe(self):
        """Test end_session without start_session doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            # Should not raise, returns None
            result = logger.end_session()
            assert result is None

    def test_special_characters_in_track_name(self):
        """Test track names with special characters are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Nürburgring-Nordschleife", "Porsche 911 GT3 R")
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["metadata"]["track"] == "Nürburgring-Nordschleife"

    def test_empty_response_in_llm_call(self):
        """Test empty LLM response is logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=tmpdir)
            logger.start_session("Spa", "GT3")
            logger.log_llm_call("Prompt", "", 50.0)
            logger.end_session()

            files = list(Path(tmpdir).glob("*.json.gz"))
            with gzip.open(files[0], "rt") as f:
                data = json.load(f)

            assert data["events"][0]["data"]["response"] == ""