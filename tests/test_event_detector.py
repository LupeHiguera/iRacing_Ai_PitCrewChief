"""
Tests for EventDetector tire-temp handling with the estimator.

iRacing has no live tire-temp channel, so tire-temp events run off the
per-corner estimate passed into detect_events (or off snapshot temps on a sim
that streams them). These tests cover the estimate path.
"""

from config import Config
from src.event_detector import EventDetector, EventType
from src.strategy import StrategyState, Urgency
from src.telemetry import TelemetrySnapshot


def make_snapshot(tire_temp: float = 0.0) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        lap=5, lap_pct=0.5, position=5,
        fuel_level=15.0, fuel_level_pct=0.5, fuel_use_per_hour=2.5,
        tire_wear_lf=10.0, tire_wear_rf=10.0, tire_wear_lr=10.0, tire_wear_rr=10.0,
        session_time_remain=1800.0, session_laps_remain=20,
        last_lap_time=90.0, best_lap_time=88.0,
        on_pit_road=False, is_on_track=True,
        tire_temp_lf_m=tire_temp, tire_temp_rf_m=tire_temp,
        tire_temp_lr_m=tire_temp, tire_temp_rr_m=tire_temp,
        track_temp_c=30.0, air_temp_c=25.0,
    )


def make_state() -> StrategyState:
    return StrategyState(
        fuel_per_lap=2.5, laps_of_fuel=12.0, pit_window=8,
        worst_tire_corner="RF", worst_tire_wear=10.0,
        needs_pit=False, pit_reason=None, urgency=Urgency.OK,
    )


def test_hot_event_fires_from_estimated_temps():
    det = EventDetector(Config())  # estimation enabled by default
    events = det.detect_events(
        make_snapshot(), make_state(),
        tire_temps_est={"LF": 85, "RF": 120, "LR": 85, "RR": 85},
    )
    assert any(e.event_type == EventType.TIRE_HOT for e in events)


def test_cold_event_fires_from_estimated_temps():
    det = EventDetector(Config())
    events = det.detect_events(
        make_snapshot(), make_state(),
        tire_temps_est={"LF": 30, "RF": 30, "LR": 30, "RR": 30},
    )
    assert any(e.event_type == EventType.TIRE_COLD for e in events)


def test_no_tire_temp_events_when_no_estimate_and_raw_dead():
    # Estimation enabled but no estimate passed and snapshot temps read 0 ->
    # the all-zero guard means no temp events fire off frozen/absent data.
    det = EventDetector(Config())
    events = det.detect_events(make_snapshot(tire_temp=0.0), make_state(), tire_temps_est=None)
    tire_types = {EventType.TIRE_HOT, EventType.TIRE_COLD, EventType.TIRE_OPTIMAL}
    assert not any(e.event_type in tire_types for e in events)


def test_lockup_wheelspin_stay_disabled_with_estimate():
    # Even a huge jump in estimated temps must not produce lockup/wheelspin:
    # those need truly live sub-second temps.
    det = EventDetector(Config())
    det.detect_events(make_snapshot(), make_state(),
                      tire_temps_est={"LF": 40, "RF": 40, "LR": 40, "RR": 40})
    events = det.detect_events(make_snapshot(), make_state(),
                               tire_temps_est={"LF": 120, "RF": 120, "LR": 120, "RR": 120})
    assert not any(e.event_type in (EventType.LOCKUP, EventType.WHEELSPIN) for e in events)
