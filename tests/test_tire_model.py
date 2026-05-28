"""
Tests for the TireStateEstimator.

iRacing has no live tire-temp channel, so we estimate per-corner temp/wear from
live load telemetry, anchored to the real pit-measured values. These tests pin
the qualitative behavior (warm-up, braking heats fronts, cornering heats the
outer pair, cooling, clamping, pit re-anchor, monotonic wear) rather than exact
tuned magnitudes.
"""

import pytest

from config import Config
from src.telemetry import TelemetrySnapshot
from src.tire_model import TireStateEstimator, TireEstimate


def make_snapshot(
    *,
    speed: float = 0.0,
    lat_accel: float = 0.0,
    long_accel: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
    on_pit_road: bool = False,
    track_temp_c: float = 30.0,
    air_temp_c: float = 25.0,
    tire_temp: float = 35.0,   # uniform measured temp per corner
    tire_wear: float = 0.0,    # uniform measured wear per corner
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        lap=5, lap_pct=0.5, position=5,
        fuel_level=15.0, fuel_level_pct=0.5, fuel_use_per_hour=2.5,
        tire_wear_lf=tire_wear, tire_wear_rf=tire_wear,
        tire_wear_lr=tire_wear, tire_wear_rr=tire_wear,
        session_time_remain=1800.0, session_laps_remain=20,
        last_lap_time=90.0, best_lap_time=88.0,
        on_pit_road=on_pit_road, is_on_track=True,
        tire_temp_lf_l=tire_temp, tire_temp_lf_m=tire_temp, tire_temp_lf_r=tire_temp,
        tire_temp_rf_l=tire_temp, tire_temp_rf_m=tire_temp, tire_temp_rf_r=tire_temp,
        tire_temp_lr_l=tire_temp, tire_temp_lr_m=tire_temp, tire_temp_lr_r=tire_temp,
        tire_temp_rr_l=tire_temp, tire_temp_rr_m=tire_temp, tire_temp_rr_r=tire_temp,
        track_temp_c=track_temp_c, air_temp_c=air_temp_c,
        speed=speed, lat_accel=lat_accel, long_accel=long_accel,
        throttle=throttle, brake=brake,
    )


def _estimator():
    return TireStateEstimator(Config())


def test_anchor_sets_temps_to_measured():
    est = _estimator()
    est.anchor(make_snapshot(tire_temp=42.0, tire_wear=12.0))
    for c in ("LF", "RF", "LR", "RR"):
        assert est._temps[c] == pytest.approx(42.0)
        assert est._wear[c] == pytest.approx(12.0)


def test_zero_measured_temp_falls_back_to_ambient():
    cfg = Config()
    est = TireStateEstimator(cfg)
    est.anchor(make_snapshot(tire_temp=0.0, track_temp_c=30.0))
    expected = 30.0 + cfg.tire_est_ambient_offset_c
    for c in ("LF", "RF", "LR", "RR"):
        assert est._temps[c] == pytest.approx(expected)


def test_cold_start_warms_up_under_load():
    est = _estimator()
    # Cold anchor, then push for ~a lap worth of ticks.
    snap = make_snapshot(speed=60.0, lat_accel=12.0, throttle=0.8, tire_temp=35.0)
    out = None
    for _ in range(90):
        out = est.update(snap, dt=1.0)
    avg = sum(out.temps.values()) / 4
    assert avg > 50.0  # meaningfully above the 35C cold anchor


def test_braking_heats_fronts_more_than_rears():
    est = _estimator()
    snap = make_snapshot(speed=55.0, long_accel=-25.0, brake=1.0, tire_temp=35.0)
    out = None
    for _ in range(40):
        out = est.update(snap, dt=1.0)
    front = (out.temps["LF"] + out.temps["RF"]) / 2
    rear = (out.temps["LR"] + out.temps["RR"]) / 2
    assert front > rear


def test_cornering_heats_outer_pair():
    est = _estimator()
    # Positive lat_accel loads the right-hand tires in our convention.
    snap = make_snapshot(speed=50.0, lat_accel=18.0, tire_temp=35.0)
    out = None
    for _ in range(40):
        out = est.update(snap, dt=1.0)
    right = (out.temps["RF"] + out.temps["RR"]) / 2
    left = (out.temps["LF"] + out.temps["LR"]) / 2
    assert right > left


def test_straight_cools_toward_ambient():
    est = _estimator()
    hot = make_snapshot(speed=60.0, lat_accel=15.0, throttle=1.0, tire_temp=35.0)
    for _ in range(60):
        est.update(hot, dt=1.0)
    warm_avg = sum(est._temps.values()) / 4
    # Now coast at speed with no load -> should cool.
    coast = make_snapshot(speed=60.0, tire_temp=35.0)
    out = None
    for _ in range(60):
        out = est.update(coast, dt=1.0)
    assert sum(out.temps.values()) / 4 < warm_avg


def test_temps_clamped_to_band():
    cfg = Config()
    est = TireStateEstimator(cfg)
    extreme = make_snapshot(speed=90.0, lat_accel=40.0, long_accel=-40.0,
                            throttle=1.0, brake=1.0, tire_temp=35.0)
    out = None
    for _ in range(500):
        out = est.update(extreme, dt=1.0)
    for c in ("LF", "RF", "LR", "RR"):
        assert out.temps[c] <= cfg.tire_est_max_temp_c + 1e-6
    # Idle settles to the resting (ambient) temp, not below it.
    idle = make_snapshot(speed=0.0, track_temp_c=30.0, tire_temp=35.0)
    for _ in range(400):
        out = est.update(idle, dt=1.0)
    ambient = 30.0 + cfg.tire_est_ambient_offset_c
    for c in ("LF", "RF", "LR", "RR"):
        assert out.temps[c] == pytest.approx(ambient, abs=1.5)


def test_pit_exit_reanchors_to_measured():
    est = _estimator()
    # Build up heat on track.
    on_track = make_snapshot(speed=60.0, lat_accel=15.0, throttle=1.0, tire_temp=35.0)
    for _ in range(60):
        est.update(on_track, dt=1.0)
    assert sum(est._temps.values()) / 4 > 45.0
    # Enter the pit, then exit with a fresh cold measurement.
    est.update(make_snapshot(speed=5.0, on_pit_road=True, tire_temp=35.0), dt=1.0)
    out = est.update(make_snapshot(speed=5.0, on_pit_road=False, tire_temp=34.0), dt=1.0)
    # Built-up heat (>45C) is discarded: re-anchored back near the fresh cold
    # measurement (~34C, trending toward the ~38C resting temp).
    for c in ("LF", "RF", "LR", "RR"):
        assert out.temps[c] < 42.0
        assert out.temps[c] == pytest.approx(34.0, abs=5.0)


def test_wear_is_monotonic_and_scales_with_load():
    est = _estimator()
    push = make_snapshot(speed=60.0, lat_accel=15.0, throttle=1.0, tire_wear=0.0)
    prev = 0.0
    for _ in range(50):
        out = est.update(push, dt=1.0)
        cur = sum(out.wear.values()) / 4
        assert cur >= prev  # never decreases mid-stint
        prev = cur
    assert prev > 0.0  # accrued some wear


def test_wear_reanchors_to_measured_at_pit():
    est = _estimator()
    for _ in range(50):
        est.update(make_snapshot(speed=60.0, throttle=1.0, tire_wear=0.0), dt=1.0)
    est.update(make_snapshot(speed=5.0, on_pit_road=True, tire_wear=0.0), dt=1.0)
    out = est.update(make_snapshot(speed=5.0, on_pit_road=False, tire_wear=40.0), dt=1.0)
    for c in ("LF", "RF", "LR", "RR"):
        assert out.wear[c] == pytest.approx(40.0, abs=2.0)


def test_tire_deg_trait_scales_wear():
    cfg = Config()
    low = TireStateEstimator(cfg); low._deg_mult = 0.6
    high = TireStateEstimator(cfg); high._deg_mult = 1.5
    snap = make_snapshot(speed=60.0, throttle=1.0, tire_wear=0.0)
    for _ in range(40):
        lo = low.update(snap, dt=1.0)
        hi = high.update(snap, dt=1.0)
    assert sum(hi.wear.values()) > sum(lo.wear.values())


def test_estimate_prompt_keys():
    out = TireEstimate(
        temps={"LF": 95.4, "RF": 102.6, "LR": 88.1, "RR": 91.9},
        wear={"LF": 15.2, "RF": 22.8, "LR": 12.1, "RR": 14.4},
    )
    assert out.temps_prompt() == {"fl": 95, "fr": 103, "rl": 88, "rr": 92}
    assert out.wear_prompt() == {"fl": 15, "fr": 23, "rl": 12, "rr": 14}


def test_estimate_error():
    err = TireStateEstimator.estimate_error(
        {"LF": 90, "RF": 100, "LR": 80, "RR": 85},
        {"LF": 88, "RF": 105, "LR": 82, "RR": 85},
    )
    assert err == {"LF": 2, "RF": 5, "LR": 2, "RR": 0}
