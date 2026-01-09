"""
Tests for StrategyCalculator class.
"""

import pytest

from src.telemetry import TelemetrySnapshot
from src.strategy import StrategyCalculator, StrategyState, Urgency


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
    """Helper to create test snapshots with defaults."""
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


class TestStrategyCalculatorInitialization:
    """Tests for StrategyCalculator initialization."""

    def test_can_instantiate_with_defaults(self):
        """Test that calculator can be created with default thresholds."""
        calc = StrategyCalculator()
        assert calc is not None

    def test_can_instantiate_with_custom_thresholds(self):
        """Test that calculator accepts custom threshold values."""
        calc = StrategyCalculator(
            fuel_warning_laps=4.0,
            fuel_critical_laps=1.5,
            tire_warning_pct=65.0,
            tire_critical_pct=80.0,
        )
        assert calc is not None


class TestFuelStrategy:
    """Tests for fuel consumption tracking and strategy."""

    def test_calculates_fuel_per_lap_after_one_lap(self):
        """Test fuel per lap calculation after completing a lap."""
        calc = StrategyCalculator()

        # Start of lap 1 with full fuel
        snap1 = make_snapshot(lap=1, lap_pct=0.0, fuel_level=20.0)
        calc.update(snap1)

        # End of lap 1
        snap2 = make_snapshot(lap=2, lap_pct=0.0, fuel_level=17.5)
        state = calc.update(snap2)

        assert state.fuel_per_lap == pytest.approx(2.5, abs=0.1)

    def test_fuel_per_lap_uses_rolling_average(self):
        """Test that fuel consumption uses rolling average of recent laps."""
        calc = StrategyCalculator()

        # Simulate 3 laps with varying fuel usage
        snapshots = [
            make_snapshot(lap=1, lap_pct=0.0, fuel_level=20.0),
            make_snapshot(lap=2, lap_pct=0.0, fuel_level=17.5),  # 2.5 used
            make_snapshot(lap=3, lap_pct=0.0, fuel_level=14.5),  # 3.0 used
            make_snapshot(lap=4, lap_pct=0.0, fuel_level=12.0),  # 2.5 used
        ]

        state = None
        for snap in snapshots:
            state = calc.update(snap)

        # Average of 2.5, 3.0, 2.5 = 2.67
        assert state.fuel_per_lap == pytest.approx(2.67, abs=0.1)

    def test_calculates_laps_of_fuel_remaining(self):
        """Test calculation of laps remaining on current fuel."""
        calc = StrategyCalculator()

        # After establishing fuel usage
        snap1 = make_snapshot(lap=1, fuel_level=20.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=18.0)  # 2.0 per lap
        state = calc.update(snap2)

        # 18.0 fuel / 2.0 per lap = 9 laps
        assert state.laps_of_fuel == pytest.approx(9.0, abs=0.5)

    def test_pit_window_calculation(self):
        """Test pit window calculation based on fuel and session remaining."""
        calc = StrategyCalculator()

        snap1 = make_snapshot(lap=1, fuel_level=20.0, session_laps_remain=30)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=18.0, session_laps_remain=29)
        state = calc.update(snap2)

        # 9 laps of fuel, should pit before running out
        # pit_window = laps_of_fuel - critical_threshold
        assert state.pit_window > 0


class TestFuelUrgency:
    """Tests for fuel urgency levels."""

    def test_fuel_ok_when_plenty_of_fuel(self):
        """Test urgency is OK when fuel is plentiful."""
        calc = StrategyCalculator(fuel_warning_laps=5.0, fuel_critical_laps=2.0)

        # Simulate enough fuel for 10 laps
        snap1 = make_snapshot(lap=1, fuel_level=20.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=18.0)  # 2.0 per lap, 9 laps left
        state = calc.update(snap2)

        assert state.urgency == Urgency.OK
        assert state.needs_pit is False

    def test_fuel_warning_when_low(self):
        """Test urgency is WARNING when fuel is getting low."""
        calc = StrategyCalculator(fuel_warning_laps=5.0, fuel_critical_laps=2.0)

        # Simulate ~4 laps of fuel (below 5 lap warning threshold)
        snap1 = make_snapshot(lap=1, fuel_level=10.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=8.0)  # 2.0 per lap, 4 laps left
        state = calc.update(snap2)

        assert state.urgency == Urgency.WARNING
        assert state.needs_pit is True
        assert "fuel" in state.pit_reason.lower()

    def test_fuel_critical_when_very_low(self):
        """Test urgency is CRITICAL when fuel is critically low."""
        calc = StrategyCalculator(fuel_warning_laps=5.0, fuel_critical_laps=2.0)

        # Simulate ~1.5 laps of fuel (below 2 lap critical threshold)
        snap1 = make_snapshot(lap=1, fuel_level=5.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=3.0)  # 2.0 per lap, 1.5 laps left
        state = calc.update(snap2)

        assert state.urgency == Urgency.CRITICAL
        assert state.needs_pit is True


class TestTireStrategy:
    """Tests for tire wear tracking and strategy."""

    def test_identifies_worst_tire_corner(self):
        """Test that worst tire corner is correctly identified."""
        calc = StrategyCalculator()

        snap = make_snapshot(
            tire_wear_lf=30.0,
            tire_wear_rf=45.0,  # Worst
            tire_wear_lr=25.0,
            tire_wear_rr=35.0,
        )
        state = calc.update(snap)

        assert state.worst_tire_corner == "RF"
        assert state.worst_tire_wear == 45.0

    def test_tire_ok_when_fresh(self):
        """Test urgency is OK when tires are fresh."""
        calc = StrategyCalculator(tire_warning_pct=70.0, tire_critical_pct=85.0)

        snap = make_snapshot(
            tire_wear_lf=20.0,
            tire_wear_rf=25.0,
            tire_wear_lr=18.0,
            tire_wear_rr=22.0,
        )
        state = calc.update(snap)

        assert state.urgency == Urgency.OK

    def test_tire_warning_when_worn(self):
        """Test urgency is WARNING when tires are worn."""
        calc = StrategyCalculator(tire_warning_pct=70.0, tire_critical_pct=85.0)

        snap = make_snapshot(
            tire_wear_lf=65.0,
            tire_wear_rf=75.0,  # Above 70% warning
            tire_wear_lr=60.0,
            tire_wear_rr=68.0,
        )
        state = calc.update(snap)

        assert state.urgency == Urgency.WARNING
        assert state.needs_pit is True
        assert "tire" in state.pit_reason.lower()

    def test_tire_critical_when_very_worn(self):
        """Test urgency is CRITICAL when tires are critically worn."""
        calc = StrategyCalculator(tire_warning_pct=70.0, tire_critical_pct=85.0)

        snap = make_snapshot(
            tire_wear_lf=80.0,
            tire_wear_rf=90.0,  # Above 85% critical
            tire_wear_lr=75.0,
            tire_wear_rr=82.0,
        )
        state = calc.update(snap)

        assert state.urgency == Urgency.CRITICAL
        assert state.needs_pit is True


class TestCombinedStrategy:
    """Tests for combined fuel and tire strategy."""

    def test_fuel_critical_takes_priority_over_tire_warning(self):
        """Test that critical fuel urgency takes priority over tire warning."""
        calc = StrategyCalculator(
            fuel_warning_laps=5.0,
            fuel_critical_laps=2.0,
            tire_warning_pct=70.0,
            tire_critical_pct=85.0,
        )

        # Low fuel (critical) + worn tires (warning)
        snap1 = make_snapshot(lap=1, fuel_level=5.0, tire_wear_rf=75.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=3.0, tire_wear_rf=76.0)
        state = calc.update(snap2)

        assert state.urgency == Urgency.CRITICAL
        assert "fuel" in state.pit_reason.lower()

    def test_worst_urgency_wins(self):
        """Test that the worst urgency level is reported."""
        calc = StrategyCalculator(
            fuel_warning_laps=5.0,
            fuel_critical_laps=2.0,
            tire_warning_pct=70.0,
            tire_critical_pct=85.0,
        )

        # Fuel warning + tire critical = CRITICAL
        snap1 = make_snapshot(lap=1, fuel_level=10.0, tire_wear_rf=90.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=8.0, tire_wear_rf=91.0)
        state = calc.update(snap2)

        assert state.urgency == Urgency.CRITICAL


class TestStrategyReset:
    """Tests for strategy calculator reset."""

    def test_reset_clears_fuel_history(self):
        """Test that reset clears fuel consumption history."""
        calc = StrategyCalculator()

        # Build up some history
        snap1 = make_snapshot(lap=1, fuel_level=20.0)
        calc.update(snap1)
        snap2 = make_snapshot(lap=2, fuel_level=18.0)
        calc.update(snap2)

        # Reset
        calc.reset()

        # Should start fresh
        snap3 = make_snapshot(lap=1, fuel_level=20.0)
        state = calc.update(snap3)

        # No history yet, fuel_per_lap should be 0 or default
        assert state.fuel_per_lap == 0.0 or state.laps_of_fuel == float('inf')

    def test_reset_clears_tire_history(self):
        """Test that reset clears any tire tracking data."""
        calc = StrategyCalculator()

        snap = make_snapshot(
            tire_wear_lf=40.0,
            tire_wear_rf=50.0,
            tire_wear_lr=35.0,
            tire_wear_rr=38.0,
        )
        calc.update(snap)
        calc.reset()

        # After reset, state should reflect new data only
        fresh_snap = make_snapshot(
            tire_wear_lf=5.0,
            tire_wear_rf=10.0,
            tire_wear_lr=5.0,
            tire_wear_rr=5.0,
        )
        state = calc.update(fresh_snap)

        assert state.worst_tire_wear == 10.0
        assert state.worst_tire_corner == "RF"