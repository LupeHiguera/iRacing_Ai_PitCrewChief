"""
Strategy calculator for fuel and tire management.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .telemetry import TelemetrySnapshot


class Urgency(Enum):
    """Urgency level for pit stops."""
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class StrategyState:
    """Current strategy state calculated from telemetry."""
    # Fuel strategy
    fuel_per_lap: float
    laps_of_fuel: float
    pit_window: int  # Laps until pit required

    # Tire strategy
    worst_tire_corner: str  # "LF", "RF", "LR", "RR"
    worst_tire_wear: float  # Percentage worn

    # Pit recommendation
    needs_pit: bool
    pit_reason: Optional[str]
    urgency: Urgency


class StrategyCalculator:
    """Calculates race strategy from telemetry data."""

    # Number of laps to use for rolling average
    ROLLING_WINDOW = 5

    # Minimum fuel usage to count as a valid lap (filters out-laps, in-laps, cautions)
    MIN_FUEL_PER_LAP = 0.5

    def __init__(
        self,
        fuel_warning_laps: float = 5.0,
        fuel_critical_laps: float = 2.0,
        tire_warning_pct: float = 70.0,
        tire_critical_pct: float = 85.0,
    ):
        self._fuel_warning_laps = fuel_warning_laps
        self._fuel_critical_laps = fuel_critical_laps
        self._tire_warning_pct = tire_warning_pct
        self._tire_critical_pct = tire_critical_pct

        # Fuel tracking
        self._fuel_usage_history: deque[float] = deque(maxlen=self.ROLLING_WINDOW)
        self._last_lap: Optional[int] = None
        self._fuel_at_lap_start: Optional[float] = None  # Fuel level when current lap started

    def update(self, snapshot: TelemetrySnapshot) -> StrategyState:
        """
        Update strategy with new telemetry snapshot.
        Returns current strategy state.
        """
        # Track fuel usage per lap
        self._update_fuel_tracking(snapshot)

        # Calculate fuel metrics
        fuel_per_lap = self._calculate_fuel_per_lap()
        laps_of_fuel = self._calculate_laps_of_fuel(snapshot.fuel_level, fuel_per_lap)

        # Calculate tire metrics
        worst_corner, worst_wear = self._find_worst_tire(snapshot)

        # Determine urgency levels
        fuel_urgency, fuel_reason = self._get_fuel_urgency(laps_of_fuel)
        tire_urgency, tire_reason = self._get_tire_urgency(worst_wear)

        # Combined urgency (worst wins)
        if self._urgency_priority(fuel_urgency) >= self._urgency_priority(tire_urgency):
            urgency = fuel_urgency
            pit_reason = fuel_reason
        else:
            urgency = tire_urgency
            pit_reason = tire_reason

        # Needs pit if warning or critical
        needs_pit = urgency in (Urgency.WARNING, Urgency.CRITICAL)

        # Calculate pit window
        pit_window = self._calculate_pit_window(laps_of_fuel)

        return StrategyState(
            fuel_per_lap=fuel_per_lap,
            laps_of_fuel=laps_of_fuel,
            pit_window=pit_window,
            worst_tire_corner=worst_corner,
            worst_tire_wear=worst_wear,
            needs_pit=needs_pit,
            pit_reason=pit_reason,
            urgency=urgency,
        )

    def reset(self) -> None:
        """Reset strategy calculator (e.g., for new session)."""
        self._fuel_usage_history.clear()
        self._last_lap = None
        self._fuel_at_lap_start = None

    def _update_fuel_tracking(self, snapshot: TelemetrySnapshot) -> None:
        """Track fuel usage when a new lap is completed."""
        current_lap = snapshot.lap
        current_fuel = snapshot.fuel_level

        if self._last_lap is not None:
            # Check if we completed a lap (lap number increased)
            if current_lap > self._last_lap:
                # Calculate fuel used during the lap that just ended
                if self._fuel_at_lap_start is not None:
                    fuel_used = self._fuel_at_lap_start - current_fuel
                    # Only count valid laps (filters out-laps, in-laps, cautions)
                    if fuel_used >= self.MIN_FUEL_PER_LAP:
                        self._fuel_usage_history.append(fuel_used)
                # New lap started - record fuel at start of this lap
                self._fuel_at_lap_start = current_fuel
        else:
            # First sample - record fuel at start of current lap
            self._fuel_at_lap_start = current_fuel

        self._last_lap = current_lap

    def _calculate_fuel_per_lap(self) -> float:
        """Calculate average fuel usage per lap from history."""
        if not self._fuel_usage_history:
            return 0.0
        return sum(self._fuel_usage_history) / len(self._fuel_usage_history)

    def _calculate_laps_of_fuel(self, fuel_level: float, fuel_per_lap: float) -> float:
        """Calculate how many laps of fuel remain."""
        if fuel_per_lap <= 0:
            return float('inf')
        return fuel_level / fuel_per_lap

    def _calculate_pit_window(self, laps_of_fuel: float) -> int:
        """Calculate laps until pit is required."""
        if laps_of_fuel == float('inf'):
            return 999
        # Pit before running out, subtract critical threshold
        window = laps_of_fuel - self._fuel_critical_laps
        return max(0, int(window))

    def _find_worst_tire(self, snapshot: TelemetrySnapshot) -> tuple[str, float]:
        """Find the tire corner with worst wear."""
        tires = {
            "LF": snapshot.tire_wear_lf,
            "RF": snapshot.tire_wear_rf,
            "LR": snapshot.tire_wear_lr,
            "RR": snapshot.tire_wear_rr,
        }
        worst_corner = max(tires, key=tires.get)
        return worst_corner, tires[worst_corner]

    def _get_fuel_urgency(self, laps_of_fuel: float) -> tuple[Urgency, Optional[str]]:
        """Determine urgency level based on fuel remaining."""
        if laps_of_fuel <= self._fuel_critical_laps:
            return Urgency.CRITICAL, "Fuel critical - pit now"
        elif laps_of_fuel <= self._fuel_warning_laps:
            return Urgency.WARNING, "Fuel low - pit soon"
        return Urgency.OK, None

    def _get_tire_urgency(self, worst_wear: float) -> tuple[Urgency, Optional[str]]:
        """Determine urgency level based on tire wear."""
        if worst_wear >= self._tire_critical_pct:
            return Urgency.CRITICAL, "Tires critical - pit now"
        elif worst_wear >= self._tire_warning_pct:
            return Urgency.WARNING, "Tires worn - pit soon"
        return Urgency.OK, None

    def _urgency_priority(self, urgency: Urgency) -> int:
        """Get numeric priority for urgency comparison."""
        priorities = {
            Urgency.OK: 0,
            Urgency.INFO: 1,
            Urgency.WARNING: 2,
            Urgency.CRITICAL: 3,
        }
        return priorities[urgency]