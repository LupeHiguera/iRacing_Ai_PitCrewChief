"""
Tire-state estimator.

iRacing has no live tire-temperature or pressure channel -- the SDK only
refreshes tire temps/wear when the pit crew measures the tires (spawn + each pit
stop), so they are frozen during a stint. This module reconstructs a per-corner
tire thermal + wear estimate from the live load telemetry that IS streamed
(speed, lateral/longitudinal G, throttle, brake), anchored to the real measured
values at session start and every pit exit.

The output is an ESTIMATE, not ground truth. It is fed to the model prompt and
the tire events, and labeled "estimated" in the overlay. The estimate is anchored
to iRacing's accurate pit-stop measurements, so it starts from a real value each
stint and tracks the dynamics (cornering load, braking, traction, cooling on
straights, and the live track/air ambient drift) in between.
"""

from dataclasses import dataclass
from typing import Optional

from config import Config
from .telemetry import TelemetrySnapshot
from .metadata import get_car_metadata

G = 9.81  # m/s^2

# Tire degradation multiplier by car metadata trait.
_DEG_MULT = {"low": 0.6, "medium": 1.0, "high": 1.5}

CORNERS = ("LF", "RF", "LR", "RR")


@dataclass
class TireEstimate:
    """Per-corner estimated tire state. Temps in C, wear in % worn (0=new)."""
    temps: dict  # {"LF","RF","LR","RR"} -> float
    wear: dict   # {"LF","RF","LR","RR"} -> float
    is_estimated: bool = True

    def temps_prompt(self) -> dict:
        """Temps keyed fl/fr/rl/rr (rounded) for the model JSON prompt."""
        return {
            "fl": round(self.temps["LF"]),
            "fr": round(self.temps["RF"]),
            "rl": round(self.temps["LR"]),
            "rr": round(self.temps["RR"]),
        }

    def wear_prompt(self) -> dict:
        """Wear keyed fl/fr/rl/rr (rounded) for the model JSON prompt."""
        return {
            "fl": round(self.wear["LF"]),
            "fr": round(self.wear["RF"]),
            "rl": round(self.wear["LR"]),
            "rr": round(self.wear["RR"]),
        }


class TireStateEstimator:
    """Stateful per-corner tire temp/wear estimator anchored to pit measurements."""

    def __init__(self, config: Config):
        self._config = config
        self._temps: dict = {c: 0.0 for c in CORNERS}
        self._wear: dict = {c: 0.0 for c in CORNERS}
        self._deg_mult: float = 1.0
        self._anchored: bool = False
        self._last_on_pit_road: bool = False

    def set_car(self, car_name: str) -> None:
        """Set the car so wear accrual scales with its tire_deg trait."""
        meta = get_car_metadata(car_name)
        tire_deg = meta.get("tire_deg", "medium") if meta else "medium"
        self._deg_mult = _DEG_MULT.get(tire_deg, 1.0)

    def reset(self) -> None:
        """Clear state (e.g. new session)."""
        self._temps = {c: 0.0 for c in CORNERS}
        self._wear = {c: 0.0 for c in CORNERS}
        self._anchored = False
        self._last_on_pit_road = False

    def _ambient(self, snapshot: TelemetrySnapshot) -> float:
        """Live resting tire temp: track surface temp + offset, air as fallback."""
        track = snapshot.track_temp_c
        if track <= 0:
            track = snapshot.air_temp_c if snapshot.air_temp_c > 0 else 25.0
        return track + self._config.tire_est_ambient_offset_c

    @staticmethod
    def measured_temps(snapshot: TelemetrySnapshot) -> dict:
        """Average the (pit-measured) L/M/R temps per corner from the snapshot."""
        return {
            "LF": (snapshot.tire_temp_lf_l + snapshot.tire_temp_lf_m + snapshot.tire_temp_lf_r) / 3,
            "RF": (snapshot.tire_temp_rf_l + snapshot.tire_temp_rf_m + snapshot.tire_temp_rf_r) / 3,
            "LR": (snapshot.tire_temp_lr_l + snapshot.tire_temp_lr_m + snapshot.tire_temp_lr_r) / 3,
            "RR": (snapshot.tire_temp_rr_l + snapshot.tire_temp_rr_m + snapshot.tire_temp_rr_r) / 3,
        }

    def anchor(self, snapshot: TelemetrySnapshot) -> None:
        """Reset the estimate to iRacing's freshly measured values.

        Called at the first sample and on every pit exit, where snapshot tire
        temps/wear hold an accurate crew measurement. Cars that report 0 temp
        fall back to the live ambient.
        """
        ambient = self._ambient(snapshot)
        measured = self.measured_temps(snapshot)
        for c in CORNERS:
            self._temps[c] = measured[c] if measured[c] > 0 else ambient
        self._wear = {
            "LF": snapshot.tire_wear_lf,
            "RF": snapshot.tire_wear_rf,
            "LR": snapshot.tire_wear_lr,
            "RR": snapshot.tire_wear_rr,
        }
        self._anchored = True

    def _corner_loads(self, snapshot: TelemetrySnapshot) -> dict:
        """Normalized per-corner load (~0..1.5) from lateral/longitudinal demand."""
        lat = snapshot.lat_accel / G
        # Lateral load transfers to the outer pair. Sign convention: positive
        # LatAccel loads the right-hand tires (flip if a car proves otherwise --
        # only affects which side is named, not the magnitude of heating).
        right_load = max(0.0, lat)
        left_load = max(0.0, -lat)
        # Braking loads/heats the fronts; traction loads/heats the rears.
        braking = max(0.0, -snapshot.long_accel / G) + snapshot.brake
        traction = max(0.0, snapshot.long_accel / G) + snapshot.throttle
        base = 0.15
        return {
            "LF": base + left_load + 0.6 * braking,
            "RF": base + right_load + 0.6 * braking,
            "LR": base + left_load + 0.6 * traction,
            "RR": base + right_load + 0.6 * traction,
        }

    def update(self, snapshot: TelemetrySnapshot, dt: float = 1.0) -> TireEstimate:
        """Advance the estimate one tick and return the current TireEstimate."""
        # Anchor on first sample and on pit exit (crew just measured the tires).
        pit_exit = self._last_on_pit_road and not snapshot.on_pit_road
        if not self._anchored or pit_exit:
            self.anchor(snapshot)
        self._last_on_pit_road = snapshot.on_pit_road

        ambient = self._ambient(snapshot)
        loads = self._corner_loads(snapshot)
        speed = max(0.0, snapshot.speed)
        cfg = self._config

        for c in CORNERS:
            load = loads[c]
            # Heat from slip work ~ load x speed; cooling toward ambient with
            # extra airflow cooling at speed.
            heat_in = cfg.tire_est_heat_gain * load * speed
            # Cooling relaxes toward the resting (ambient) temp; when a tire is
            # below ambient this term is negative, so it warms gradually toward
            # ambient rather than being hard-snapped up.
            cooling = cfg.tire_est_cool_rate * (self._temps[c] - ambient) * (1.0 + 0.02 * speed)
            self._temps[c] += (heat_in - cooling) * dt
            self._temps[c] = max(0.0, min(self._temps[c], cfg.tire_est_max_temp_c))

            # Wear accrues with load and speed, scaled by the car's tire_deg.
            self._wear[c] = min(
                100.0,
                self._wear[c] + cfg.tire_est_wear_gain * load * (speed / 50.0) * self._deg_mult * dt,
            )

        return TireEstimate(temps=dict(self._temps), wear=dict(self._wear))

    @staticmethod
    def estimate_error(estimated: dict, measured: dict) -> dict:
        """Per-corner |estimated - measured| temp error, for the pit accuracy check."""
        return {c: abs(estimated.get(c, 0.0) - measured.get(c, 0.0)) for c in CORNERS}
