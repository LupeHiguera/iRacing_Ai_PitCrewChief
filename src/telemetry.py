"""
Telemetry reader for iRacing using pyirsdk.
"""

from dataclasses import dataclass
from typing import Optional

import irsdk


@dataclass
class TelemetrySnapshot:
    """Snapshot of current telemetry data from iRacing."""
    # Lap info
    lap: int
    lap_pct: float
    position: int

    # Fuel
    fuel_level: float
    fuel_level_pct: float
    fuel_use_per_hour: float

    # Tire wear (percentage worn, 0=new, 100=gone)
    tire_wear_lf: float
    tire_wear_rf: float
    tire_wear_lr: float
    tire_wear_rr: float

    # Session info
    session_time_remain: float
    session_laps_remain: int

    # Lap times
    last_lap_time: float
    best_lap_time: float

    # Status
    on_pit_road: bool
    is_on_track: bool

    # Tire temps (Celsius) - Left/Middle/Right across tire surface
    tire_temp_lf_l: float = 0.0
    tire_temp_lf_m: float = 0.0
    tire_temp_lf_r: float = 0.0
    tire_temp_rf_l: float = 0.0
    tire_temp_rf_m: float = 0.0
    tire_temp_rf_r: float = 0.0
    tire_temp_lr_l: float = 0.0
    tire_temp_lr_m: float = 0.0
    tire_temp_lr_r: float = 0.0
    tire_temp_rr_l: float = 0.0
    tire_temp_rr_m: float = 0.0
    tire_temp_rr_r: float = 0.0

    # Tire pressures (kPa)
    tire_pressure_lf: float = 0.0
    tire_pressure_rf: float = 0.0
    tire_pressure_lr: float = 0.0
    tire_pressure_rr: float = 0.0

    # Gaps (seconds)
    gap_ahead_sec: Optional[float] = None
    gap_behind_sec: Optional[float] = None

    # Track conditions
    track_temp_c: float = 0.0
    air_temp_c: float = 0.0

    # Brake temps (Celsius)
    brake_temp_lf: float = 0.0
    brake_temp_rf: float = 0.0
    brake_temp_lr: float = 0.0
    brake_temp_rr: float = 0.0


class TelemetryReader:
    """Reads telemetry data from iRacing via pyirsdk."""

    def __init__(self):
        self._ir: Optional[irsdk.IRSDK] = None
        self._connected: bool = False

    def connect(self) -> bool:
        """Connect to iRacing. Returns True if successful."""
        if self._ir is None:
            self._ir = irsdk.IRSDK()

        if self._ir.startup():
            self._connected = True
            return True

        self._connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect from iRacing."""
        if self._ir is not None:
            self._ir.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to iRacing."""
        return self._connected

    def get_track_name(self) -> str:
        """Get the current track name from session info."""
        if not self._connected or self._ir is None:
            return "Unknown Track"
        try:
            weekend_info = self._ir['WeekendInfo']
            if weekend_info:
                return weekend_info.get('TrackDisplayName', 'Unknown Track')
        except (KeyError, TypeError):
            pass
        return "Unknown Track"

    def get_car_name(self) -> str:
        """Get the player's car name from session info."""
        if not self._connected or self._ir is None:
            return "Unknown Car"
        try:
            driver_info = self._ir['DriverInfo']
            if driver_info:
                drivers = driver_info.get('Drivers', [])
                player_idx = self._ir['PlayerCarIdx']
                if player_idx is not None and player_idx < len(drivers):
                    return drivers[player_idx].get('CarScreenName', 'Unknown Car')
        except (KeyError, TypeError, IndexError):
            pass
        return "Unknown Car"

    def get_snapshot(self) -> Optional[TelemetrySnapshot]:
        """Get current telemetry snapshot. Returns None if not connected."""
        if not self._connected or self._ir is None:
            return None

        try:
            # Get raw values, handle None gracefully
            lap = self._ir['Lap']
            if lap is None:
                return None

            # Calculate gaps to cars ahead and behind
            gap_ahead, gap_behind = self._calculate_gaps()

            return TelemetrySnapshot(
                lap=lap,
                lap_pct=self._ir['LapDistPct'] or 0.0,
                position=self._ir['PlayerCarPosition'] or 0,
                fuel_level=self._ir['FuelLevel'] or 0.0,
                fuel_level_pct=self._ir['FuelLevelPct'] or 0.0,
                fuel_use_per_hour=self._ir['FuelUsePerHour'] or 0.0,
                tire_wear_lf=self._calculate_tire_wear('LF'),
                tire_wear_rf=self._calculate_tire_wear('RF'),
                tire_wear_lr=self._calculate_tire_wear('LR'),
                tire_wear_rr=self._calculate_tire_wear('RR'),
                session_time_remain=self._ir['SessionTimeRemain'] or 0.0,
                session_laps_remain=self._ir['SessionLapsRemain'] or 0,
                last_lap_time=self._ir['LapLastLapTime'] or 0.0,
                best_lap_time=self._ir['LapBestLapTime'] or 0.0,
                on_pit_road=bool(self._ir['OnPitRoad']),
                is_on_track=bool(self._ir['IsOnTrack']),
                # Tire temps
                tire_temp_lf_l=self._ir['LFtempCL'] or 0.0,
                tire_temp_lf_m=self._ir['LFtempCM'] or 0.0,
                tire_temp_lf_r=self._ir['LFtempCR'] or 0.0,
                tire_temp_rf_l=self._ir['RFtempCL'] or 0.0,
                tire_temp_rf_m=self._ir['RFtempCM'] or 0.0,
                tire_temp_rf_r=self._ir['RFtempCR'] or 0.0,
                tire_temp_lr_l=self._ir['LRtempCL'] or 0.0,
                tire_temp_lr_m=self._ir['LRtempCM'] or 0.0,
                tire_temp_lr_r=self._ir['LRtempCR'] or 0.0,
                tire_temp_rr_l=self._ir['RRtempCL'] or 0.0,
                tire_temp_rr_m=self._ir['RRtempCM'] or 0.0,
                tire_temp_rr_r=self._ir['RRtempCR'] or 0.0,
                # Tire pressures
                tire_pressure_lf=self._ir['LFpressure'] or 0.0,
                tire_pressure_rf=self._ir['RFpressure'] or 0.0,
                tire_pressure_lr=self._ir['LRpressure'] or 0.0,
                tire_pressure_rr=self._ir['RRpressure'] or 0.0,
                # Gaps
                gap_ahead_sec=gap_ahead,
                gap_behind_sec=gap_behind,
                # Track conditions
                track_temp_c=self._ir['TrackTempCrew'] or 0.0,
                air_temp_c=self._ir['AirTemp'] or 0.0,
                # Brake temps
                brake_temp_lf=self._ir['LFbrakeTemp'] or 0.0,
                brake_temp_rf=self._ir['RFbrakeTemp'] or 0.0,
                brake_temp_lr=self._ir['LRbrakeTemp'] or 0.0,
                brake_temp_rr=self._ir['RRbrakeTemp'] or 0.0,
            )
        except Exception:
            return None

    def _calculate_tire_wear(self, corner: str) -> float:
        """
        Calculate tire wear percentage for a corner.

        iRacing reports wear as 1.0 = new, 0.0 = worn.
        We return percentage worn (0 = new, 100 = gone).
        Uses minimum of L/M/R values (worst wear spot).
        """
        if self._ir is None:
            return 0.0

        left = self._ir[f'{corner}wearL'] or 1.0
        mid = self._ir[f'{corner}wearM'] or 1.0
        right = self._ir[f'{corner}wearR'] or 1.0

        # Min value = worst wear, convert to percentage worn
        min_wear = min(left, mid, right)
        return (1.0 - min_wear) * 100.0

    def _calculate_gaps(self) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate gaps to car ahead and behind in seconds.

        Returns:
            Tuple of (gap_ahead_sec, gap_behind_sec). None if no car ahead/behind.
        """
        if self._ir is None:
            return None, None

        try:
            player_idx = self._ir['PlayerCarIdx']
            positions = self._ir['CarIdxPosition']
            est_times = self._ir['CarIdxEstTime']

            if player_idx is None or positions is None or est_times is None:
                return None, None

            player_position = positions[player_idx]
            player_time = est_times[player_idx]

            if player_position <= 0 or player_time <= 0:
                return None, None

            gap_ahead = None
            gap_behind = None

            # Find car ahead (position - 1) and behind (position + 1)
            for idx, pos in enumerate(positions):
                if pos <= 0 or idx == player_idx:
                    continue

                car_time = est_times[idx]
                if car_time <= 0:
                    continue

                if pos == player_position - 1:
                    # Car ahead
                    gap_ahead = player_time - car_time
                elif pos == player_position + 1:
                    # Car behind
                    gap_behind = car_time - player_time

            return gap_ahead, gap_behind

        except (TypeError, IndexError):
            return None, None
