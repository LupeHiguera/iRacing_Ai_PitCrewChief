"""
Tests for TelemetryReader class.
"""

import pytest
from unittest.mock import Mock, patch

from src.telemetry import TelemetryReader, TelemetrySnapshot


class TestTelemetrySnapshot:
    """Tests for the TelemetrySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test that we can create a valid snapshot with all fields."""
        snapshot = TelemetrySnapshot(
            lap=5,
            lap_pct=0.45,
            position=3,
            fuel_level=15.5,
            fuel_level_pct=0.65,
            fuel_use_per_hour=2.3,
            tire_wear_lf=15.0,
            tire_wear_rf=18.0,
            tire_wear_lr=12.0,
            tire_wear_rr=14.0,
            session_time_remain=1200.0,
            session_laps_remain=20,
            last_lap_time=85.432,
            best_lap_time=84.123,
            on_pit_road=False,
            is_on_track=True,
            # Expanded fields
            tire_temp_lf_l=85.0, tire_temp_lf_m=90.0, tire_temp_lf_r=88.0,
            tire_temp_rf_l=87.0, tire_temp_rf_m=92.0, tire_temp_rf_r=89.0,
            tire_temp_lr_l=82.0, tire_temp_lr_m=86.0, tire_temp_lr_r=84.0,
            tire_temp_rr_l=83.0, tire_temp_rr_m=88.0, tire_temp_rr_r=85.0,
            tire_pressure_lf=165.0, tire_pressure_rf=168.0,
            tire_pressure_lr=158.0, tire_pressure_rr=160.0,
            gap_ahead_sec=1.2, gap_behind_sec=0.8,
            track_temp_c=35.5, air_temp_c=22.0,
            brake_press_lf=450.0, brake_press_rf=460.0,
            brake_press_lr=380.0, brake_press_rr=390.0,
        )

        assert snapshot.lap == 5
        assert snapshot.lap_pct == 0.45
        assert snapshot.position == 3
        assert snapshot.fuel_level == 15.5
        assert snapshot.is_on_track is True
        assert snapshot.on_pit_road is False

    def test_snapshot_tire_wear_fields(self):
        """Test tire wear fields are correctly stored."""
        snapshot = TelemetrySnapshot(
            lap=1,
            lap_pct=0.0,
            position=1,
            fuel_level=20.0,
            fuel_level_pct=1.0,
            fuel_use_per_hour=2.0,
            tire_wear_lf=25.0,
            tire_wear_rf=30.0,
            tire_wear_lr=20.0,
            tire_wear_rr=22.0,
            session_time_remain=3600.0,
            session_laps_remain=40,
            last_lap_time=0.0,
            best_lap_time=0.0,
            on_pit_road=False,
            is_on_track=True,
            # Expanded fields
            tire_temp_lf_l=85.0, tire_temp_lf_m=90.0, tire_temp_lf_r=88.0,
            tire_temp_rf_l=87.0, tire_temp_rf_m=92.0, tire_temp_rf_r=89.0,
            tire_temp_lr_l=82.0, tire_temp_lr_m=86.0, tire_temp_lr_r=84.0,
            tire_temp_rr_l=83.0, tire_temp_rr_m=88.0, tire_temp_rr_r=85.0,
            tire_pressure_lf=165.0, tire_pressure_rf=168.0,
            tire_pressure_lr=158.0, tire_pressure_rr=160.0,
            gap_ahead_sec=None, gap_behind_sec=None,
            track_temp_c=35.5, air_temp_c=22.0,
            brake_press_lf=450.0, brake_press_rf=460.0,
            brake_press_lr=380.0, brake_press_rr=390.0,
        )

        assert snapshot.tire_wear_lf == 25.0
        assert snapshot.tire_wear_rf == 30.0
        assert snapshot.tire_wear_lr == 20.0
        assert snapshot.tire_wear_rr == 22.0


class TestTelemetryReader:
    """Tests for the TelemetryReader class."""

    def test_reader_initialization(self):
        """Test that reader can be instantiated."""
        reader = TelemetryReader()
        assert reader is not None

    def test_initial_state_not_connected(self):
        """Test that reader starts in disconnected state."""
        reader = TelemetryReader()
        assert reader.is_connected() is False

    def test_get_snapshot_when_disconnected_returns_none(self):
        """Test that get_snapshot returns None when not connected."""
        reader = TelemetryReader()
        snapshot = reader.get_snapshot()
        assert snapshot is None

    @patch('src.telemetry.irsdk')
    def test_connect_to_iracing(self, mock_irsdk):
        """Test connecting to iRacing."""
        mock_ir = Mock()
        mock_ir.startup.return_value = True
        mock_irsdk.IRSDK.return_value = mock_ir

        reader = TelemetryReader()
        result = reader.connect()

        assert result is True
        assert reader.is_connected() is True

    @patch('src.telemetry.irsdk')
    def test_connect_fails_when_iracing_not_running(self, mock_irsdk):
        """Test that connect returns False when iRacing isn't running."""
        mock_ir = Mock()
        mock_ir.startup.return_value = False
        mock_irsdk.IRSDK.return_value = mock_ir

        reader = TelemetryReader()
        result = reader.connect()

        assert result is False
        assert reader.is_connected() is False

    @patch('src.telemetry.irsdk')
    def test_disconnect(self, mock_irsdk):
        """Test disconnecting from iRacing."""
        mock_ir = Mock()
        mock_ir.startup.return_value = True
        mock_irsdk.IRSDK.return_value = mock_ir

        reader = TelemetryReader()
        reader.connect()
        reader.disconnect()

        assert reader.is_connected() is False
        mock_ir.shutdown.assert_called_once()

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_returns_valid_data(self, mock_irsdk):
        """Test that get_snapshot returns valid telemetry data."""
        mock_ir = Mock()
        mock_ir.startup.return_value = True
        mock_ir.__getitem__ = Mock(side_effect=lambda key: {
            'Lap': 5,
            'LapDistPct': 0.45,
            'PlayerCarPosition': 3,
            'FuelLevel': 15.5,
            'FuelLevelPct': 0.65,
            'FuelUsePerHour': 2.3,
            'LFwearL': 0.85, 'LFwearM': 0.82, 'LFwearR': 0.84,
            'RFwearL': 0.80, 'RFwearM': 0.78, 'RFwearR': 0.79,
            'LRwearL': 0.88, 'LRwearM': 0.86, 'LRwearR': 0.87,
            'RRwearL': 0.83, 'RRwearM': 0.81, 'RRwearR': 0.82,
            'SessionTimeRemain': 1200.0,
            'SessionLapsRemain': 20,
            'LapLastLapTime': 85.432,
            'LapBestLapTime': 84.123,
            'OnPitRoad': False,
            'IsOnTrack': True,
        }.get(key))
        mock_irsdk.IRSDK.return_value = mock_ir

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot is not None
        assert isinstance(snapshot, TelemetrySnapshot)
        assert snapshot.lap == 5
        assert snapshot.position == 3
        assert snapshot.fuel_level == 15.5

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_handles_missing_data_gracefully(self, mock_irsdk):
        """Test that get_snapshot handles missing/None data."""
        mock_ir = Mock()
        mock_ir.startup.return_value = True
        mock_ir.__getitem__ = Mock(return_value=None)
        mock_irsdk.IRSDK.return_value = mock_ir

        reader = TelemetryReader()
        reader.connect()

        # Should not raise, should return snapshot with defaults or None
        snapshot = reader.get_snapshot()
        # Either returns None or a snapshot with default values
        assert snapshot is None or isinstance(snapshot, TelemetrySnapshot)

    @patch('src.telemetry.irsdk')
    def test_tire_wear_calculated_from_min_values(self, mock_irsdk):
        """Test that tire wear uses minimum of L/M/R values per corner."""
        mock_ir = Mock()
        mock_ir.startup.return_value = True
        # Tire wear in iRacing: 1.0 = new, 0.0 = worn
        # We want to track worst wear, so we use min value
        mock_ir.__getitem__ = Mock(side_effect=lambda key: {
            'Lap': 1,
            'LapDistPct': 0.0,
            'PlayerCarPosition': 1,
            'FuelLevel': 20.0,
            'FuelLevelPct': 1.0,
            'FuelUsePerHour': 2.0,
            'LFwearL': 0.70, 'LFwearM': 0.75, 'LFwearR': 0.72,  # Min: 0.70 = 30% worn
            'RFwearL': 0.65, 'RFwearM': 0.68, 'RFwearR': 0.66,  # Min: 0.65 = 35% worn
            'LRwearL': 0.80, 'LRwearM': 0.82, 'LRwearR': 0.81,  # Min: 0.80 = 20% worn
            'RRwearL': 0.75, 'RRwearM': 0.78, 'RRwearR': 0.76,  # Min: 0.75 = 25% worn
            'SessionTimeRemain': 3600.0,
            'SessionLapsRemain': 40,
            'LapLastLapTime': 0.0,
            'LapBestLapTime': 0.0,
            'OnPitRoad': False,
            'IsOnTrack': True,
        }.get(key))
        mock_irsdk.IRSDK.return_value = mock_ir

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        # Wear percentage = (1 - min_value) * 100
        assert snapshot.tire_wear_lf == pytest.approx(30.0, abs=0.1)
        assert snapshot.tire_wear_rf == pytest.approx(35.0, abs=0.1)
        assert snapshot.tire_wear_lr == pytest.approx(20.0, abs=0.1)
        assert snapshot.tire_wear_rr == pytest.approx(25.0, abs=0.1)


class TestExpandedTelemetry:
    """Tests for expanded telemetry data collection."""

    def test_snapshot_has_tire_temp_fields(self):
        """Test that snapshot includes tire temperature fields (L/M/R per corner)."""
        snapshot = self._create_full_snapshot()

        # Each corner has left/middle/right temps across the tire surface
        assert hasattr(snapshot, 'tire_temp_lf_l')
        assert hasattr(snapshot, 'tire_temp_lf_m')
        assert hasattr(snapshot, 'tire_temp_lf_r')
        assert hasattr(snapshot, 'tire_temp_rf_l')
        assert hasattr(snapshot, 'tire_temp_rf_m')
        assert hasattr(snapshot, 'tire_temp_rf_r')
        assert hasattr(snapshot, 'tire_temp_lr_l')
        assert hasattr(snapshot, 'tire_temp_lr_m')
        assert hasattr(snapshot, 'tire_temp_lr_r')
        assert hasattr(snapshot, 'tire_temp_rr_l')
        assert hasattr(snapshot, 'tire_temp_rr_m')
        assert hasattr(snapshot, 'tire_temp_rr_r')

    def test_snapshot_has_tire_pressure_fields(self):
        """Test that snapshot includes tire pressure for each corner."""
        snapshot = self._create_full_snapshot()

        assert hasattr(snapshot, 'tire_pressure_lf')
        assert hasattr(snapshot, 'tire_pressure_rf')
        assert hasattr(snapshot, 'tire_pressure_lr')
        assert hasattr(snapshot, 'tire_pressure_rr')

    def test_snapshot_has_gap_fields(self):
        """Test that snapshot includes gap to car ahead and behind."""
        snapshot = self._create_full_snapshot()

        assert hasattr(snapshot, 'gap_ahead_sec')
        assert hasattr(snapshot, 'gap_behind_sec')

    def test_snapshot_has_track_condition_fields(self):
        """Test that snapshot includes track and weather conditions."""
        snapshot = self._create_full_snapshot()

        assert hasattr(snapshot, 'track_temp_c')
        assert hasattr(snapshot, 'air_temp_c')

    def test_snapshot_has_brake_pressure_fields(self):
        """Test that snapshot includes brake line pressure for each corner."""
        snapshot = self._create_full_snapshot()

        assert hasattr(snapshot, 'brake_press_lf')
        assert hasattr(snapshot, 'brake_press_rf')
        assert hasattr(snapshot, 'brake_press_lr')
        assert hasattr(snapshot, 'brake_press_rr')

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_tire_temps(self, mock_irsdk):
        """Test that tire temps are read from iRacing telemetry."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'LFtempCL': 85.0, 'LFtempCM': 90.0, 'LFtempCR': 88.0,
            'RFtempCL': 87.0, 'RFtempCM': 92.0, 'RFtempCR': 89.0,
            'LRtempCL': 82.0, 'LRtempCM': 86.0, 'LRtempCR': 84.0,
            'RRtempCL': 83.0, 'RRtempCM': 88.0, 'RRtempCR': 85.0,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.tire_temp_lf_l == pytest.approx(85.0, abs=0.1)
        assert snapshot.tire_temp_lf_m == pytest.approx(90.0, abs=0.1)
        assert snapshot.tire_temp_lf_r == pytest.approx(88.0, abs=0.1)

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_tire_pressures(self, mock_irsdk):
        """Test that tire pressures are read from iRacing telemetry."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'LFpressure': 165.0,
            'RFpressure': 168.0,
            'LRpressure': 158.0,
            'RRpressure': 160.0,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.tire_pressure_lf == pytest.approx(165.0, abs=0.1)
        assert snapshot.tire_pressure_rf == pytest.approx(168.0, abs=0.1)
        assert snapshot.tire_pressure_lr == pytest.approx(158.0, abs=0.1)
        assert snapshot.tire_pressure_rr == pytest.approx(160.0, abs=0.1)

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_track_conditions(self, mock_irsdk):
        """Test that track and air temps are read from iRacing telemetry."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'TrackTempCrew': 35.5,
            'AirTemp': 22.0,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.track_temp_c == pytest.approx(35.5, abs=0.1)
        assert snapshot.air_temp_c == pytest.approx(22.0, abs=0.1)

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_brake_pressures(self, mock_irsdk):
        """Test that brake line pressures are read from iRacing telemetry."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'LFbrakeLinePress': 4500.0,
            'RFbrakeLinePress': 4600.0,
            'LRbrakeLinePress': 3800.0,
            'RRbrakeLinePress': 3900.0,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.brake_press_lf == pytest.approx(4500.0, abs=0.1)
        assert snapshot.brake_press_rf == pytest.approx(4600.0, abs=0.1)
        assert snapshot.brake_press_lr == pytest.approx(3800.0, abs=0.1)
        assert snapshot.brake_press_rr == pytest.approx(3900.0, abs=0.1)

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_calculates_gaps(self, mock_irsdk):
        """Test that gaps to car ahead and behind are calculated."""
        # Uses CarIdxLapDistPct (track position 0-1) and best lap time
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'CarIdxLapDistPct': [0, 0.52, 0.50, 0.48, 0.55],  # Track positions per car
            'PlayerCarIdx': 2,  # We are car index 2 at 50%
            'CarIdxPosition': [0, 2, 3, 4, 1],  # Position per car (we are P3)
            'LapBestLapTime': 60.0,  # 60s lap time for easy math
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        # Car ahead (P2) is index 1 at 52%, car behind (P4) is index 3 at 48%
        # gap_ahead = (0.52 - 0.50) * 60 = 1.2s
        # gap_behind = (0.50 - 0.48) * 60 = 1.2s
        assert snapshot.gap_ahead_sec == pytest.approx(1.2, abs=0.1)
        assert snapshot.gap_behind_sec == pytest.approx(1.2, abs=0.1)

    def _create_full_snapshot(self):
        """Helper to create a snapshot with all expanded fields."""
        return TelemetrySnapshot(
            # Original fields
            lap=5,
            lap_pct=0.45,
            position=3,
            fuel_level=15.5,
            fuel_level_pct=0.65,
            fuel_use_per_hour=2.3,
            tire_wear_lf=15.0,
            tire_wear_rf=18.0,
            tire_wear_lr=12.0,
            tire_wear_rr=14.0,
            session_time_remain=1200.0,
            session_laps_remain=20,
            last_lap_time=85.432,
            best_lap_time=84.123,
            on_pit_road=False,
            is_on_track=True,
            # Expanded: Tire temps (Celsius)
            tire_temp_lf_l=85.0, tire_temp_lf_m=90.0, tire_temp_lf_r=88.0,
            tire_temp_rf_l=87.0, tire_temp_rf_m=92.0, tire_temp_rf_r=89.0,
            tire_temp_lr_l=82.0, tire_temp_lr_m=86.0, tire_temp_lr_r=84.0,
            tire_temp_rr_l=83.0, tire_temp_rr_m=88.0, tire_temp_rr_r=85.0,
            # Expanded: Tire pressures (kPa)
            tire_pressure_lf=165.0,
            tire_pressure_rf=168.0,
            tire_pressure_lr=158.0,
            tire_pressure_rr=160.0,
            # Expanded: Gaps (seconds)
            gap_ahead_sec=1.2,
            gap_behind_sec=0.8,
            # Expanded: Track conditions
            track_temp_c=35.5,
            air_temp_c=22.0,
            # Expanded: Brake line pressure (kPa)
            brake_press_lf=4500.0,
            brake_press_rf=4600.0,
            brake_press_lr=3800.0,
            brake_press_rr=3900.0,
        )

    def _create_mock_iracing(self, mock_irsdk, extra_values=None):
        """Helper to create a mock iRacing SDK with standard + extra values."""
        base_values = {
            'Lap': 5,
            'LapDistPct': 0.45,
            'PlayerCarPosition': 3,
            'FuelLevel': 15.5,
            'FuelLevelPct': 0.65,
            'FuelUsePerHour': 2.3,
            'LFwearL': 0.85, 'LFwearM': 0.82, 'LFwearR': 0.84,
            'RFwearL': 0.80, 'RFwearM': 0.78, 'RFwearR': 0.79,
            'LRwearL': 0.88, 'LRwearM': 0.86, 'LRwearR': 0.87,
            'RRwearL': 0.83, 'RRwearM': 0.81, 'RRwearR': 0.82,
            'SessionTimeRemain': 1200.0,
            'SessionLapsRemain': 20,
            'LapLastLapTime': 85.432,
            'LapBestLapTime': 84.123,
            'OnPitRoad': False,
            'IsOnTrack': True,
            # Default expanded values
            'LFtempCL': 85.0, 'LFtempCM': 90.0, 'LFtempCR': 88.0,
            'RFtempCL': 87.0, 'RFtempCM': 92.0, 'RFtempCR': 89.0,
            'LRtempCL': 82.0, 'LRtempCM': 86.0, 'LRtempCR': 84.0,
            'RRtempCL': 83.0, 'RRtempCM': 88.0, 'RRtempCR': 85.0,
            'LFpressure': 165.0, 'RFpressure': 168.0,
            'LRpressure': 158.0, 'RRpressure': 160.0,
            'TrackTempCrew': 35.5,
            'AirTemp': 22.0,
            'LFbrakeLinePress': 4500.0, 'RFbrakeLinePress': 4600.0,
            'LRbrakeLinePress': 3800.0, 'RRbrakeLinePress': 3900.0,
            'CarIdxLapDistPct': [0, 0.52, 0.50, 0.48, 0.55],
            'PlayerCarIdx': 2,
            'CarIdxPosition': [0, 2, 3, 4, 1],
        }

        if extra_values:
            base_values.update(extra_values)

        mock_ir = Mock()
        mock_ir.startup.return_value = True
        mock_ir.__getitem__ = Mock(side_effect=lambda key: base_values.get(key))
        mock_irsdk.IRSDK.return_value = mock_ir

        return mock_ir


class TestSessionFlagsAndIncidents:
    """Tests for session flags, incidents, and lap delta telemetry."""

    def test_snapshot_has_session_flags_field(self):
        """Test that snapshot includes session flags."""
        snapshot = self._create_snapshot_with_new_fields()
        assert hasattr(snapshot, 'session_flags')

    def test_snapshot_has_incident_count_field(self):
        """Test that snapshot includes driver incident count."""
        snapshot = self._create_snapshot_with_new_fields()
        assert hasattr(snapshot, 'incident_count')

    def test_snapshot_has_lap_delta_to_best_field(self):
        """Test that snapshot includes lap delta to best."""
        snapshot = self._create_snapshot_with_new_fields()
        assert hasattr(snapshot, 'lap_delta_to_best')

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_session_flags(self, mock_irsdk):
        """Test that session flags are read from iRacing telemetry."""
        # SessionFlags is a bitmask - 0x8 = yellow, 0x4 = green, etc.
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'SessionFlags': 0x8,  # Yellow flag
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.session_flags == 0x8

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_incident_count(self, mock_irsdk):
        """Test that incident count is read from iRacing telemetry."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'PlayerCarDriverIncidentCount': 8,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.incident_count == 8

    @patch('src.telemetry.irsdk')
    def test_get_snapshot_reads_lap_delta_to_best(self, mock_irsdk):
        """Test that lap delta to best is read from iRacing telemetry."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'LapDeltaToBestLap': -0.352,  # 0.352 seconds faster than best
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.lap_delta_to_best == pytest.approx(-0.352, abs=0.001)

    @patch('src.telemetry.irsdk')
    def test_session_flags_defaults_to_zero_when_missing(self, mock_irsdk):
        """Test that session flags default to 0 when not available."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'SessionFlags': None,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.session_flags == 0

    @patch('src.telemetry.irsdk')
    def test_incident_count_defaults_to_zero_when_missing(self, mock_irsdk):
        """Test that incident count defaults to 0 when not available."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'PlayerCarDriverIncidentCount': None,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.incident_count == 0

    @patch('src.telemetry.irsdk')
    def test_lap_delta_defaults_to_zero_when_missing(self, mock_irsdk):
        """Test that lap delta defaults to 0.0 when not available."""
        mock_ir = self._create_mock_iracing(mock_irsdk, {
            'LapDeltaToBestLap': None,
        })

        reader = TelemetryReader()
        reader.connect()
        snapshot = reader.get_snapshot()

        assert snapshot.lap_delta_to_best == 0.0

    def _create_snapshot_with_new_fields(self):
        """Helper to create a snapshot with all fields including new ones."""
        return TelemetrySnapshot(
            lap=5,
            lap_pct=0.45,
            position=3,
            fuel_level=15.5,
            fuel_level_pct=0.65,
            fuel_use_per_hour=2.3,
            tire_wear_lf=15.0,
            tire_wear_rf=18.0,
            tire_wear_lr=12.0,
            tire_wear_rr=14.0,
            session_time_remain=1200.0,
            session_laps_remain=20,
            last_lap_time=85.432,
            best_lap_time=84.123,
            on_pit_road=False,
            is_on_track=True,
            # Tire temps
            tire_temp_lf_l=85.0, tire_temp_lf_m=90.0, tire_temp_lf_r=88.0,
            tire_temp_rf_l=87.0, tire_temp_rf_m=92.0, tire_temp_rf_r=89.0,
            tire_temp_lr_l=82.0, tire_temp_lr_m=86.0, tire_temp_lr_r=84.0,
            tire_temp_rr_l=83.0, tire_temp_rr_m=88.0, tire_temp_rr_r=85.0,
            # Tire pressures
            tire_pressure_lf=165.0, tire_pressure_rf=168.0,
            tire_pressure_lr=158.0, tire_pressure_rr=160.0,
            # Gaps
            gap_ahead_sec=1.2, gap_behind_sec=0.8,
            # Track conditions
            track_temp_c=35.5, air_temp_c=22.0,
            # Brake line pressure (kPa)
            brake_press_lf=4500.0, brake_press_rf=4600.0,
            brake_press_lr=3800.0, brake_press_rr=3900.0,
            # New fields
            session_flags=0x4,  # Green flag
            incident_count=4,
            lap_delta_to_best=-0.5,
        )

    def _create_mock_iracing(self, mock_irsdk, extra_values=None):
        """Helper to create a mock iRacing SDK with standard + extra values."""
        base_values = {
            'Lap': 5,
            'LapDistPct': 0.45,
            'PlayerCarPosition': 3,
            'FuelLevel': 15.5,
            'FuelLevelPct': 0.65,
            'FuelUsePerHour': 2.3,
            'LFwearL': 0.85, 'LFwearM': 0.82, 'LFwearR': 0.84,
            'RFwearL': 0.80, 'RFwearM': 0.78, 'RFwearR': 0.79,
            'LRwearL': 0.88, 'LRwearM': 0.86, 'LRwearR': 0.87,
            'RRwearL': 0.83, 'RRwearM': 0.81, 'RRwearR': 0.82,
            'SessionTimeRemain': 1200.0,
            'SessionLapsRemain': 20,
            'LapLastLapTime': 85.432,
            'LapBestLapTime': 84.123,
            'OnPitRoad': False,
            'IsOnTrack': True,
            # Expanded values
            'LFtempCL': 85.0, 'LFtempCM': 90.0, 'LFtempCR': 88.0,
            'RFtempCL': 87.0, 'RFtempCM': 92.0, 'RFtempCR': 89.0,
            'LRtempCL': 82.0, 'LRtempCM': 86.0, 'LRtempCR': 84.0,
            'RRtempCL': 83.0, 'RRtempCM': 88.0, 'RRtempCR': 85.0,
            'LFpressure': 165.0, 'RFpressure': 168.0,
            'LRpressure': 158.0, 'RRpressure': 160.0,
            'TrackTempCrew': 35.5,
            'AirTemp': 22.0,
            'LFbrakeLinePress': 4500.0, 'RFbrakeLinePress': 4600.0,
            'LRbrakeLinePress': 3800.0, 'RRbrakeLinePress': 3900.0,
            'CarIdxLapDistPct': [0, 0.52, 0.50, 0.48, 0.55],
            'PlayerCarIdx': 2,
            'CarIdxPosition': [0, 2, 3, 4, 1],
            # New fields
            'SessionFlags': 0x4,  # Green flag
            'PlayerCarDriverIncidentCount': 4,
            'LapDeltaToBestLap': -0.5,
        }

        if extra_values:
            base_values.update(extra_values)

        mock_ir = Mock()
        mock_ir.startup.return_value = True
        mock_ir.__getitem__ = Mock(side_effect=lambda key: base_values.get(key))
        mock_irsdk.IRSDK.return_value = mock_ir

        return mock_ir