"""
Tests for car and track metadata module.

TDD: These tests are written BEFORE the implementation.
"""

import pytest


class TestCarsMetadata:
    """Tests for CARS dictionary."""

    def test_cars_dict_exists(self):
        """Test CARS dictionary is importable."""
        from src.metadata import CARS
        assert isinstance(CARS, dict)

    def test_cars_has_minimum_entries(self):
        """Test CARS has at least 5 cars per spec."""
        from src.metadata import CARS
        assert len(CARS) >= 5

    def test_car_has_required_fields(self):
        """Test each car entry has all required fields."""
        from src.metadata import CARS

        required_fields = {"name", "class", "traits"}

        for car_key, car_data in CARS.items():
            for field in required_fields:
                assert field in car_data, f"Car {car_key} missing field: {field}"

    def test_car_traits_is_list(self):
        """Test car traits is a list of strings."""
        from src.metadata import CARS

        for car_key, car_data in CARS.items():
            assert isinstance(car_data["traits"], list), f"Car {car_key} traits should be a list"
            for trait in car_data["traits"]:
                assert isinstance(trait, str), f"Car {car_key} trait should be string"

    def test_car_class_is_valid(self):
        """Test car class is one of expected values."""
        from src.metadata import CARS

        valid_classes = {
            # GT Classes
            "GT3", "GT4", "GTE",
            # Prototype Classes
            "GTP", "LMDh", "LMP2", "LMP3", "hypercar", "prototype",
            # Production / Touring
            "production", "touring", "TCR", "TCX", "supercars",
            # Cup / One-Make
            "cup",
            # Open Wheel
            "open_wheel", "F1", "IndyCar", "super_formula", "F3", "F4",
            "historic_f1",
            # Other
            "rallycross", "stock_car"
        }

        for car_key, car_data in CARS.items():
            assert car_data["class"] in valid_classes, f"Car {car_key} has invalid class: {car_data['class']}"

    def test_has_bmw_m4_gt3(self):
        """Test CARS includes BMW M4 GT3 (used in collected data)."""
        from src.metadata import CARS
        assert "bmw_m4_gt3" in CARS
        assert CARS["bmw_m4_gt3"]["class"] == "GT3"

    def test_has_mx5_cup(self):
        """Test CARS includes MX-5 Cup (used in collected data)."""
        from src.metadata import CARS
        assert "mx5_cup" in CARS
        assert CARS["mx5_cup"]["class"] == "production"


class TestTracksMetadata:
    """Tests for TRACKS dictionary."""

    def test_tracks_dict_exists(self):
        """Test TRACKS dictionary is importable."""
        from src.metadata import TRACKS
        assert isinstance(TRACKS, dict)

    def test_tracks_has_minimum_entries(self):
        """Test TRACKS has at least 5 tracks per spec."""
        from src.metadata import TRACKS
        assert len(TRACKS) >= 5

    def test_track_has_required_fields(self):
        """Test each track entry has all required fields."""
        from src.metadata import TRACKS

        required_fields = {"name", "type", "key_corners"}

        for track_key, track_data in TRACKS.items():
            for field in required_fields:
                assert field in track_data, f"Track {track_key} missing field: {field}"

    def test_track_key_corners_is_dict(self):
        """Test track key_corners is a dict mapping lap_pct to corner name."""
        from src.metadata import TRACKS

        for track_key, track_data in TRACKS.items():
            corners = track_data["key_corners"]
            assert isinstance(corners, dict), f"Track {track_key} key_corners should be dict"

            for lap_pct, corner_name in corners.items():
                assert isinstance(lap_pct, float), f"Track {track_key} corner key should be float"
                assert 0.0 <= lap_pct <= 1.0, f"Track {track_key} lap_pct should be 0-1"
                assert isinstance(corner_name, str), f"Track {track_key} corner name should be string"

    def test_track_type_is_valid(self):
        """Test track type is one of expected values."""
        from src.metadata import TRACKS

        valid_types = {
            # Road course types
            "high_speed", "mixed", "technical", "street",
            # Oval types
            "superspeedway", "intermediate", "short_track"
        }

        for track_key, track_data in TRACKS.items():
            assert track_data["type"] in valid_types, f"Track {track_key} has invalid type: {track_data['type']}"

    def test_has_monza(self):
        """Test TRACKS includes Monza."""
        from src.metadata import TRACKS
        assert "monza" in TRACKS
        assert TRACKS["monza"]["type"] == "high_speed"

    def test_has_spa(self):
        """Test TRACKS includes Spa."""
        from src.metadata import TRACKS
        assert "spa" in TRACKS


class TestCarNameMapping:
    """Tests for mapping iRacing car names to metadata keys."""

    def test_get_car_key_function_exists(self):
        """Test get_car_key function is importable."""
        from src.metadata import get_car_key
        assert callable(get_car_key)

    def test_maps_exact_name(self):
        """Test exact car name mapping."""
        from src.metadata import get_car_key
        # iRacing returns "BMW M4 GT3"
        assert get_car_key("BMW M4 GT3") == "bmw_m4_gt3"

    def test_maps_mx5_cup(self):
        """Test MX-5 Cup mapping (various possible iRacing names)."""
        from src.metadata import get_car_key
        key = get_car_key("Mazda MX-5 Cup")
        assert key == "mx5_cup"

    def test_returns_none_for_unknown_car(self):
        """Test unknown car returns None."""
        from src.metadata import get_car_key
        result = get_car_key("Unknown Fake Car 9000")
        assert result is None

    def test_case_insensitive_matching(self):
        """Test car name matching is case-insensitive."""
        from src.metadata import get_car_key
        # Should handle case variations
        key1 = get_car_key("bmw m4 gt3")
        key2 = get_car_key("BMW M4 GT3")
        assert key1 == key2 == "bmw_m4_gt3"


class TestTrackNameMapping:
    """Tests for mapping iRacing track names to metadata keys."""

    def test_get_track_key_function_exists(self):
        """Test get_track_key function is importable."""
        from src.metadata import get_track_key
        assert callable(get_track_key)

    def test_maps_monza(self):
        """Test Monza mapping."""
        from src.metadata import get_track_key
        # iRacing returns "Autodromo Nazionale Monza"
        key = get_track_key("Autodromo Nazionale Monza")
        assert key == "monza"

    def test_maps_spa(self):
        """Test Spa mapping."""
        from src.metadata import get_track_key
        key = get_track_key("Circuit de Spa-Francorchamps")
        assert key == "spa"

    def test_returns_none_for_unknown_track(self):
        """Test unknown track returns None."""
        from src.metadata import get_track_key
        result = get_track_key("Fake Circuit 3000")
        assert result is None

    def test_case_insensitive_matching(self):
        """Test track name matching is case-insensitive."""
        from src.metadata import get_track_key
        key1 = get_track_key("monza")
        key2 = get_track_key("MONZA")
        # Both should find monza (partial match)
        assert key1 == key2


class TestGetUpcomingCorner:
    """Tests for getting upcoming corner from lap percentage."""

    def test_get_upcoming_corner_function_exists(self):
        """Test get_upcoming_corner function is importable."""
        from src.metadata import get_upcoming_corner
        assert callable(get_upcoming_corner)

    def test_returns_corner_name_for_valid_track(self):
        """Test returns corner name string."""
        from src.metadata import get_upcoming_corner
        corner = get_upcoming_corner("monza", 0.15)
        assert isinstance(corner, str)
        assert len(corner) > 0

    def test_monza_rettifilo_at_015(self):
        """Test Monza lap_pct 0.15 returns Rettifilo."""
        from src.metadata import get_upcoming_corner
        corner = get_upcoming_corner("monza", 0.15)
        assert "Rettifilo" in corner

    def test_returns_none_for_unknown_track(self):
        """Test unknown track returns None."""
        from src.metadata import get_upcoming_corner
        result = get_upcoming_corner("fake_track", 0.5)
        assert result is None

    def test_finds_nearest_upcoming_corner(self):
        """Test finds the next corner ahead of current position."""
        from src.metadata import get_upcoming_corner
        # At lap_pct 0.10, should find the next corner (not one we passed)
        corner = get_upcoming_corner("monza", 0.10)
        assert corner is not None


class TestGetCarMetadata:
    """Tests for getting full car metadata."""

    def test_get_car_metadata_function_exists(self):
        """Test get_car_metadata function is importable."""
        from src.metadata import get_car_metadata
        assert callable(get_car_metadata)

    def test_returns_dict_for_known_car(self):
        """Test returns metadata dict for known car."""
        from src.metadata import get_car_metadata
        meta = get_car_metadata("BMW M4 GT3")
        assert isinstance(meta, dict)
        assert "name" in meta
        assert "class" in meta
        assert "traits" in meta

    def test_returns_none_for_unknown_car(self):
        """Test returns None for unknown car."""
        from src.metadata import get_car_metadata
        result = get_car_metadata("Unknown Car")
        assert result is None


class TestGetTrackMetadata:
    """Tests for getting full track metadata."""

    def test_get_track_metadata_function_exists(self):
        """Test get_track_metadata function is importable."""
        from src.metadata import get_track_metadata
        assert callable(get_track_metadata)

    def test_returns_dict_for_known_track(self):
        """Test returns metadata dict for known track."""
        from src.metadata import get_track_metadata
        meta = get_track_metadata("Autodromo Nazionale Monza")
        assert isinstance(meta, dict)
        assert "name" in meta
        assert "type" in meta
        assert "key_corners" in meta

    def test_returns_none_for_unknown_track(self):
        """Test returns None for unknown track."""
        from src.metadata import get_track_metadata
        result = get_track_metadata("Unknown Track")
        assert result is None
