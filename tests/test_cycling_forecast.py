"""Tests for cycling weather and daylight summaries."""

from datetime import datetime

import pytest

from intervals_icu_mcp.cycling_forecast import (
    build_cycling_forecast_context,
    parse_aware_datetime,
)


def _payload() -> dict[str, object]:
    return {
        "timezone": "Europe/Madrid",
        "hourly": {
            "time": [
                "2026-08-13T07:00",
                "2026-08-13T08:00",
                "2026-08-13T09:00",
                "2026-08-13T10:00",
            ],
            "temperature_2m": [20.0, 21.0, 23.0, 25.0],
            "apparent_temperature": [19.0, 21.5, 24.0, 26.0],
            "precipitation_probability": [0, 10, 70, 20],
            "precipitation": [0.0, 0.0, 1.2, 0.1],
            "weather_code": [0, 1, 95, 61],
            "wind_speed_10m": [5.0, 8.0, 12.0, 10.0],
            "wind_direction_10m": [90, 100, 110, 120],
            "wind_gusts_10m": [8.0, 12.0, 22.0, 15.0],
        },
        "daily": {
            "time": ["2026-08-13"],
            "sunrise": ["2026-08-13T07:02"],
            "sunset": ["2026-08-13T20:48"],
        },
    }


def test_parse_aware_datetime_accepts_offset_and_z() -> None:
    assert parse_aware_datetime("2026-08-13T08:00:00+02:00").utcoffset()
    assert parse_aware_datetime("2026-08-13T06:00:00Z").utcoffset() is not None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "must not be empty"),
        ("not-a-date", "valid ISO 8601"),
        ("2026-08-13T08:00:00", "must include a UTC offset"),
    ],
)
def test_parse_aware_datetime_rejects_ambiguous_values(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_aware_datetime(value)


def test_build_context_summarizes_interval_and_warnings() -> None:
    context = build_cycling_forecast_context(
        _payload(),
        departure_time=datetime.fromisoformat("2026-08-13T08:30:00+02:00"),
        route_duration_s=5400.0,
    )

    assert context.timezone == "Europe/Madrid"
    assert context.departure_time == "2026-08-13T08:30:00+02:00"
    assert context.estimated_finish_time == "2026-08-13T10:00:00+02:00"
    assert context.sampled_forecast_hours == (
        "2026-08-13T08:00",
        "2026-08-13T09:00",
        "2026-08-13T10:00",
    )
    assert context.temperature_min_c == 21.0
    assert context.temperature_max_c == 25.0
    assert context.apparent_temperature_min_c == 21.5
    assert context.apparent_temperature_max_c == 26.0
    assert context.precipitation_probability_max_percentage == 70.0
    assert context.precipitation_sum_mm == pytest.approx(1.3)
    assert context.wind_speed_max_kmh == 12.0
    assert context.wind_gusts_max_kmh == 22.0
    assert context.wind_directions_degrees == (100, 110, 120)
    assert context.weather_codes == (1, 61, 95)
    assert context.sunrise == "2026-08-13T07:02"
    assert context.sunset == "2026-08-13T20:48"
    assert context.entirely_in_daylight is True
    assert context.warnings == (
        "precipitation_forecast",
        "thunderstorm_forecast",
    )


def test_build_context_warns_about_darkness_without_thresholds() -> None:
    payload = _payload()
    context = build_cycling_forecast_context(
        payload,
        departure_time=datetime.fromisoformat("2026-08-13T06:30:00+02:00"),
        route_duration_s=1800.0,
    )
    assert context.departure_after_sunrise is False
    assert context.estimated_finish_before_sunset is True
    assert context.entirely_in_daylight is False
    assert "departure_before_sunrise" in context.warnings


def test_build_context_converts_departure_to_origin_timezone() -> None:
    context = build_cycling_forecast_context(
        _payload(),
        departure_time=datetime.fromisoformat("2026-08-13T06:30:00+00:00"),
        route_duration_s=1800.0,
    )
    assert context.departure_time == "2026-08-13T08:30:00+02:00"
    assert context.estimated_finish_time == "2026-08-13T09:00:00+02:00"


def test_build_context_rejects_missing_interval() -> None:
    with pytest.raises(
        ValueError,
        match="does not cover the estimated route interval",
    ):
        build_cycling_forecast_context(
            _payload(),
            departure_time=datetime.fromisoformat(
                "2026-08-14T08:30:00+02:00"
            ),
            route_duration_s=1800.0,
        )


def test_build_context_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="must be greater than zero"):
        build_cycling_forecast_context(
            _payload(),
            departure_time=datetime.fromisoformat(
                "2026-08-13T08:30:00+02:00"
            ),
            route_duration_s=0.0,
        )
