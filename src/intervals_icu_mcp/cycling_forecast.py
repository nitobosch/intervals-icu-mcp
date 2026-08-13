"""Origin-level weather and daylight context for cycling routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .open_meteo_client import OPEN_METEO_ATTRIBUTION

_THUNDERSTORM_WEATHER_CODES = frozenset({95, 96, 99})


@dataclass(frozen=True)
class CyclingForecastContext:
    """Forecast facts covering one estimated route interval."""

    source: dict[str, str]
    timezone: str
    departure_time: str
    estimated_finish_time: str
    sampled_forecast_hours: tuple[str, ...]
    temperature_min_c: float | None
    temperature_max_c: float | None
    apparent_temperature_min_c: float | None
    apparent_temperature_max_c: float | None
    precipitation_probability_max_percentage: float | None
    precipitation_sum_mm: float | None
    wind_speed_max_kmh: float | None
    wind_gusts_max_kmh: float | None
    wind_directions_degrees: tuple[int, ...]
    weather_codes: tuple[int, ...]
    sunrise: str | None
    sunset: str | None
    departure_after_sunrise: bool | None
    estimated_finish_before_sunset: bool | None
    entirely_in_daylight: bool | None
    warnings: tuple[str, ...]


def parse_aware_datetime(value: str) -> datetime:
    """Parse ISO 8601 while requiring an explicit UTC offset."""

    text = value.strip()
    if not text:
        raise ValueError("departure_time must not be empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "departure_time must be a valid ISO 8601 datetime"
        ) from exc
    if parsed.utcoffset() is None:
        raise ValueError(
            "departure_time must include a UTC offset, for example "
            "2026-08-13T08:00:00+02:00"
        )
    return parsed


def _dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Open-Meteo response field '{field}' must be an object")
    return cast(dict[str, Any], value)


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Open-Meteo response field '{field}' must be an array")
    return cast(list[Any], value)


def _numeric_values(
    hourly: dict[str, Any],
    field: str,
    indexes: tuple[int, ...],
) -> tuple[float, ...]:
    values = _sequence(hourly.get(field), f"hourly.{field}")
    selected: list[float] = []
    for index in indexes:
        if index >= len(values) or values[index] is None:
            continue
        value = values[index]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Open-Meteo response field 'hourly.{field}' "
                "must contain numbers or null"
            )
        selected.append(float(value))
    return tuple(selected)


def _optional_min(values: tuple[float, ...]) -> float | None:
    return min(values) if values else None


def _optional_max(values: tuple[float, ...]) -> float | None:
    return max(values) if values else None


def _daily_value(
    daily: dict[str, Any],
    field: str,
    wanted_date: str,
) -> str | None:
    dates = _sequence(daily.get("time"), "daily.time")
    values = _sequence(daily.get(field), f"daily.{field}")
    try:
        index = dates.index(wanted_date)
    except ValueError:
        return None
    if index >= len(values) or values[index] is None:
        return None
    value = values[index]
    if not isinstance(value, str):
        raise ValueError(
            f"Open-Meteo response field 'daily.{field}' must contain strings"
        )
    return value


def build_cycling_forecast_context(
    payload: dict[str, Any],
    *,
    departure_time: datetime,
    route_duration_s: float,
) -> CyclingForecastContext:
    """Summarize origin weather and daylight for an estimated ride interval."""

    if departure_time.utcoffset() is None:
        raise ValueError("departure_time must include a UTC offset")
    if route_duration_s <= 0:
        raise ValueError("route_duration_s must be greater than zero")

    timezone_name = payload.get("timezone")
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise ValueError("Open-Meteo response must include a timezone")
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Open-Meteo returned unknown timezone '{timezone_name}'"
        ) from exc

    local_departure = departure_time.astimezone(timezone)
    local_finish = local_departure + timedelta(seconds=route_duration_s)
    sample_start = local_departure.replace(minute=0, second=0, microsecond=0)
    sample_end = local_finish.replace(minute=0, second=0, microsecond=0)

    hourly = _dict(payload.get("hourly"), "hourly")
    hourly_times = _sequence(hourly.get("time"), "hourly.time")
    parsed_hourly_times: list[datetime] = []
    for value in hourly_times:
        if not isinstance(value, str):
            raise ValueError(
                "Open-Meteo response field 'hourly.time' must contain strings"
            )
        try:
            parsed_hourly_times.append(datetime.fromisoformat(value))
        except ValueError as exc:
            raise ValueError(
                "Open-Meteo response contains an invalid hourly timestamp"
            ) from exc

    indexes = tuple(
        index
        for index, value in enumerate(parsed_hourly_times)
        if sample_start.replace(tzinfo=None)
        <= value
        <= sample_end.replace(tzinfo=None)
    )
    if not indexes:
        raise ValueError(
            "Open-Meteo response does not cover the estimated route interval"
        )

    temperatures = _numeric_values(hourly, "temperature_2m", indexes)
    apparent_temperatures = _numeric_values(
        hourly, "apparent_temperature", indexes
    )
    precipitation_probabilities = _numeric_values(
        hourly, "precipitation_probability", indexes
    )
    precipitation = _numeric_values(hourly, "precipitation", indexes)
    wind_speeds = _numeric_values(hourly, "wind_speed_10m", indexes)
    wind_gusts = _numeric_values(hourly, "wind_gusts_10m", indexes)
    wind_directions = _numeric_values(hourly, "wind_direction_10m", indexes)
    weather_code_values = _numeric_values(hourly, "weather_code", indexes)
    weather_codes = tuple(sorted({int(value) for value in weather_code_values}))

    daily = _dict(payload.get("daily"), "daily")
    departure_date = local_departure.date().isoformat()
    finish_date = local_finish.date().isoformat()
    sunrise_text = _daily_value(daily, "sunrise", departure_date)
    sunset_text = _daily_value(daily, "sunset", finish_date)
    sunrise = datetime.fromisoformat(sunrise_text) if sunrise_text else None
    sunset = datetime.fromisoformat(sunset_text) if sunset_text else None
    naive_departure = local_departure.replace(tzinfo=None)
    naive_finish = local_finish.replace(tzinfo=None)
    departure_after_sunrise = (
        naive_departure >= sunrise if sunrise is not None else None
    )
    finish_before_sunset = (
        naive_finish <= sunset if sunset is not None else None
    )
    entirely_in_daylight = (
        departure_after_sunrise
        and finish_before_sunset
        and departure_date == finish_date
        if departure_after_sunrise is not None
        and finish_before_sunset is not None
        else None
    )

    warnings: list[str] = []
    if any(value > 0 for value in precipitation):
        warnings.append("precipitation_forecast")
    if any(code in _THUNDERSTORM_WEATHER_CODES for code in weather_codes):
        warnings.append("thunderstorm_forecast")
    if departure_after_sunrise is False:
        warnings.append("departure_before_sunrise")
    if finish_before_sunset is False:
        warnings.append("estimated_finish_after_sunset")

    return CyclingForecastContext(
        source=dict(OPEN_METEO_ATTRIBUTION),
        timezone=timezone_name,
        departure_time=local_departure.isoformat(),
        estimated_finish_time=local_finish.isoformat(),
        sampled_forecast_hours=tuple(
            str(hourly_times[index]) for index in indexes
        ),
        temperature_min_c=_optional_min(temperatures),
        temperature_max_c=_optional_max(temperatures),
        apparent_temperature_min_c=_optional_min(apparent_temperatures),
        apparent_temperature_max_c=_optional_max(apparent_temperatures),
        precipitation_probability_max_percentage=_optional_max(
            precipitation_probabilities
        ),
        precipitation_sum_mm=sum(precipitation) if precipitation else None,
        wind_speed_max_kmh=_optional_max(wind_speeds),
        wind_gusts_max_kmh=_optional_max(wind_gusts),
        wind_directions_degrees=tuple(
            sorted({int(value) for value in wind_directions})
        ),
        weather_codes=weather_codes,
        sunrise=sunrise_text,
        sunset=sunset_text,
        departure_after_sunrise=departure_after_sunrise,
        estimated_finish_before_sunset=finish_before_sunset,
        entirely_in_daylight=entirely_in_daylight,
        warnings=tuple(warnings),
    )
