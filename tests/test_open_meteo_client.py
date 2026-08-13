"""Tests for the Open-Meteo forecast API client."""

from datetime import date

import httpx
import pytest
import respx
from httpx import Response

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.open_meteo_client import (
    OpenMeteoAPIError,
    OpenMeteoClient,
)

OPEN_METEO_BASE_URL = "https://api.open-meteo.com"


def _config(*, base_url: str = OPEN_METEO_BASE_URL) -> ICUConfig:
    return ICUConfig(open_meteo_base_url=base_url)


async def test_forecast_requests_weather_and_daylight_in_local_time() -> None:
    with respx.mock(
        base_url=OPEN_METEO_BASE_URL,
        assert_all_called=True,
    ) as router:
        route = router.get("/v1/forecast").mock(
            return_value=Response(
                200,
                json={
                    "timezone": "Europe/Madrid",
                    "hourly": {
                        "time": ["2026-08-13T08:00"],
                        "temperature_2m": [24.0],
                    },
                    "daily": {
                        "time": ["2026-08-13"],
                        "sunrise": ["2026-08-13T07:03"],
                        "sunset": ["2026-08-13T20:48"],
                    },
                },
            )
        )

        async with OpenMeteoClient(_config()) as client:
            result = await client.forecast(
                39.59,
                2.63,
                start_date=date(2026, 8, 13),
                end_date=date(2026, 8, 14),
            )

    assert result["timezone"] == "Europe/Madrid"
    request = route.calls[0].request
    assert request.url.params["latitude"] == "39.59"
    assert request.url.params["longitude"] == "2.63"
    assert request.url.params["timezone"] == "auto"
    assert request.url.params["start_date"] == "2026-08-13"
    assert request.url.params["end_date"] == "2026-08-14"
    assert request.url.params["daily"] == "sunrise,sunset"
    assert request.url.params["hourly"] == (
        "temperature_2m,apparent_temperature,"
        "precipitation_probability,precipitation,weather_code,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
    )


@pytest.mark.parametrize(
    ("latitude", "longitude", "message"),
    [
        (91.0, 2.63, "latitude must be between -90 and 90"),
        (39.59, 181.0, "longitude must be between -180 and 180"),
    ],
)
async def test_forecast_rejects_invalid_coordinates(
    latitude: float,
    longitude: float,
    message: str,
) -> None:
    async with OpenMeteoClient(_config()) as client:
        with pytest.raises(ValueError, match=message):
            await client.forecast(
                latitude,
                longitude,
                start_date=date(2026, 8, 13),
                end_date=date(2026, 8, 13),
            )


async def test_forecast_rejects_reversed_date_range() -> None:
    async with OpenMeteoClient(_config()) as client:
        with pytest.raises(
            ValueError,
            match="end_date must not be earlier than start_date",
        ):
            await client.forecast(
                39.59,
                2.63,
                start_date=date(2026, 8, 14),
                end_date=date(2026, 8, 13),
            )


async def test_forecast_wraps_api_error_reason() -> None:
    with respx.mock(base_url=OPEN_METEO_BASE_URL) as router:
        router.get("/v1/forecast").mock(
            return_value=Response(
                400,
                json={"reason": "Invalid date"},
            )
        )
        async with OpenMeteoClient(_config()) as client:
            with pytest.raises(
                OpenMeteoAPIError,
                match="HTTP 400: Invalid date",
            ):
                await client.forecast(
                    39.59,
                    2.63,
                    start_date=date(2026, 8, 13),
                    end_date=date(2026, 8, 13),
                )


async def test_forecast_wraps_connection_error() -> None:
    with respx.mock(base_url=OPEN_METEO_BASE_URL) as router:
        router.get("/v1/forecast").mock(
            side_effect=httpx.ConnectError("network unavailable")
        )
        async with OpenMeteoClient(_config()) as client:
            with pytest.raises(
                OpenMeteoAPIError,
                match="Could not connect to Open-Meteo",
            ):
                await client.forecast(
                    39.59,
                    2.63,
                    start_date=date(2026, 8, 13),
                    end_date=date(2026, 8, 13),
                )
