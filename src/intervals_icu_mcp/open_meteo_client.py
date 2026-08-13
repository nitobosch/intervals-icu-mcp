"""Async client for the Open-Meteo forecast API."""

from __future__ import annotations

from datetime import date
from types import TracebackType
from typing import Any, cast

import httpx

from .auth import ICUConfig

OPEN_METEO_ATTRIBUTION = {
    "name": "Open-Meteo.com",
    "url": "https://open-meteo.com/",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
}

_HOURLY_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)
_DAILY_VARIABLES = ("sunrise", "sunset")


class OpenMeteoAPIError(Exception):
    """Error returned while communicating with Open-Meteo."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint


class OpenMeteoClient:
    """Small async client for origin-level forecasts."""

    def __init__(self, config: ICUConfig) -> None:
        self.base_url = config.open_meteo_base_url.strip().rstrip("/")
        self._http: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def __aenter__(self) -> OpenMeteoClient:
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=15.0,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Fetch hourly weather and daily daylight in local time."""

        if not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")
        if end_date < start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if self._http is None:
            raise RuntimeError(
                "OpenMeteoClient must be used as an async context manager"
            )
        if not self.configured:
            raise RuntimeError("Open-Meteo API is not configured")

        endpoint = "/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(_HOURLY_VARIABLES),
            "daily": ",".join(_DAILY_VARIABLES),
            "timezone": "auto",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        try:
            response = await self._http.get(endpoint, params=params)
        except httpx.RequestError as exc:
            raise OpenMeteoAPIError(
                f"Could not connect to Open-Meteo: {exc}",
                endpoint=endpoint,
            ) from exc

        if not response.is_success:
            message = f"Open-Meteo API returned HTTP {response.status_code}"
            try:
                body: Any = response.json()
                if isinstance(body, dict):
                    reason = cast(dict[str, Any], body).get("reason")
                    if reason:
                        message += f": {reason}"
            except ValueError:
                if response.text:
                    message += f": {response.text[:500]}"
            raise OpenMeteoAPIError(
                message,
                status_code=response.status_code,
                endpoint=endpoint,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise OpenMeteoAPIError(
                "Open-Meteo returned invalid JSON",
                status_code=response.status_code,
                endpoint=endpoint,
            ) from exc

        if not isinstance(body, dict):
            raise OpenMeteoAPIError(
                "Open-Meteo returned an unexpected response",
                status_code=response.status_code,
                endpoint=endpoint,
            )

        return cast(dict[str, Any], body)
