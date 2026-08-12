"""Async client for the OpenRouteService API."""

from __future__ import annotations

from types import TracebackType
from typing import Any, cast

import httpx

from .auth import ICUConfig


class OpenRouteServiceAPIError(Exception):
    """Error returned while communicating with OpenRouteService."""

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


class OpenRouteServiceClient:
    """Small async client for OpenRouteService."""

    def __init__(self, config: ICUConfig) -> None:
        self.api_key = config.openrouteservice_api_key.strip()
        self.base_url = config.openrouteservice_base_url.strip().rstrip("/")
        self._http: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    async def __aenter__(self) -> OpenRouteServiceClient:
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={
                "Authorization": self.api_key,
            },
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

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, str | int | float] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError(
                "OpenRouteServiceClient must be used as an async context manager"
            )

        if not self.configured:
            raise RuntimeError("OpenRouteService API is not configured")

        try:
            response = await self._http.request(
                method,
                endpoint,
                params=params,
                json=json_body,
            )
        except httpx.RequestError as exc:
            raise OpenRouteServiceAPIError(
                f"Could not connect to OpenRouteService: {exc}",
                endpoint=endpoint,
            ) from exc

        if not response.is_success:
            message = (
                f"OpenRouteService API returned HTTP "
                f"{response.status_code}"
            )

            try:
                body: Any = response.json()

                if isinstance(body, dict):
                    error = cast(dict[str, Any], body).get("error")

                    if isinstance(error, dict):
                        detail = cast(
                            dict[str, Any],
                            error,
                        ).get("message")

                        if detail:
                            message += f": {detail}"

            except ValueError:
                if response.text:
                    message += f": {response.text[:500]}"

            raise OpenRouteServiceAPIError(
                message,
                status_code=response.status_code,
                endpoint=endpoint,
            )

        try:
            body = response.json()

        except ValueError as exc:
            raise OpenRouteServiceAPIError(
                "OpenRouteService returned invalid JSON",
                status_code=response.status_code,
                endpoint=endpoint,
            ) from exc

        if not isinstance(body, dict):
            raise OpenRouteServiceAPIError(
                "OpenRouteService returned an unexpected response",
                status_code=response.status_code,
                endpoint=endpoint,
            )

        return cast(dict[str, Any], body)

    async def geocode(
        self,
        text: str,
        *,
        size: int = 5,
        country: str | None = None,
        focus_lon: float | None = None,
        focus_lat: float | None = None,
    ) -> dict[str, Any]:
        """Resolve a place name using the OpenRouteService geocoder."""

        query = text.strip()

        if not query:
            raise ValueError("text must not be empty")

        if size < 1:
            raise ValueError("size must be >= 1")

        params: dict[str, str | int | float] = {
            "text": query,
            "size": size,
        }

        if country:
            params["boundary.country"] = country

        if focus_lon is not None:
            params["focus.point.lon"] = focus_lon

        if focus_lat is not None:
            params["focus.point.lat"] = focus_lat

        return await self._request(
            "GET",
            "/geocode/search",
            params=params,
        )

    async def snap(
        self,
        locations: list[list[float]],
        *,
        profile: str = "cycling-road",
        radius: float = 350.0,
    ) -> dict[str, Any]:
        """Snap coordinates to the routable network for a routing profile."""

        if not locations:
            raise ValueError("locations must not be empty")

        if len(locations) > 5000:
            raise ValueError("locations must contain at most 5000 points")

        for location in locations:
            if len(location) != 2:
                raise ValueError(
                    "each location must contain exactly [longitude, latitude]"
                )

            lon, lat = location

            if not -180 <= lon <= 180:
                raise ValueError("longitude must be between -180 and 180")

            if not -90 <= lat <= 90:
                raise ValueError("latitude must be between -90 and 90")

        routing_profile = profile.strip()

        if not routing_profile:
            raise ValueError("profile must not be empty")

        if radius <= 0:
            raise ValueError("radius must be > 0")

        return await self._request(
            "POST",
            f"/v2/snap/{routing_profile}/json",
            json_body={
                "locations": locations,
                "radius": radius,
            },
        )

    async def directions(
        self,
        coordinates: list[list[float]],
        *,
        profile: str = "cycling-road",
        elevation: bool = True,
        instructions: bool = False,
        extra_info: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calculate a route or a round trip through the given coordinates."""

        round_trip_requested = bool(
            options
            and "round_trip" in options
        )

        if not coordinates:
            raise ValueError("coordinates must not be empty")

        if len(coordinates) < 2 and not round_trip_requested:
            raise ValueError("coordinates must contain at least two points")

        for coordinate in coordinates:
            if len(coordinate) != 2:
                raise ValueError(
                    "each coordinate must contain exactly [longitude, latitude]"
                )

            lon, lat = coordinate

            if not -180 <= lon <= 180:
                raise ValueError("longitude must be between -180 and 180")

            if not -90 <= lat <= 90:
                raise ValueError("latitude must be between -90 and 90")

        routing_profile = profile.strip()

        if not routing_profile:
            raise ValueError("profile must not be empty")

        payload: dict[str, Any] = {
            "coordinates": coordinates,
            "elevation": elevation,
            "instructions": instructions,
        }

        if extra_info:
            payload["extra_info"] = extra_info

        if options:
            payload["options"] = options

        return await self._request(
            "POST",
            f"/v2/directions/{routing_profile}/geojson",
            json_body=payload,
        )
