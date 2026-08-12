"""Tests for the OpenRouteService API client."""

import json

import httpx
import pytest
import respx
from httpx import Response

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.openrouteservice_client import (
    OpenRouteServiceAPIError,
    OpenRouteServiceClient,
)

ORS_BASE_URL = "https://api.openrouteservice.org"


def _config(
    *,
    api_key: str = "test-ors-key",
    base_url: str = ORS_BASE_URL,
) -> ICUConfig:
    return ICUConfig(
        openrouteservice_api_key=api_key,
        openrouteservice_base_url=base_url,
    )


def test_configured_requires_api_key() -> None:
    configured = OpenRouteServiceClient(_config())
    missing_key = OpenRouteServiceClient(_config(api_key=""))

    assert configured.configured is True
    assert missing_key.configured is False


async def test_geocode_uses_expected_endpoint_and_parameters() -> None:
    config = _config()

    with respx.mock(
        base_url=ORS_BASE_URL,
        assert_all_called=True,
    ) as router:
        route = router.get("/geocode/search").mock(
            return_value=Response(
                200,
                json={
                    "features": [
                        {
                            "properties": {
                                "label": "Visit Mallorca Estadi, Palma, PM, Spain",
                                "layer": "venue",
                            },
                            "geometry": {
                                "coordinates": [2.630108, 39.589985],
                            },
                        }
                    ]
                },
            )
        )

        async with OpenRouteServiceClient(config) as client:
            result = await client.geocode(
                "Estadi Mallorca Son Moix, Palma de Mallorca",
                size=5,
                country="ES",
                focus_lon=2.65,
                focus_lat=39.58,
            )

    assert len(result["features"]) == 1
    assert route.call_count == 1

    request = route.calls[0].request

    assert request.headers["Authorization"] == "test-ors-key"
    assert request.url.params["text"] == (
        "Estadi Mallorca Son Moix, Palma de Mallorca"
    )
    assert request.url.params["size"] == "5"
    assert request.url.params["boundary.country"] == "ES"
    assert request.url.params["focus.point.lon"] == "2.65"
    assert request.url.params["focus.point.lat"] == "39.58"


async def test_geocode_rejects_empty_text() -> None:
    async with OpenRouteServiceClient(_config()) as client:
        with pytest.raises(ValueError, match="text must not be empty"):
            await client.geocode("   ")


async def test_snap_uses_expected_endpoint_and_payload() -> None:
    config = _config()

    with respx.mock(
        base_url=ORS_BASE_URL,
        assert_all_called=True,
    ) as router:
        route = router.post("/v2/snap/cycling-road/json").mock(
            return_value=Response(
                200,
                json={
                    "locations": [
                        {
                            "location": [2.631246, 39.590265],
                            "snapped_distance": 102.39,
                        }
                    ]
                },
            )
        )

        async with OpenRouteServiceClient(config) as client:
            result = await client.snap(
                [[2.630108, 39.589985]],
                profile="cycling-road",
                radius=350.0,
            )

    assert result["locations"][0]["location"] == [
        2.631246,
        39.590265,
    ]

    request = route.calls[0].request
    payload = json.loads(request.content)

    assert payload == {
        "locations": [[2.630108, 39.589985]],
        "radius": 350.0,
    }


async def test_snap_rejects_invalid_coordinates() -> None:
    async with OpenRouteServiceClient(_config()) as client:
        with pytest.raises(
            ValueError,
            match="locations must not be empty",
        ):
            await client.snap([])

        with pytest.raises(
            ValueError,
            match="longitude must be between -180 and 180",
        ):
            await client.snap([[200.0, 39.5]])

        with pytest.raises(
            ValueError,
            match="latitude must be between -90 and 90",
        ):
            await client.snap([[2.6, 95.0]])


async def test_directions_uses_expected_endpoint_and_payload() -> None:
    config = _config()

    with respx.mock(
        base_url=ORS_BASE_URL,
        assert_all_called=True,
    ) as router:
        route = router.post(
            "/v2/directions/cycling-road/geojson"
        ).mock(
            return_value=Response(
                200,
                json={
                    "features": [
                        {
                            "properties": {
                                "summary": {
                                    "distance": 3740.0,
                                    "duration": 630.0,
                                },
                                "way_points": [0, 119],
                                "segments": [{}],
                                "extras": {
                                    "surface": {},
                                    "waytype": {},
                                    "steepness": {},
                                    "suitability": {},
                                },
                            },
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [
                                    [2.631246, 39.590265, 61.0],
                                    [2.656712, 39.575678, 30.0],
                                ],
                            },
                        }
                    ]
                },
            )
        )

        async with OpenRouteServiceClient(config) as client:
            result = await client.directions(
                [
                    [2.631246, 39.590265],
                    [2.656712, 39.575678],
                ],
                profile="cycling-road",
                elevation=True,
                instructions=True,
                extra_info=[
                    "surface",
                    "waytype",
                    "steepness",
                    "suitability",
                ],
                options={
                    "avoid_features": [
                        "ferries",
                        "steps",
                    ]
                },
            )

    assert result["features"][0]["properties"]["summary"]["distance"] == 3740.0

    request = route.calls[0].request
    payload = json.loads(request.content)

    assert payload["coordinates"] == [
        [2.631246, 39.590265],
        [2.656712, 39.575678],
    ]
    assert payload["elevation"] is True
    assert payload["instructions"] is True
    assert payload["extra_info"] == [
        "surface",
        "waytype",
        "steepness",
        "suitability",
    ]
    assert payload["options"] == {
        "avoid_features": [
            "ferries",
            "steps",
        ]
    }


async def test_directions_requires_at_least_two_coordinates() -> None:
    async with OpenRouteServiceClient(_config()) as client:
        with pytest.raises(
            ValueError,
            match="coordinates must contain at least two points",
        ):
            await client.directions(
                [[2.631246, 39.590265]]
            )


async def test_directions_allows_one_coordinate_for_round_trip() -> None:
    config = _config()

    with respx.mock(
        base_url=ORS_BASE_URL,
        assert_all_called=True,
    ) as router:
        route = router.post(
            "/v2/directions/cycling-road/geojson"
        ).mock(
            return_value=Response(
                200,
                json={"features": []},
            )
        )

        async with OpenRouteServiceClient(config) as client:
            await client.directions(
                [[2.631246, 39.590265]],
                options={
                    "round_trip": {
                        "length": 50_000,
                        "points": 5,
                        "seed": 1,
                    }
                },
            )

    payload = json.loads(route.calls[0].request.content)

    assert payload["coordinates"] == [[2.631246, 39.590265]]
    assert payload["options"] == {
        "round_trip": {
            "length": 50_000,
            "points": 5,
            "seed": 1,
        }
    }


async def test_directions_rejects_empty_round_trip_coordinates() -> None:
    async with OpenRouteServiceClient(_config()) as client:
        with pytest.raises(
            ValueError,
            match="coordinates must not be empty",
        ):
            await client.directions(
                [],
                options={
                    "round_trip": {
                        "length": 50_000,
                        "points": 5,
                    }
                },
            )


async def test_http_error_is_wrapped() -> None:
    config = _config()

    with respx.mock(base_url=ORS_BASE_URL) as router:
        router.get("/geocode/search").mock(
            return_value=Response(
                429,
                json={
                    "error": {
                        "code": 2004,
                        "message": "Rate limit exceeded",
                    }
                },
            )
        )

        async with OpenRouteServiceClient(config) as client:
            with pytest.raises(
                OpenRouteServiceAPIError
            ) as exc_info:
                await client.geocode("Palma")

    error = exc_info.value

    assert error.status_code == 429
    assert error.endpoint == "/geocode/search"
    assert "Rate limit exceeded" in str(error)


async def test_connection_error_is_wrapped() -> None:
    config = _config()

    with respx.mock(base_url=ORS_BASE_URL) as router:
        router.get("/geocode/search").mock(
            side_effect=httpx.ConnectError(
                "Connection failed"
            )
        )

        async with OpenRouteServiceClient(config) as client:
            with pytest.raises(
                OpenRouteServiceAPIError,
                match="Could not connect to OpenRouteService",
            ):
                await client.geocode("Palma")


async def test_invalid_json_response_is_wrapped() -> None:
    config = _config()

    with respx.mock(base_url=ORS_BASE_URL) as router:
        router.get("/geocode/search").mock(
            return_value=Response(
                200,
                text="this is not json",
            )
        )

        async with OpenRouteServiceClient(config) as client:
            with pytest.raises(
                OpenRouteServiceAPIError,
                match="returned invalid JSON",
            ):
                await client.geocode("Palma")


async def test_client_must_be_used_as_context_manager() -> None:
    client = OpenRouteServiceClient(_config())

    with pytest.raises(
        RuntimeError,
        match="must be used as an async context manager",
    ):
        await client.geocode("Palma")
