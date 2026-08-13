"""Tests for generic ordered-point cycling route construction."""

import base64
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import BlobResourceContents

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.gpx_delivery import CyclingToolResponse, response_text_and_resources
from intervals_icu_mcp.openrouteservice_client import OpenRouteServiceAPIError
from intervals_icu_mcp.tools import cycling_route_builder as builder
from intervals_icu_mcp.tools.routing import (
    CyclingRoute,
    CyclingRouteQualityMetrics,
    LocationResolutionError,
    ResolvedLocation,
    RouteCoordinate,
)

_NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}


def _resolved(value: str, longitude: float, latitude: float) -> ResolvedLocation:
    return ResolvedLocation(
        input_value=value,
        source="geocode",
        label=f"Resolved {value}",
        original_longitude=longitude - 0.0001,
        original_latitude=latitude - 0.0001,
        longitude=longitude,
        latitude=latitude,
        snapped_distance_m=12.0,
    )


def _route() -> CyclingRoute:
    geometry = tuple(
        RouteCoordinate(2.63 + index * 0.001, 39.59 + index * 0.0002, 100 + index)
        for index in range(8)
    )
    return CyclingRoute(
        distance_m=750.0,
        duration_s=180.0,
        elevation_gain_m=22.0,
        elevation_loss_m=7.0,
        ors_ascent_m=23.0,
        ors_descent_m=8.0,
        geometry=geometry,
        waypoint_indices=(0, len(geometry) - 1),
        segments=(),
        extras={},
    )


def _quality(route: CyclingRoute) -> CyclingRouteQualityMetrics:
    return CyclingRouteQualityMetrics(
        total_distance_m=route.distance_m,
        total_duration_s=route.duration_s,
        elevation_gain_m=route.elevation_gain_m,
        elevation_loss_m=route.elevation_loss_m,
        asphalt_percentage=90.0,
        unknown_surface_percentage=10.0,
        road_or_cycleway_percentage=95.0,
        footway_percentage=0.0,
        suitability_7_plus_percentage=85.0,
        suitability_8_plus_percentage=70.0,
        incline_7_plus_percentage=5.0,
        incline_10_plus_percentage=1.0,
        decline_7_plus_percentage=2.0,
        decline_10_plus_percentage=0.0,
        maneuver_count=4,
        maneuver_rate_per_hour=80.0,
        roundabout_count=1,
        sharp_turn_count=1,
    )


class FakeClient:
    def __init__(self, _config: ICUConfig) -> None:
        pass

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None


async def _call(
    monkeypatch: pytest.MonkeyPatch,
    locations: list[str],
    *,
    include_gpx: bool = False,
) -> tuple[CyclingToolResponse, AsyncMock, AsyncMock, CyclingRoute]:
    route = _route()
    resolved = {
        value: _resolved(value, 2.63 + index * 0.01, 39.59 + index * 0.01)
        for index, value in enumerate(dict.fromkeys(locations))
    }
    resolve = AsyncMock(side_effect=lambda _client, value, **_kwargs: resolved[value])
    build = AsyncMock(return_value=route)
    monkeypatch.setattr(builder, "OpenRouteServiceClient", FakeClient)
    monkeypatch.setattr(builder, "resolve_location", resolve)
    monkeypatch.setattr(builder, "build_cycling_route", build)
    monkeypatch.setattr(builder, "calculate_route_timeline", lambda _route: object())
    monkeypatch.setattr(
        builder,
        "calculate_cycling_route_quality_metrics",
        lambda supplied_route, _timeline: _quality(supplied_route),
    )
    context = SimpleNamespace(
        get_state=AsyncMock(return_value=ICUConfig(openrouteservice_api_key="test-ors-key"))
    )
    result = await builder.build_ordered_cycling_route(
        locations=locations,
        include_gpx=include_gpx,
        ctx=context,  # type: ignore[arg-type]
    )
    return result, resolve, build, route


async def test_builds_simple_ordered_route(monkeypatch: pytest.MonkeyPatch) -> None:
    result, resolve, build, route = await _call(monkeypatch, ["A", "B"])
    response_text, resources = response_text_and_resources(result)
    response = json.loads(response_text)

    assert resources == []
    assert [item[0][1] for item in resolve.await_args_list] == ["A", "B"]
    assert build.await_args is not None
    assert [location.input_value for location in build.await_args.args[1]] == ["A", "B"]
    assert response["data"]["route"]["distance_meters"] == route.distance_m
    assert response["data"]["route"]["duration_seconds"] == route.duration_s
    assert response["data"]["route"]["elevation_gain_meters"] == 22.0
    assert response["data"]["route"]["elevation_loss_meters"] == 7.0
    assert response["data"]["route_quality"]["asphalt_percentage"] == 90.0
    assert "gpx" not in response["data"]["route"]


async def test_preserves_explicit_circular_route_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, resolve, build, _route_value = await _call(monkeypatch, ["A", "B", "A"])
    response = json.loads(response_text_and_resources(result)[0])

    assert [item[0][1] for item in resolve.await_args_list] == ["A", "B", "A"]
    assert build.await_args is not None
    assert [location.input_value for location in build.await_args.args[1]] == [
        "A",
        "B",
        "A",
    ]
    assert [item["input"] for item in response["data"]["resolved_locations"]] == [
        "A",
        "B",
        "A",
    ]


async def test_exposes_resolved_and_snapped_coordinates_for_multiple_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _resolve, _build, route = await _call(
        monkeypatch, ["Address A", "Pass B", "Town C", "Address A"]
    )
    response = json.loads(response_text_and_resources(result)[0])
    points = response["data"]["resolved_locations"]

    assert len(points) == 4
    assert points[0]["input"] == "Address A"
    assert points[0]["source"] == "geocode"
    assert points[0]["resolved_name"] == "Resolved Address A"
    assert points[0]["resolved_coordinates"] == pytest.approx(
        {"longitude": 2.6299, "latitude": 39.5899}
    )
    assert points[0]["routing_coordinates"] == pytest.approx({"longitude": 2.63, "latitude": 39.59})
    assert points[0]["snapped_distance_meters"] == 12.0
    assert response["data"]["route"]["geometry"]["coordinates"] == [
        [point.longitude, point.latitude, point.elevation_m] for point in route.geometry
    ]


async def test_include_gpx_delivers_exact_serializer_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _resolve, _build, route = await _call(monkeypatch, ["A", "B", "A"], include_gpx=True)
    response_text, resources = response_text_and_resources(result)
    response = json.loads(response_text)
    assert len(resources) == 1
    resource = resources[0].resource
    assert isinstance(resource, BlobResourceContents)
    assert resource.mimeType == "application/gpx+xml"
    filename = str(resource.uri).removeprefix("file:///")
    assert filename.startswith("cycling-route-")
    assert filename.endswith(".gpx")
    assert len(filename.removeprefix("cycling-route-").removesuffix(".gpx")) == 12

    delivered = base64.b64decode(resource.blob, validate=True)
    compatibility = base64.b64decode(
        response["data"]["route"]["gpx"]["content_base64"], validate=True
    )
    expected = builder.cycling_route_to_gpx(
        route,
        name=f"Cycling route {response['data']['route']['id']}",
        description="Ordered-point route generated by intervals-icu-mcp.",
    )
    assert delivered == compatibility == expected
    assert len(delivered) == response["data"]["route"]["gpx"]["size_bytes"]
    root = ET.fromstring(delivered)
    assert root.findall("gpx:wpt", _NAMESPACE) == []
    assert len(root.findall("gpx:trk/gpx:trkseg/gpx:trkpt", _NAMESPACE)) == len(route.geometry)


@pytest.mark.parametrize("locations", [[], ["A"], ["A", " "]])
async def test_validates_locations_before_ors(
    monkeypatch: pytest.MonkeyPatch,
    locations: list[str],
) -> None:
    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS must not be created")

    monkeypatch.setattr(builder, "OpenRouteServiceClient", UnexpectedClient)
    context = SimpleNamespace(
        get_state=AsyncMock(return_value=ICUConfig(openrouteservice_api_key="test-ors-key"))
    )
    result = await builder.build_ordered_cycling_route(
        locations=locations,
        ctx=context,  # type: ignore[arg-type]
    )
    response = json.loads(response_text_and_resources(result)[0])
    assert response["error"]["type"] == "validation_error"


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (LocationResolutionError("unknown location"), "location_resolution_error"),
        (OpenRouteServiceAPIError("ORS unavailable"), "api_error"),
    ],
)
async def test_maps_location_and_ors_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    error_type: str,
) -> None:
    monkeypatch.setattr(builder, "OpenRouteServiceClient", FakeClient)
    monkeypatch.setattr(builder, "resolve_location", AsyncMock(side_effect=error))
    context = SimpleNamespace(
        get_state=AsyncMock(return_value=ICUConfig(openrouteservice_api_key="test-ors-key"))
    )
    result = await builder.build_ordered_cycling_route(
        locations=["A", "B"],
        ctx=context,  # type: ignore[arg-type]
    )
    response = json.loads(response_text_and_resources(result)[0])
    assert response["error"]["type"] == error_type
