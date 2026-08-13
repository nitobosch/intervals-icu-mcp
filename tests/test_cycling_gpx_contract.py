"""Integrated contracts for navigation-ready cycling GPX responses."""

import base64
import json
import math
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.tools import routing
from intervals_icu_mcp.tools.routing import (
    CyclingRoute,
    CyclingRouteCandidate,
    CyclingRouteCandidateAnalysis,
    CyclingRouteCandidateSearchResult,
    RouteCoordinate,
    RouteTimelinePoint,
    RouteTrainingWindow,
)

_GPX_NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}
_EARTH_RADIUS_M = 6_371_008.8


def _track_distance_m(coordinates: list[tuple[float, float]]) -> float:
    distance_m = 0.0
    for first, second in zip(coordinates, coordinates[1:], strict=False):
        first_latitude = math.radians(first[0])
        second_latitude = math.radians(second[0])
        latitude_delta = second_latitude - first_latitude
        longitude_delta = math.radians(second[1] - first[1])
        haversine = (
            math.sin(latitude_delta / 2.0) ** 2
            + math.cos(first_latitude)
            * math.cos(second_latitude)
            * math.sin(longitude_delta / 2.0) ** 2
        )
        distance_m += 2.0 * _EARTH_RADIUS_M * math.asin(math.sqrt(haversine))
    return distance_m


def _route() -> CyclingRoute:
    geometry = tuple(
        RouteCoordinate(
            longitude=2.630108 + index * 0.001,
            latitude=39.589985 + (index % 3) * 0.0002,
            elevation_m=100.0 + index * 5.0,
        )
        for index in range(10)
    )
    distance_m = _track_distance_m([(point.latitude, point.longitude) for point in geometry])
    return CyclingRoute(
        distance_m=distance_m,
        duration_s=3_600.0,
        elevation_gain_m=45.0,
        elevation_loss_m=0.0,
        ors_ascent_m=45.0,
        ors_descent_m=0.0,
        geometry=geometry,
        waypoint_indices=(0, len(geometry) - 1),
        segments=(),
        extras={},
    )


def _window(route: CyclingRoute, start_index: int, end_index: int) -> RouteTrainingWindow:
    start = RouteTimelinePoint(
        geometry_index=start_index,
        distance_m=route.distance_m * start_index / (len(route.geometry) - 1),
        time_s=route.duration_s * start_index / (len(route.geometry) - 1),
        elevation_m=route.geometry[start_index].elevation_m,
    )
    end = RouteTimelinePoint(
        geometry_index=end_index,
        distance_m=route.distance_m * end_index / (len(route.geometry) - 1),
        time_s=route.duration_s * end_index / (len(route.geometry) - 1),
        elevation_m=route.geometry[end_index].elevation_m,
    )
    return RouteTrainingWindow(
        start=start,
        end=end,
        distance_m=end.distance_m - start.distance_m,
        duration_s=end.time_s - start.time_s,
        elevation_gain_m=0.0,
        elevation_loss_m=0.0,
        net_elevation_gain_m=0.0,
    )


async def _export_response(
    monkeypatch: pytest.MonkeyPatch,
    route: CyclingRoute,
    *,
    training_window: RouteTrainingWindow,
    training_block_sequence: Any = None,
) -> dict[str, object]:
    origin = routing.ResolvedLocation(
        input_value="Start",
        source="coordinates",
        label="Start",
        original_longitude=route.geometry[0].longitude,
        original_latitude=route.geometry[0].latitude,
        longitude=route.geometry[0].longitude,
        latitude=route.geometry[0].latitude,
        snapped_distance_m=0.0,
    )
    analysis = CyclingRouteCandidateAnalysis(
        candidate=CyclingRouteCandidate(
            candidate_id="round-trip-1",
            strategy="ors_round_trip",
            seed=0,
            target_distance_m=route.distance_m,
            route=route,
        ),
        best_training_window=cast(Any, SimpleNamespace(window=training_window)),
        best_by_duration=(),
        best_training_block_sequence=training_block_sequence,
    )
    search = AsyncMock(
        return_value=CyclingRouteCandidateSearchResult(
            ranked=(analysis,),
            candidates_generated=1,
            candidates_after_distance_filter=1,
            candidates_after_duration_filter=1,
            candidates_after_deduplication=1,
        )
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

    monkeypatch.setattr(routing, "OpenRouteServiceClient", FakeClient)
    monkeypatch.setattr(
        routing,
        "resolve_location",
        AsyncMock(return_value=origin),
    )
    monkeypatch.setattr(
        routing,
        "find_cycling_training_route_candidates",
        search,
    )
    monkeypatch.setattr(
        routing,
        "serialize_cycling_route_candidate_analysis",
        lambda _analysis: {"route": {"distance_meters": route.distance_m}},
    )
    context = SimpleNamespace(
        get_state=AsyncMock(return_value=ICUConfig(openrouteservice_api_key="test-ors-key"))
    )

    raw_response = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=route.distance_m / 1_000.0,
        training_durations_minutes=[20.0],
        training_repetitions=(3 if training_block_sequence is not None else 1),
        recovery_min_minutes=(5.0 if training_block_sequence is not None else None),
        recovery_max_minutes=(8.0 if training_block_sequence is not None else None),
        candidate_count=2,
        include_gpx=True,
        ctx=context,  # type: ignore[arg-type]
    )
    return json.loads(raw_response)


def _assert_navigation_track_contract(
    response: dict[str, object],
    route: CyclingRoute,
) -> tuple[ET.Element, list[ET.Element]]:
    data = response["data"]
    assert isinstance(data, dict)
    best_route = data["best_route"]
    assert isinstance(best_route, dict)
    gpx = best_route["gpx"]
    assert isinstance(gpx, dict)
    encoded = gpx["content_base64"]
    assert isinstance(encoded, str)
    content = base64.b64decode(encoded, validate=True)
    assert len(content) == gpx["size_bytes"]

    root = ET.fromstring(content)
    tracks = root.findall("gpx:trk", _GPX_NAMESPACE)
    track_segments = root.findall("gpx:trk/gpx:trkseg", _GPX_NAMESPACE)
    track_points = root.findall(
        "gpx:trk/gpx:trkseg/gpx:trkpt",
        _GPX_NAMESPACE,
    )

    assert len(tracks) == 1
    assert len(track_segments) == 1
    assert len(track_points) == len(route.geometry)
    assert track_points  # Explicitly forbid a waypoint-only GPX regression.
    assert root.findall(".//gpx:rtept", _GPX_NAMESPACE) == []

    track_coordinates = [
        (float(point.attrib["lat"]), float(point.attrib["lon"])) for point in track_points
    ]
    assert track_coordinates[0] == pytest.approx(
        (route.geometry[0].latitude, route.geometry[0].longitude)
    )
    assert track_coordinates[-1] == pytest.approx(
        (route.geometry[-1].latitude, route.geometry[-1].longitude)
    )
    assert _track_distance_m(track_coordinates) == pytest.approx(
        route.distance_m,
        rel=1e-6,
    )
    return root, track_points


async def test_continuous_route_gpx_response_preserves_navigation_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    training_window = _window(route, 2, 7)

    response = await _export_response(
        monkeypatch,
        route,
        training_window=training_window,
    )
    root, track_points = _assert_navigation_track_contract(response, route)
    waypoints = root.findall("gpx:wpt", _GPX_NAMESPACE)

    assert [waypoint.findtext("gpx:name", namespaces=_GPX_NAMESPACE) for waypoint in waypoints] == [
        "TRAINING START",
        "TRAINING END",
    ]
    assert len(waypoints) == 2
    assert len(track_points) > len(waypoints)


async def test_multiblock_route_gpx_response_preserves_navigation_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    windows = tuple(_window(route, start, end) for start, end in ((1, 2), (4, 5), (7, 8)))
    sequence = SimpleNamespace(
        work_blocks=tuple(SimpleNamespace(window=window) for window in windows)
    )

    response = await _export_response(
        monkeypatch,
        route,
        training_window=windows[0],
        training_block_sequence=sequence,
    )
    root, track_points = _assert_navigation_track_contract(response, route)
    waypoints = root.findall("gpx:wpt", _GPX_NAMESPACE)

    assert [waypoint.findtext("gpx:name", namespaces=_GPX_NAMESPACE) for waypoint in waypoints] == [
        "BLOCK 1 START",
        "BLOCK 1 END",
        "BLOCK 2 START",
        "BLOCK 2 END",
        "BLOCK 3 START",
        "BLOCK 3 END",
    ]
    assert len(waypoints) == 6
    assert len(track_points) > len(waypoints)
