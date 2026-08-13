"""Tests for high-level cycling-profile MCP tools."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intervals_icu_mcp.tools import cycling_profiles


async def test_profiled_route_delegates_resolved_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock(
        return_value=json.dumps(
            {
                "data": {
                    "best_route": {"candidate_id": "round-trip-1"},
                },
                "metadata": {"query_type": "cycling_training_route"},
            }
        )
    )
    monkeypatch.setattr(
        cycling_profiles,
        "find_cycling_training_route",
        route_tool,
    )
    ctx = SimpleNamespace()

    result = await cycling_profiles.find_profiled_cycling_training_route(
        profile="steady_climb",
        start_location="39.59,2.63",
        target_distance_km=50.0,
        work_duration_minutes=35.0,
        target_duration_minutes=120.0,
        departure_time="2026-08-14T08:00:00+02:00",
        candidate_count=4,
        include_gpx=True,
        avoid_features=["ferries"],
        ctx=ctx,  # type: ignore[arg-type]
    )

    response = json.loads(result)
    profile = response["data"]["training_profile"]
    assert profile["profile"] == "steady_climb"
    assert profile["training_durations_minutes"] == [35.0]
    assert profile["hard_constraints_applied"] == []
    route_tool.assert_awaited_once()
    kwargs = route_tool.await_args.kwargs
    assert kwargs["start_location"] == "39.59,2.63"
    assert kwargs["target_distance_km"] == 50.0
    assert kwargs["target_duration_minutes"] == 120.0
    assert kwargs["training_durations_minutes"] == [35.0]
    assert kwargs["training_repetitions"] == 1
    assert kwargs["recovery_min_minutes"] is None
    assert kwargs["recovery_max_minutes"] is None
    assert kwargs["candidate_count"] == 4
    assert kwargs["include_gpx"] is True
    assert kwargs["avoid_features"] == ["ferries"]
    assert kwargs["departure_time"] == "2026-08-14T08:00:00+02:00"
    assert kwargs["ctx"] is ctx


async def test_profiled_route_uses_profile_duration_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock(return_value='{"data": {}}')
    monkeypatch.setattr(
        cycling_profiles,
        "find_cycling_training_route",
        route_tool,
    )

    await cycling_profiles.find_profiled_cycling_training_route(
        profile="steady_climb",
        start_location="Start",
        target_distance_km=30.0,
        ctx=SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert route_tool.await_args.kwargs["training_durations_minutes"] == [
        20.0,
        30.0,
        40.0,
    ]


async def test_profiled_route_validates_before_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock()
    monkeypatch.setattr(
        cycling_profiles,
        "find_cycling_training_route",
        route_tool,
    )

    result = await cycling_profiles.find_profiled_cycling_training_route(
        profile="steady_climb",
        start_location="Start",
        target_distance_km=30.0,
        work_duration_minutes=0.0,
        ctx=SimpleNamespace(),  # type: ignore[arg-type]
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "work_duration_minutes" in response["error"]["message"]
    route_tool.assert_not_awaited()


async def test_profiled_route_preserves_routing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock(
        return_value='{"error": {"type": "not_found", "message": "none"}}'
    )
    monkeypatch.setattr(
        cycling_profiles,
        "find_cycling_training_route",
        route_tool,
    )
    result = await cycling_profiles.find_profiled_cycling_training_route(
        profile="steady_climb",
        start_location="Start",
        target_distance_km=30.0,
        ctx=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert json.loads(result)["error"]["type"] == "not_found"
