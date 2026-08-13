"""Tests for the integrated cycling-coach MCP tool."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intervals_icu_mcp.tools import cycling_coach


async def test_coach_delegates_the_auditable_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock(
        return_value=json.dumps({"data": {"best_route": {"candidate_id": "r1"}}})
    )
    monkeypatch.setattr(
        cycling_coach,
        "find_profiled_cycling_training_route",
        route_tool,
    )
    ctx = SimpleNamespace()

    result = await cycling_coach.find_cycling_coach_route(
        profile="sweet_spot_climb",
        start_location=" Son Moix ",
        target_distance_km=60.0,
        available_time_minutes=135.0,
        departure_time="2026-08-14T08:00:00+02:00",
        candidate_count=4,
        avoid_features=["ferries"],
        ctx=ctx,  # type: ignore[arg-type]
    )

    response = json.loads(result)
    plan = response["data"]["coach_plan"]
    assert plan["start_location"] == "Son Moix"
    assert plan["target_duration_minutes"] == 135.0
    assert plan["duration_source"] == "available_time"
    assert plan["profile"]["training_repetitions"] == 3
    route_tool.assert_awaited_once_with(
        profile="sweet_spot_climb",
        start_location="Son Moix",
        target_distance_km=60.0,
        work_duration_minutes=None,
        target_duration_minutes=135.0,
        departure_time="2026-08-14T08:00:00+02:00",
        candidate_count=4,
        include_gpx=True,
        avoid_features=["ferries"],
        ctx=ctx,
    )


async def test_coach_preserves_explicit_shorter_route_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock(return_value='{"data": {}}')
    monkeypatch.setattr(
        cycling_coach,
        "find_profiled_cycling_training_route",
        route_tool,
    )

    result = await cycling_coach.find_cycling_coach_route(
        profile="steady_climb",
        start_location="Start",
        target_distance_km=50.0,
        available_time_minutes=150.0,
        target_duration_minutes=120.0,
        include_gpx=False,
        ctx=SimpleNamespace(),  # type: ignore[arg-type]
    )

    plan = json.loads(result)["data"]["coach_plan"]
    assert plan["target_duration_minutes"] == 120.0
    assert plan["duration_source"] == "explicit_target"
    assert route_tool.await_args.kwargs["include_gpx"] is False


async def test_coach_validates_before_route_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_tool = AsyncMock()
    monkeypatch.setattr(
        cycling_coach,
        "find_profiled_cycling_training_route",
        route_tool,
    )

    result = await cycling_coach.find_cycling_coach_route(
        profile="steady_climb",
        start_location="Start",
        target_distance_km=50.0,
        available_time_minutes=120.0,
        target_duration_minutes=130.0,
        ctx=None,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "must not exceed" in response["error"]["message"]
    route_tool.assert_not_awaited()
