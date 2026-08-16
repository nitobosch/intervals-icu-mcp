"""Contracts for exposing Intervals.icu's official Strava activity reference."""

import json
from unittest.mock import AsyncMock, MagicMock

from httpx import Response

from intervals_icu_mcp.models import Activity, ActivitySummary
from intervals_icu_mcp.tools.activities import (
    get_activities_by_date,
    get_activity_details,
    get_recent_activities,
)

STRAVA_ACTIVITY_ID = "900719925474099312345"


def _ctx(config):
    ctx = MagicMock()
    ctx.get_state = AsyncMock(return_value=config)
    return ctx


def _raw_activity(*, include_strava_id: bool = True) -> dict:
    activity = {
        "id": "i176311262",
        "name": "Palma Ciclismo en ruta",
        "start_date_local": "2026-08-16T06:37:11",
        "type": "Ride",
        "source": "GARMIN_CONNECT",
        "external_id": "23993879938",
    }
    if include_strava_id:
        activity["strava_id"] = STRAVA_ACTIVITY_ID
    return activity


def test_activity_summary_preserves_large_strava_id_as_string():
    raw = _raw_activity()
    raw["strava_id"] = int(STRAVA_ACTIVITY_ID)

    activity = ActivitySummary(**raw)

    assert activity.strava_id == STRAVA_ACTIVITY_ID
    assert isinstance(activity.strava_id, str)


def test_activity_inherits_and_preserves_strava_id():
    activity = Activity(**_raw_activity())

    assert activity.strava_id == STRAVA_ACTIVITY_ID
    assert isinstance(activity.strava_id, str)


async def test_recent_activities_exposes_garmin_source_and_strava_id(
    mock_config,
    respx_mock,
):
    respx_mock.get("/athlete/i123456/activities").mock(
        return_value=Response(200, json=[_raw_activity()])
    )

    result = await get_recent_activities(ctx=_ctx(mock_config))
    item = json.loads(result)["data"]["activities"][0]

    assert item["id"] == "i176311262"
    assert item["source"] == "GARMIN_CONNECT"
    assert item["strava_activity_id"] == STRAVA_ACTIVITY_ID
    assert isinstance(item["strava_activity_id"], str)


async def test_activities_by_date_exposes_same_strava_id(
    mock_config,
    respx_mock,
):
    respx_mock.get("/athlete/i123456/activities").mock(
        return_value=Response(200, json=[_raw_activity()])
    )

    result = await get_activities_by_date(
        oldest="2026-08-16",
        newest="2026-08-16",
        ctx=_ctx(mock_config),
    )
    item = json.loads(result)["data"]["activities"][0]

    assert item["source"] == "GARMIN_CONNECT"
    assert item["strava_activity_id"] == STRAVA_ACTIVITY_ID


async def test_activity_details_exposes_exact_same_strava_id(
    mock_config,
    respx_mock,
):
    respx_mock.get("/activity/i176311262").mock(
        return_value=Response(200, json=_raw_activity())
    )

    result = await get_activity_details(
        activity_id="i176311262",
        ctx=_ctx(mock_config),
    )
    data = json.loads(result)["data"]

    assert data["id"] == "i176311262"
    assert data["source"] == "GARMIN_CONNECT"
    assert data["strava_activity_id"] == STRAVA_ACTIVITY_ID
    assert isinstance(data["strava_activity_id"], str)


async def test_external_id_never_substitutes_for_missing_strava_id(
    mock_config,
    respx_mock,
):
    raw = _raw_activity(include_strava_id=False)
    respx_mock.get("/athlete/i123456/activities").mock(
        return_value=Response(200, json=[raw])
    )
    respx_mock.get("/activity/i176311262").mock(
        return_value=Response(200, json=raw)
    )

    recent = json.loads(
        await get_recent_activities(ctx=_ctx(mock_config))
    )["data"]["activities"][0]
    detail = json.loads(
        await get_activity_details(
            activity_id="i176311262",
            ctx=_ctx(mock_config),
        )
    )["data"]

    assert recent["source"] == "GARMIN_CONNECT"
    assert detail["source"] == "GARMIN_CONNECT"
    assert "strava_activity_id" not in recent
    assert "strava_activity_id" not in detail
    assert "external_id" not in recent
    assert "external_id" not in detail
