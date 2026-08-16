"""Tests for Strava MCP formatting and stream handling."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import Client

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.server import mcp
from intervals_icu_mcp.strava_client import StravaAPIError
from intervals_icu_mcp.tools import strava as strava_tools
from intervals_icu_mcp.tools.strava import (
    _format_effort,
    _format_power,
    _format_streams,
    _stream_summary,
    get_starred_segments_in_activity,
)

EFFORT_ID = 9007199254740993


def test_device_power_is_exposed_as_measured_only():
    result = _format_power(
        {
            "device_watts": True,
            "average_watts": 242.0,
        }
    )

    assert result["power_source"] == "device"
    assert result["measured_watts"] == 242.0
    assert result["estimated_watts"] is None


def test_strava_estimated_power_is_never_exposed_as_measured():
    result = _format_power(
        {
            "device_watts": False,
            "average_watts": 185.8,
        }
    )

    assert result["power_source"] == "strava_estimate"
    assert result["measured_watts"] is None
    assert result["estimated_watts"] == 185.8


def test_missing_power_returns_no_power_source():
    result = _format_power(
        {
            "device_watts": False,
            "average_watts": None,
        }
    )

    assert result == {
        "power_source": None,
        "measured_watts": None,
        "estimated_watts": None,
    }


def test_large_strava_ids_are_formatted_as_strings_without_precision_loss():
    result = _format_effort(
        {
            "id": EFFORT_ID,
            "activity": {"id": 1234567890},
            "device_watts": False,
            "average_watts": None,
        }
    )

    assert result["effort_id"] == "9007199254740993"
    assert isinstance(result["effort_id"], str)

    assert result["activity_id"] == "1234567890"
    assert isinstance(result["activity_id"], str)


def _streams(size: int = 2240):
    time_data = list(range(size))
    distance_data = [round(index * (10035.0 / (size - 1)), 3) for index in range(size)]

    return {
        "time": {
            "data": time_data,
            "original_size": size,
            "resolution": "high",
            "series_type": "distance",
        },
        "distance": {
            "data": distance_data,
            "original_size": size,
            "resolution": "high",
            "series_type": "distance",
        },
        "heartrate": {
            "data": [170 + (index % 5) for index in range(size)],
            "original_size": size,
            "resolution": "high",
            "series_type": "distance",
        },
        "cadence": {
            "data": [0 if index % 50 == 0 else 78 for index in range(size)],
            "original_size": size,
            "resolution": "high",
            "series_type": "distance",
        },
    }


def test_stream_summary_uses_full_resolution_before_downsampling():
    streams = _streams()

    full_summary = _stream_summary(streams)

    formatted, downsampled = _format_streams(
        streams,
        max_points=500,
    )

    # Downsampling must not mutate or change the source streams used
    # to calculate the summary.
    after_summary = _stream_summary(streams)

    assert full_summary == after_summary
    assert full_summary["sample_count"] == 2240
    assert full_summary["streams_aligned"] is True

    assert downsampled is True

    assert formatted["time"]["returned_points"] == 500
    assert formatted["time"]["original_size"] == 2240

    assert formatted["time"]["data"][0] == streams["time"]["data"][0]
    assert formatted["time"]["data"][-1] == streams["time"]["data"][-1]

    assert formatted["distance"]["data"][0] == streams["distance"]["data"][0]
    assert formatted["distance"]["data"][-1] == streams["distance"]["data"][-1]


def test_full_resolution_streams_are_preserved_when_max_points_is_none():
    streams = _streams()

    formatted, downsampled = _format_streams(
        streams,
        max_points=None,
    )

    assert downsampled is False
    assert formatted["time"]["returned_points"] == 2240
    assert formatted["time"]["data"] == streams["time"]["data"]


class _FakeStravaClient:
    def __init__(
        self,
        *,
        activity: dict | None = None,
        starred: list[dict] | None = None,
        error: StravaAPIError | None = None,
    ) -> None:
        self.configured = True
        self.activity = activity or {}
        self.starred = starred or []
        self.error = error
        self.activity_calls: list[tuple[str, bool]] = []
        self.starred_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get_activity(
        self,
        activity_id: str,
        *,
        include_all_efforts: bool,
    ) -> dict:
        self.activity_calls.append((activity_id, include_all_efforts))
        if self.error is not None:
            raise self.error
        return self.activity

    async def get_starred_segments(self) -> list[dict]:
        self.starred_calls += 1
        return self.starred


def _ctx() -> Mock:
    ctx = Mock()
    ctx.get_state = AsyncMock(return_value=ICUConfig())
    return ctx


async def _call_activity_tool(monkeypatch, fake: _FakeStravaClient) -> dict:
    monkeypatch.setattr(strava_tools, "StravaClient", lambda _config: fake)
    result = await get_starred_segments_in_activity(
        strava_activity_id="9007199254740997",
        ctx=_ctx(),
    )
    return json.loads(result)


def _effort(
    effort_id: int,
    segment_id: int,
    *,
    start_index: int,
    end_index: int,
    device_watts: bool | None = None,
    average_watts: float | None = None,
    include_optional_metrics: bool = True,
) -> dict:
    result = {
        "id": effort_id,
        "name": f"Segment {segment_id}",
        "activity": {"id": 9007199254740997},
        "segment": {
            "id": segment_id,
            "name": f"Segment {segment_id}",
            "distance": 1234.5,
            "average_grade": 5.6,
            "climb_category": 2,
            "elevation_low": 100.0,
            "elevation_high": 169.0,
        },
        "start_date": "2026-08-15T07:30:00Z",
        "start_date_local": "2026-08-15T09:30:00+02:00",
        "elapsed_time": 240,
        "moving_time": 230,
        "distance": 1234.5,
        "start_index": start_index,
        "end_index": end_index,
        "pr_rank": None,
        "achievements": [],
        "device_watts": device_watts,
        "average_watts": average_watts,
    }
    if include_optional_metrics:
        result.update(
            {
                "average_heartrate": 165.0,
                "max_heartrate": 177.0,
                "average_cadence": 82.0,
            }
        )
    return result


async def test_activity_with_no_ridden_starred_segments_returns_empty(monkeypatch):
    fake = _FakeStravaClient(
        activity={"segment_efforts": [_effort(10, 100, start_index=1, end_index=2)]},
        starred=[{"id": 200}],
    )

    response = await _call_activity_tool(monkeypatch, fake)

    assert response["data"] == {
        "source": "strava",
        "activity_id": "9007199254740997",
        "starred_segment_count": 0,
        "effort_count": 0,
        "efforts": [],
    }
    assert fake.activity_calls == [("9007199254740997", True)]
    assert fake.starred_calls == 1


@pytest.mark.parametrize("activity", [{}, {"segment_efforts": None}])
async def test_activity_without_segment_efforts_is_success(monkeypatch, activity):
    fake = _FakeStravaClient(activity=activity, starred=[{"id": 100}])

    response = await _call_activity_tool(monkeypatch, fake)

    assert response["data"]["starred_segment_count"] == 0
    assert response["data"]["effort_count"] == 0
    assert response["data"]["efforts"] == []


async def test_activity_with_one_starred_effort(monkeypatch):
    fake = _FakeStravaClient(
        activity={
            "segment_efforts": [
                _effort(
                    9007199254740993,
                    9007199254740995,
                    start_index=10,
                    end_index=20,
                    include_optional_metrics=False,
                )
            ]
        },
        starred=[{"id": 9007199254740995}],
    )

    response = await _call_activity_tool(monkeypatch, fake)
    effort = response["data"]["efforts"][0]

    assert response["data"]["starred_segment_count"] == 1
    assert response["data"]["effort_count"] == 1
    assert effort["segment_id"] == "9007199254740995"
    assert effort["effort_id"] == "9007199254740993"
    assert effort["activity_id"] == "9007199254740997"
    assert effort["average_heartrate_bpm"] is None
    assert effort["max_heartrate_bpm"] is None
    assert effort["average_cadence_rpm"] is None
    assert effort["power_source"] is None
    assert effort["measured_watts"] is None
    assert effort["estimated_watts"] is None


async def test_multiple_starred_efforts_are_complete_and_deterministic(monkeypatch):
    non_starred = _effort(99, 999, start_index=1, end_index=2)
    estimated = _effort(
        30,
        300,
        start_index=50,
        end_index=60,
        device_watts=False,
        average_watts=188.5,
    )
    device = _effort(
        20,
        200,
        start_index=30,
        end_index=40,
        device_watts=True,
        average_watts=251.0,
    )
    repeated = _effort(10, 200, start_index=10, end_index=20)
    fake = _FakeStravaClient(
        activity={"segment_efforts": [estimated, non_starred, device, repeated]},
        starred=[{"id": 300}, {"id": 200}],
    )

    response = await _call_activity_tool(monkeypatch, fake)
    data = response["data"]
    efforts = data["efforts"]

    assert data["starred_segment_count"] == 2
    assert data["effort_count"] == 3
    assert [effort["effort_id"] for effort in efforts] == ["10", "20", "30"]
    assert [effort["segment_id"] for effort in efforts] == ["200", "200", "300"]
    assert efforts[1]["power_source"] == "device"
    assert efforts[1]["measured_watts"] == 251.0
    assert efforts[1]["estimated_watts"] is None
    assert efforts[2]["power_source"] == "strava_estimate"
    assert efforts[2]["measured_watts"] is None
    assert efforts[2]["estimated_watts"] == 188.5
    assert efforts[0]["segment"]["average_grade_pct"] == 5.6
    assert response["metadata"]["ids_are_strings"] is True
    assert response["metadata"]["read_only"] is True


@pytest.mark.parametrize(
    ("status_code", "message"),
    [(401, "Authorization Error"), (404, "Record Not Found")],
)
async def test_activity_api_errors_use_strava_error_envelope(
    monkeypatch,
    status_code,
    message,
):
    fake = _FakeStravaClient(
        error=StravaAPIError(
            message,
            status_code=status_code,
            endpoint="/activities/9007199254740997",
        )
    )

    response = await _call_activity_tool(monkeypatch, fake)

    assert response["error"]["type"] == "strava_api_error"
    assert response["error"]["message"] == message
    assert fake.starred_calls == 0


async def test_all_strava_tools_are_registered_read_only():
    expected = {
        "strava_get_starred_segments",
        "strava_get_starred_segments_in_activity",
        "strava_get_segment",
        "strava_get_segment_efforts",
        "strava_get_segment_effort_streams",
    }

    async with Client(mcp) as client:
        tools = await client.list_tools()

    by_name = {tool.name: tool for tool in tools}

    assert expected <= set(by_name)

    for name in expected:
        annotations = by_name[name].annotations
        assert annotations is not None

        if hasattr(annotations, "model_dump"):
            values = annotations.model_dump(
                by_alias=True,
                exclude_none=True,
            )
        else:
            values = dict(annotations)

        assert values["readOnlyHint"] is True
        assert values["destructiveHint"] is False
