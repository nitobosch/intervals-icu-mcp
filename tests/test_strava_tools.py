"""Tests for Strava MCP formatting and stream handling."""

from fastmcp import Client

from intervals_icu_mcp.server import mcp
from intervals_icu_mcp.tools.strava import (
    _format_effort,
    _format_power,
    _format_streams,
    _stream_summary,
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


async def test_all_strava_tools_are_registered_read_only():
    expected = {
        "strava_get_starred_segments",
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
