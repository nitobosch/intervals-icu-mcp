"""Direct read-only Strava tools."""

from __future__ import annotations

from statistics import mean
from typing import Annotated, Any, cast

from fastmcp import Context

from ..auth import ICUConfig
from ..response_builder import ResponseBuilder
from ..strava_client import StravaAPIError, StravaClient

DEFAULT_STREAM_TYPES = [
    "time",
    "distance",
    "altitude",
    "heartrate",
    "cadence",
    "velocity_smooth",
]

ALLOWED_STREAM_TYPES = set(DEFAULT_STREAM_TYPES)


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Return a dynamically decoded JSON object with explicit typing."""

    if not isinstance(value, dict):
        return None

    return cast(dict[str, Any], value)


def _as_list(value: Any) -> list[Any] | None:
    """Return a dynamically decoded JSON array with explicit typing."""

    if not isinstance(value, list):
        return None

    return cast(list[Any], value)


def _strava_id(value: Any) -> str | None:
    """Return a Strava identifier as a precision-safe string."""

    if value is None:
        return None
    return str(value)


def _format_power(effort: dict[str, Any]) -> dict[str, Any]:
    """Expose watts without confusing Strava estimates with measured power."""

    average_watts = effort.get("average_watts")
    device_watts = effort.get("device_watts")

    if average_watts is None:
        return {
            "power_source": None,
            "measured_watts": None,
            "estimated_watts": None,
        }

    if device_watts is True:
        return {
            "power_source": "device",
            "measured_watts": average_watts,
            "estimated_watts": None,
        }

    if device_watts is False:
        return {
            "power_source": "strava_estimate",
            "measured_watts": None,
            "estimated_watts": average_watts,
        }

    # Defensive case for old or incomplete Strava records.
    return {
        "power_source": "unknown",
        "measured_watts": None,
        "estimated_watts": None,
        "unclassified_watts": average_watts,
    }


def _format_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Format compact Strava segment metadata."""

    result: dict[str, Any] = {
        "id": _strava_id(segment.get("id")),
        "name": segment.get("name"),
        "activity_type": segment.get("activity_type"),
        "distance_m": segment.get("distance"),
        "average_grade_pct": segment.get("average_grade"),
        "maximum_grade_pct": segment.get("maximum_grade"),
        "climb_category": segment.get("climb_category"),
        "starred": segment.get("starred"),
    }

    if segment.get("elevation_low") is not None:
        result["elevation_low_m"] = segment["elevation_low"]

    if segment.get("elevation_high") is not None:
        result["elevation_high_m"] = segment["elevation_high"]

    location = {key: segment.get(key) for key in ("city", "state", "country") if segment.get(key)}

    if location:
        result["location"] = location

    return result


def _format_segment_detail(segment: dict[str, Any]) -> dict[str, Any]:
    """Format detailed segment metadata and athlete-specific statistics."""

    result = _format_segment(segment)

    if segment.get("start_latlng") is not None:
        result["start_latlng"] = segment["start_latlng"]

    if segment.get("end_latlng") is not None:
        result["end_latlng"] = segment["end_latlng"]

    athlete_stats = _as_dict(segment.get("athlete_segment_stats")) or {}

    result["athlete_segment_stats"] = {
        "pr_elapsed_seconds": athlete_stats.get("pr_elapsed_time"),
        "pr_date": athlete_stats.get("pr_date"),
        "effort_count": athlete_stats.get("effort_count"),
    }

    return result


def _format_effort(effort: dict[str, Any]) -> dict[str, Any]:
    """Format a segment effort for LLM-facing use."""

    activity = _as_dict(effort.get("activity")) or {}
    achievements = _as_list(effort.get("achievements")) or []

    result: dict[str, Any] = {
        "effort_id": _strava_id(effort.get("id")),
        "activity_id": _strava_id(activity.get("id")),
        "name": effort.get("name"),
        "start_date": effort.get("start_date"),
        "start_date_local": effort.get("start_date_local"),
        "elapsed_seconds": effort.get("elapsed_time"),
        "moving_seconds": effort.get("moving_time"),
        "distance_m": effort.get("distance"),
        "start_index": effort.get("start_index"),
        "end_index": effort.get("end_index"),
        "average_heartrate_bpm": effort.get("average_heartrate"),
        "max_heartrate_bpm": effort.get("max_heartrate"),
        "average_cadence_rpm": effort.get("average_cadence"),
        "pr_rank": effort.get("pr_rank"),
        "achievement_count": len(achievements),
    }

    result.update(_format_power(effort))

    return result


def _format_activity_starred_effort(
    effort: dict[str, Any],
    *,
    strava_activity_id: str,
) -> dict[str, Any] | None:
    """Format one activity effort and retain its official segment identity."""

    segment = _as_dict(effort.get("segment"))

    if segment is None:
        return None

    segment_id = _strava_id(segment.get("id"))

    if segment_id is None:
        return None

    result = _format_effort(effort)
    if result["name"] is None:
        result["name"] = segment.get("name")

    result["segment_id"] = segment_id

    if result["activity_id"] is None:
        result["activity_id"] = strava_activity_id

    result["segment"] = _format_segment(segment)
    return result


def _effort_order_key(effort: dict[str, Any]) -> tuple[float, float, str, str]:
    """Return a deterministic activity-order key for a formatted effort."""

    def index_value(value: Any) -> float:
        if isinstance(value, int | float):
            return float(value)
        return float("inf")

    return (
        index_value(effort.get("start_index")),
        index_value(effort.get("end_index")),
        str(effort.get("segment_id") or ""),
        str(effort.get("effort_id") or ""),
    )


def _numeric_stream(streams: dict[str, Any], name: str) -> list[float]:
    """Return numeric values for one stream."""

    stream = _as_dict(streams.get(name))

    if stream is None:
        return []

    values = _as_list(stream.get("data"))

    if values is None:
        return []

    return [float(value) for value in values if isinstance(value, int | float)]


def _stream_summary(streams: dict[str, Any]) -> dict[str, Any]:
    """Calculate summary values from the complete high-resolution streams."""

    summary: dict[str, Any] = {}

    sizes: dict[str, int] = {}

    for name, stream_raw in streams.items():
        stream = _as_dict(stream_raw)

        if stream is None:
            continue

        data = _as_list(stream.get("data")) or []
        sizes[name] = len(data)

    if sizes:
        summary["sample_count"] = max(sizes.values())
        summary["stream_sizes"] = sizes
        summary["streams_aligned"] = len(set(sizes.values())) <= 1

    time_values = _numeric_stream(streams, "time")
    distance_values = _numeric_stream(streams, "distance")
    hr_values = _numeric_stream(streams, "heartrate")
    cadence_values = _numeric_stream(streams, "cadence")
    altitude_values = _numeric_stream(streams, "altitude")
    speed_values = _numeric_stream(streams, "velocity_smooth")

    if len(time_values) >= 2:
        summary["elapsed_stream_seconds"] = round(
            time_values[-1] - time_values[0],
            1,
        )

    if len(distance_values) >= 2:
        summary["recorded_stream_distance_m"] = round(
            distance_values[-1] - distance_values[0],
            1,
        )

    if hr_values:
        summary["average_heartrate_bpm"] = round(mean(hr_values), 1)
        summary["max_heartrate_bpm"] = round(max(hr_values), 1)

    if cadence_values:
        summary["average_cadence_rpm"] = round(mean(cadence_values), 1)

        nonzero_cadence = [value for value in cadence_values if value > 0]

        if nonzero_cadence:
            summary["average_cadence_when_pedaling_rpm"] = round(
                mean(nonzero_cadence),
                1,
            )

    if altitude_values:
        summary["minimum_altitude_m"] = round(min(altitude_values), 1)
        summary["maximum_altitude_m"] = round(max(altitude_values), 1)

    if speed_values:
        summary["average_velocity_smooth_mps"] = round(
            mean(speed_values),
            2,
        )
        summary["maximum_velocity_smooth_mps"] = round(
            max(speed_values),
            2,
        )

    return summary


def _sample_indices(
    size: int,
    max_points: int | None,
) -> list[int]:
    """Generate aligned sample indices while always preserving endpoints."""

    if max_points is None or size <= max_points:
        return list(range(size))

    if max_points < 2:
        return [0]

    step = (size - 1) / (max_points - 1)

    indices = [round(index * step) for index in range(max_points)]

    # Rounding can theoretically produce duplicates.
    return list(dict.fromkeys(indices))


def _format_streams(
    streams: dict[str, Any],
    max_points: int | None,
) -> tuple[dict[str, Any], bool]:
    """Format/downsample aligned streams after summary calculation."""

    sizes: list[int] = []

    for stream_raw in streams.values():
        stream = _as_dict(stream_raw)

        if stream is None:
            continue

        data = _as_list(stream.get("data")) or []
        sizes.append(len(data))

    reference_size = max(sizes, default=0)
    indices = _sample_indices(reference_size, max_points)

    formatted: dict[str, Any] = {}

    for name, stream_raw in streams.items():
        stream = _as_dict(stream_raw)

        if stream is None:
            continue

        data = _as_list(stream.get("data")) or []

        if len(data) == reference_size:
            returned_data = [data[index] for index in indices if index < len(data)]
        else:
            # Defensive case: preserve the entire stream if alignment differs.
            returned_data = data

        formatted[name] = {
            "data": returned_data,
            "returned_points": len(returned_data),
            "original_size": stream.get(
                "original_size",
                len(data),
            ),
            "resolution": stream.get("resolution"),
            "series_type": stream.get("series_type"),
        }

    downsampled = max_points is not None and reference_size > max_points

    return formatted, downsampled


async def get_starred_segments(
    limit: Annotated[
        int,
        "Maximum number of starred Strava segments to return (1-1000).",
    ] = 500,
    ctx: Context | None = None,
) -> str:
    """Get all starred/favourite Strava segments for the authenticated athlete."""

    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    if limit < 1 or limit > 1000:
        return ResponseBuilder.build_error_response(
            "limit must be between 1 and 1000.",
            error_type="validation_error",
        )

    try:
        async with StravaClient(config) as client:
            if not client.configured:
                return ResponseBuilder.build_error_response(
                    "Direct Strava integration is not configured.",
                    error_type="configuration_error",
                )

            segments = await client.get_starred_segments(limit=limit)

            return ResponseBuilder.build_response(
                data={
                    "source": "strava",
                    "count": len(segments),
                    "segments": [_format_segment(segment) for segment in segments],
                },
                metadata={
                    "read_only": True,
                    "ids_are_strings": True,
                    "pagination": "automatic",
                },
                query_type="strava_starred_segments",
            )

    except StravaAPIError as e:
        return ResponseBuilder.build_error_response(
            e.message,
            error_type="strava_api_error",
        )
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected Strava integration error: {e}",
            error_type="internal_error",
        )


async def get_starred_segments_in_activity(
    strava_activity_id: Annotated[
        str,
        "Strava activity ID (not an Intervals.icu activity ID).",
    ],
    ctx: Context | None = None,
) -> str:
    """Return starred Strava segment efforts present in one activity.

    Matching uses only official Strava segment IDs from the detailed activity
    and the authenticated athlete's starred segment list. No GPS matching or
    per-segment effort queries are performed.
    """

    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")
    activity_id = strava_activity_id.strip()

    if not activity_id:
        return ResponseBuilder.build_error_response(
            "strava_activity_id must not be empty.",
            error_type="validation_error",
        )

    try:
        async with StravaClient(config) as client:
            if not client.configured:
                return ResponseBuilder.build_error_response(
                    "Direct Strava integration is not configured.",
                    error_type="configuration_error",
                )

            activity = await client.get_activity(
                activity_id,
                include_all_efforts=True,
            )
            starred_segments = await client.get_starred_segments()

        starred_ids = {
            segment_id
            for segment in starred_segments
            if (segment_id := _strava_id(segment.get("id"))) is not None
        }
        raw_efforts = _as_list(activity.get("segment_efforts")) or []
        efforts: list[dict[str, Any]] = []

        for raw_effort in raw_efforts:
            effort = _as_dict(raw_effort)

            if effort is None:
                continue

            segment = _as_dict(effort.get("segment"))
            segment_id = (
                _strava_id(segment.get("id"))
                if segment is not None
                else None
            )

            if segment_id not in starred_ids:
                continue

            formatted = _format_activity_starred_effort(
                effort,
                strava_activity_id=activity_id,
            )

            if formatted is not None:
                efforts.append(formatted)

        efforts.sort(key=_effort_order_key)
        ridden_starred_ids = {
            str(effort["segment_id"])
            for effort in efforts
        }

        return ResponseBuilder.build_response(
            data={
                "source": "strava",
                "activity_id": activity_id,
                "starred_segment_count": len(ridden_starred_ids),
                "effort_count": len(efforts),
                "efforts": efforts,
            },
            metadata={
                "read_only": True,
                "ids_are_strings": True,
                "matching": "official_strava_segment_id_intersection",
                "power_semantics": {
                    "device": "measured by a recording device/power meter",
                    "strava_estimate": "estimated by Strava; not measured power",
                },
            },
            query_type="strava_starred_segments_in_activity",
        )

    except StravaAPIError as e:
        return ResponseBuilder.build_error_response(
            e.message,
            error_type="strava_api_error",
        )
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected Strava integration error: {e}",
            error_type="internal_error",
        )


async def get_segment(
    segment_id: Annotated[
        str,
        "Strava segment ID.",
    ],
    ctx: Context | None = None,
) -> str:
    """Get detailed metadata and personal statistics for one Strava segment."""

    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    try:
        async with StravaClient(config) as client:
            if not client.configured:
                return ResponseBuilder.build_error_response(
                    "Direct Strava integration is not configured.",
                    error_type="configuration_error",
                )

            segment = await client.get_segment(segment_id)

        return ResponseBuilder.build_response(
            data={
                "source": "strava",
                "segment": _format_segment_detail(segment),
            },
            metadata={
                "read_only": True,
                "ids_are_strings": True,
                "official_segment_distance": True,
            },
            query_type="strava_segment",
        )

    except StravaAPIError as e:
        return ResponseBuilder.build_error_response(
            e.message,
            error_type="strava_api_error",
        )
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected Strava integration error: {e}",
            error_type="internal_error",
        )


async def get_segment_efforts(
    segment_id: Annotated[
        str,
        "Strava segment ID.",
    ],
    limit: Annotated[
        int,
        "Maximum number of personal efforts to return (1-1000).",
    ] = 200,
    ctx: Context | None = None,
) -> str:
    """Get the authenticated athlete's effort history for a Strava segment.

    Device-recorded watts and Strava-estimated watts are deliberately exposed
    as different fields so an estimated value cannot be mistaken for power
    meter data.
    """

    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    if limit < 1 or limit > 1000:
        return ResponseBuilder.build_error_response(
            "limit must be between 1 and 1000.",
            error_type="validation_error",
        )

    try:
        async with StravaClient(config) as client:
            if not client.configured:
                return ResponseBuilder.build_error_response(
                    "Direct Strava integration is not configured.",
                    error_type="configuration_error",
                )

            efforts = await client.get_segment_efforts(
                segment_id,
                limit=limit,
            )

        return ResponseBuilder.build_response(
            data={
                "source": "strava",
                "segment_id": segment_id,
                "count": len(efforts),
                "efforts": [_format_effort(effort) for effort in efforts],
            },
            metadata={
                "read_only": True,
                "ids_are_strings": True,
                "power_semantics": {
                    "device": "measured by a recording device/power meter",
                    "strava_estimate": "estimated by Strava; not measured power",
                },
            },
            query_type="strava_segment_efforts",
        )

    except StravaAPIError as e:
        return ResponseBuilder.build_error_response(
            e.message,
            error_type="strava_api_error",
        )
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected Strava integration error: {e}",
            error_type="internal_error",
        )


async def get_segment_effort_streams(
    effort_id: Annotated[
        str,
        "Strava segment effort ID.",
    ],
    stream_types: Annotated[
        list[str] | None,
        (
            "Streams to return. Supported: time, distance, altitude, "
            "heartrate, cadence, velocity_smooth. Defaults to all six."
        ),
    ] = None,
    max_points: Annotated[
        int | None,
        (
            "Optional maximum number of returned points per aligned stream. "
            "Summary statistics are always calculated from the complete "
            "high-resolution Strava streams before downsampling."
        ),
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Get official high-resolution streams for one Strava segment effort."""

    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    requested_streams = DEFAULT_STREAM_TYPES if stream_types is None else stream_types

    invalid = sorted(set(requested_streams) - ALLOWED_STREAM_TYPES)

    if invalid:
        return ResponseBuilder.build_error_response(
            "Unsupported stream types: "
            + ", ".join(invalid)
            + ". Supported types: "
            + ", ".join(sorted(ALLOWED_STREAM_TYPES)),
            error_type="validation_error",
        )

    if not requested_streams:
        return ResponseBuilder.build_error_response(
            "At least one stream type is required.",
            error_type="validation_error",
        )

    if max_points is not None and not 2 <= max_points <= 10000:
        return ResponseBuilder.build_error_response(
            "max_points must be between 2 and 10000, or null for full resolution.",
            error_type="validation_error",
        )

    try:
        async with StravaClient(config) as client:
            if not client.configured:
                return ResponseBuilder.build_error_response(
                    "Direct Strava integration is not configured.",
                    error_type="configuration_error",
                )

            # Detail is useful for activity association and correct power semantics.
            effort = await client.get_segment_effort(effort_id)

            streams = await client.get_segment_effort_streams(
                effort_id,
                keys=requested_streams,
            )

        # Summary MUST be calculated before any optional downsampling.
        summary = _stream_summary(streams)

        formatted_streams, downsampled = _format_streams(
            streams,
            max_points=max_points,
        )

        return ResponseBuilder.build_response(
            data={
                "source": "strava",
                "effort": _format_effort(effort),
                "summary": summary,
                "streams": formatted_streams,
            },
            metadata={
                "read_only": True,
                "ids_are_strings": True,
                "stream_source": "official_segment_effort_streams",
                "summary_uses_full_resolution": True,
                "downsampled": downsampled,
                "requested_max_points": max_points,
            },
            query_type="strava_segment_effort_streams",
        )

    except StravaAPIError as e:
        return ResponseBuilder.build_error_response(
            e.message,
            error_type="strava_api_error",
        )
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected Strava integration error: {e}",
            error_type="internal_error",
        )
