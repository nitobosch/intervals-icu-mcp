"""High-level cycling-profile MCP tools."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Any, Literal, cast

from fastmcp import Context

from ..cycling_profiles import (
    CyclingTrainingProfileName,
    resolve_cycling_training_profile,
)
from ..gpx_delivery import (
    CyclingToolResponse,
    rebuild_response,
    response_text_and_resources,
)
from ..response_builder import ResponseBuilder
from .routing import find_cycling_training_route


async def find_profiled_cycling_training_route(
    profile: Annotated[
        CyclingTrainingProfileName,
        "Training objective: steady_climb or sweet_spot_climb.",
    ],
    start_location: Annotated[
        str,
        "Round-trip start as a place name or 'latitude,longitude' coordinates.",
    ],
    target_distance_km: Annotated[
        float,
        "Preferred total round-trip distance in kilometers.",
    ],
    work_duration_minutes: Annotated[
        float | None,
        "Optional duration of each work block; profile defaults apply if omitted.",
    ] = None,
    target_duration_minutes: Annotated[
        float | None,
        "Optional preferred total route duration in minutes.",
    ] = None,
    departure_time: Annotated[
        str | None,
        "Planned ISO 8601 departure with UTC offset for forecast context.",
    ] = None,
    candidate_count: Annotated[
        int,
        "Number of deterministic route candidates, from 2 to 10.",
    ] = 3,
    include_gpx: Annotated[
        bool,
        "Include the best route as base64-encoded GPX 1.1.",
    ] = False,
    avoid_features: Annotated[
        list[Literal["ferries", "fords", "steps"]] | None,
        "Native ORS cycling features to avoid.",
    ] = None,
    ctx: Context | None = None,
) -> CyclingToolResponse:
    """Resolve a training profile and delegate to the routing engine."""

    assert ctx is not None
    try:
        resolution = resolve_cycling_training_profile(
            profile,
            work_duration_minutes=work_duration_minutes,
        )
    except ValueError as exc:
        return ResponseBuilder.build_error_response(
            str(exc),
            error_type="validation_error",
        )

    raw_response = await find_cycling_training_route(
        start_location=start_location,
        target_distance_km=target_distance_km,
        target_duration_minutes=target_duration_minutes,
        include_gpx=include_gpx,
        training_durations_minutes=list(
            resolution.training_durations_minutes
        ),
        training_repetitions=resolution.training_repetitions,
        recovery_min_minutes=resolution.recovery_min_minutes,
        recovery_max_minutes=resolution.recovery_max_minutes,
        candidate_count=candidate_count,
        training_start_time_min_minutes=(
            resolution.training_start_time_min_minutes
        ),
        training_start_time_max_minutes=(
            resolution.training_start_time_max_minutes
        ),
        avoid_features=avoid_features,
        departure_time=departure_time,
        ctx=ctx,
    )
    response_text, resources = response_text_and_resources(raw_response)
    response = json.loads(response_text)
    if not isinstance(response, dict):
        return raw_response
    typed_response = cast(dict[str, Any], response)
    data = typed_response.get("data")
    if isinstance(data, dict):
        cast(dict[str, Any], data)["training_profile"] = asdict(resolution)
    return rebuild_response(
        json.dumps(typed_response, ensure_ascii=False, default=str),
        resources,
    )
