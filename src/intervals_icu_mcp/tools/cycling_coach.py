"""Integrated cycling-coach MCP tool."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Any, Literal, cast

from fastmcp import Context

from ..cycling_coach import plan_cycling_coach_session
from ..cycling_profiles import CyclingTrainingProfileName
from ..response_builder import ResponseBuilder
from .cycling_profiles import find_profiled_cycling_training_route


async def find_cycling_coach_route(
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
    available_time_minutes: Annotated[
        float,
        "Total time available for the complete cycling session.",
    ],
    target_duration_minutes: Annotated[
        float | None,
        "Optional route target; must fit within the available time.",
    ] = None,
    work_duration_minutes: Annotated[
        float | None,
        "Optional duration of each work block; profile defaults apply if omitted.",
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
    ] = True,
    avoid_features: Annotated[
        list[Literal["ferries", "fords", "steps"]] | None,
        "Native ORS cycling features to avoid.",
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Plan an explicit session and execute it through profiled routing."""

    try:
        plan = plan_cycling_coach_session(
            profile,
            start_location=start_location,
            target_distance_km=target_distance_km,
            available_time_minutes=available_time_minutes,
            target_duration_minutes=target_duration_minutes,
            work_duration_minutes=work_duration_minutes,
            departure_time=departure_time,
            include_gpx=include_gpx,
        )
    except ValueError as exc:
        return ResponseBuilder.build_error_response(
            str(exc),
            error_type="validation_error",
        )

    assert ctx is not None
    raw_response = await find_profiled_cycling_training_route(
        profile=profile,
        start_location=plan.start_location,
        target_distance_km=plan.target_distance_km,
        work_duration_minutes=work_duration_minutes,
        target_duration_minutes=plan.target_duration_minutes,
        departure_time=plan.departure_time,
        candidate_count=candidate_count,
        include_gpx=plan.include_gpx,
        avoid_features=avoid_features,
        ctx=ctx,
    )
    response = json.loads(raw_response)
    if not isinstance(response, dict):
        return raw_response
    typed_response = cast(dict[str, Any], response)
    data = typed_response.get("data")
    if isinstance(data, dict):
        cast(dict[str, Any], data)["coach_plan"] = asdict(plan)
    return json.dumps(typed_response, ensure_ascii=False, default=str)
