"""Deterministic planning primitives for the integrated cycling coach."""

from __future__ import annotations

from dataclasses import dataclass

from .cycling_profiles import (
    CyclingTrainingProfileName,
    CyclingTrainingProfileResolution,
    resolve_cycling_training_profile,
)


@dataclass(frozen=True)
class CyclingCoachSessionPlan:
    """Auditable plan produced before route or athlete-data calls."""

    profile: CyclingTrainingProfileResolution
    start_location: str
    target_distance_km: float
    available_time_minutes: float
    target_duration_minutes: float
    duration_source: str
    departure_time: str | None
    include_gpx: bool
    rationale: tuple[str, ...]


def plan_cycling_coach_session(
    profile: CyclingTrainingProfileName,
    *,
    start_location: str,
    target_distance_km: float,
    available_time_minutes: float,
    target_duration_minutes: float | None = None,
    work_duration_minutes: float | None = None,
    departure_time: str | None = None,
    include_gpx: bool = True,
) -> CyclingCoachSessionPlan:
    """Resolve explicit user constraints without external calls."""

    location = start_location.strip()
    if not location:
        raise ValueError("start_location must not be empty")
    if target_distance_km <= 0:
        raise ValueError("target_distance_km must be greater than zero")
    if available_time_minutes <= 0:
        raise ValueError("available_time_minutes must be greater than zero")
    if target_duration_minutes is not None and target_duration_minutes <= 0:
        raise ValueError("target_duration_minutes must be greater than zero")
    if (
        target_duration_minutes is not None
        and target_duration_minutes > available_time_minutes
    ):
        raise ValueError(
            "target_duration_minutes must not exceed available_time_minutes"
        )

    profile_resolution = resolve_cycling_training_profile(
        profile,
        work_duration_minutes=work_duration_minutes,
    )
    route_duration = (
        target_duration_minutes
        if target_duration_minutes is not None
        else available_time_minutes
    )
    duration_source = (
        "explicit_target"
        if target_duration_minutes is not None
        else "available_time"
    )

    return CyclingCoachSessionPlan(
        profile=profile_resolution,
        start_location=location,
        target_distance_km=target_distance_km,
        available_time_minutes=available_time_minutes,
        target_duration_minutes=route_duration,
        duration_source=duration_source,
        departure_time=departure_time,
        include_gpx=include_gpx,
        rationale=(
            "Uses the explicitly selected training profile.",
            (
                "Uses the explicit route-duration target."
                if target_duration_minutes is not None
                else "Uses all declared available time as the route target."
            ),
            "Adds no implicit time buffer or athlete-readiness adjustment.",
        ),
    )
