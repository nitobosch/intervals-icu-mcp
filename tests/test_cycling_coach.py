"""Tests for integrated cycling-coach planning primitives."""

import pytest

from intervals_icu_mcp.cycling_coach import plan_cycling_coach_session


def test_plan_uses_available_time_as_route_target() -> None:
    plan = plan_cycling_coach_session(
        "sweet_spot_climb",
        start_location=" Son Moix ",
        target_distance_km=60.0,
        available_time_minutes=135.0,
        departure_time="2026-08-14T08:00:00+02:00",
    )

    assert plan.start_location == "Son Moix"
    assert plan.available_time_minutes == 135.0
    assert plan.target_duration_minutes == 135.0
    assert plan.duration_source == "available_time"
    assert plan.include_gpx is True
    assert plan.profile.profile == "sweet_spot_climb"
    assert plan.profile.training_repetitions == 3
    assert plan.rationale[-1] == (
        "Adds no implicit time buffer or athlete-readiness adjustment."
    )


def test_plan_preserves_shorter_explicit_route_target() -> None:
    plan = plan_cycling_coach_session(
        "steady_climb",
        start_location="39.59,2.63",
        target_distance_km=50.0,
        available_time_minutes=150.0,
        target_duration_minutes=120.0,
        work_duration_minutes=35.0,
        include_gpx=False,
    )

    assert plan.target_duration_minutes == 120.0
    assert plan.duration_source == "explicit_target"
    assert plan.include_gpx is False
    assert plan.profile.training_durations_minutes == (35.0,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "start_location": " ",
                "target_distance_km": 50.0,
                "available_time_minutes": 120.0,
            },
            "start_location must not be empty",
        ),
        (
            {
                "start_location": "Start",
                "target_distance_km": 0.0,
                "available_time_minutes": 120.0,
            },
            "target_distance_km must be greater than zero",
        ),
        (
            {
                "start_location": "Start",
                "target_distance_km": 50.0,
                "available_time_minutes": 0.0,
            },
            "available_time_minutes must be greater than zero",
        ),
        (
            {
                "start_location": "Start",
                "target_distance_km": 50.0,
                "available_time_minutes": 120.0,
                "target_duration_minutes": 130.0,
            },
            "must not exceed available_time_minutes",
        ),
    ],
)
def test_plan_validates_explicit_constraints(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_cycling_coach_session(
            "steady_climb",
            **kwargs,  # type: ignore[arg-type]
        )
