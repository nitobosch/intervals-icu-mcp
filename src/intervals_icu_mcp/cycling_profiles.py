"""High-level cycling training profiles.

Profiles translate a training objective into existing route-search parameters.
They deliberately remain separate from low-level routing and ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CyclingTrainingProfileName = Literal[
    "steady_climb",
    "sweet_spot_climb",
]


@dataclass(frozen=True)
class CyclingTrainingProfileResolution:
    """Auditable translation from one profile to route-search parameters."""

    profile: CyclingTrainingProfileName
    objective: str
    training_durations_minutes: tuple[float, ...]
    training_repetitions: int
    recovery_min_minutes: float | None
    recovery_max_minutes: float | None
    ranking_intent: str
    hard_constraints_applied: tuple[str, ...]
    rationale: tuple[str, ...]


def resolve_cycling_training_profile(
    profile: CyclingTrainingProfileName,
    *,
    work_duration_minutes: float | None = None,
) -> CyclingTrainingProfileResolution:
    """Resolve a supported profile without calling routing or external APIs."""

    if profile not in ("steady_climb", "sweet_spot_climb"):
        raise ValueError(f"unsupported cycling training profile: {profile}")
    if work_duration_minutes is not None and work_duration_minutes <= 0:
        raise ValueError("work_duration_minutes must be greater than zero")

    if profile == "sweet_spot_climb":
        work_duration = (
            work_duration_minutes
            if work_duration_minutes is not None
            else 12.0
        )
        duration_rationale = (
            "Uses the explicitly requested duration for each work block."
            if work_duration_minutes is not None
            else "Uses the profile default of three 12 minute work blocks."
        )
        return CyclingTrainingProfileResolution(
            profile="sweet_spot_climb",
            objective=(
                "Find three sustained climbing work blocks with bounded "
                "recoveries inside a complete road-cycling route."
            ),
            training_durations_minutes=(work_duration,),
            training_repetitions=3,
            recovery_min_minutes=5.0,
            recovery_max_minutes=8.0,
            ranking_intent=(
                "Use the existing deterministic multi-block ranking, which "
                "considers the weakest work block first, then consistency, "
                "climbing balance, interruptions and recovery quality."
            ),
            hard_constraints_applied=(),
            rationale=(
                duration_rationale,
                "Uses three repetitions with explicit 5 to 8 minute recoveries.",
                (
                    "Adds no hidden surface, gradient, elevation, power-zone "
                    "or interruption thresholds."
                ),
            ),
        )

    durations = (
        (work_duration_minutes,)
        if work_duration_minutes is not None
        else (20.0, 30.0, 40.0)
    )
    duration_rationale = (
        "Uses the explicitly requested continuous work duration."
        if work_duration_minutes is not None
        else (
            "Compares the existing 20, 30 and 40 minute continuous-window "
            "defaults."
        )
    )

    return CyclingTrainingProfileResolution(
        profile="steady_climb",
        objective=(
            "Find one continuous, steady climbing window inside a complete "
            "road-cycling route."
        ),
        training_durations_minutes=durations,
        training_repetitions=1,
        recovery_min_minutes=None,
        recovery_max_minutes=None,
        ranking_intent=(
            "Use the existing deterministic climbing-oriented window ranking, "
            "then warmup, cooldown, route quality and target fit."
        ),
        hard_constraints_applied=(),
        rationale=(
            duration_rationale,
            "Keeps the session continuous by using exactly one work block.",
            (
                "Adds no hidden surface, gradient, elevation or interruption "
                "thresholds."
            ),
        ),
    )
