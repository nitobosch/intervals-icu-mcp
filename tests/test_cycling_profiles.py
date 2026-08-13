"""Tests for high-level cycling training profile resolution."""

import pytest

from intervals_icu_mcp.cycling_profiles import (
    resolve_cycling_training_profile,
)


def test_steady_climb_uses_auditable_defaults_without_hard_constraints() -> None:
    resolution = resolve_cycling_training_profile("steady_climb")

    assert resolution.profile == "steady_climb"
    assert resolution.training_durations_minutes == (20.0, 30.0, 40.0)
    assert resolution.training_repetitions == 1
    assert resolution.recovery_min_minutes is None
    assert resolution.recovery_max_minutes is None
    assert resolution.hard_constraints_applied == ()
    assert "deterministic climbing-oriented" in resolution.ranking_intent
    assert any(
        "no hidden" in rationale
        for rationale in resolution.rationale
    )


def test_steady_climb_accepts_explicit_work_duration() -> None:
    resolution = resolve_cycling_training_profile(
        "steady_climb",
        work_duration_minutes=35.0,
    )

    assert resolution.training_durations_minutes == (35.0,)
    assert resolution.rationale[0] == (
        "Uses the explicitly requested continuous work duration."
    )


def test_steady_climb_rejects_non_positive_work_duration() -> None:
    with pytest.raises(
        ValueError,
        match="work_duration_minutes must be greater than zero",
    ):
        resolve_cycling_training_profile(
            "steady_climb",
            work_duration_minutes=0.0,
        )


def test_sweet_spot_climb_resolves_auditable_multiblock_defaults() -> None:
    resolution = resolve_cycling_training_profile("sweet_spot_climb")

    assert resolution.profile == "sweet_spot_climb"
    assert resolution.training_durations_minutes == (12.0,)
    assert resolution.training_repetitions == 3
    assert resolution.recovery_min_minutes == 5.0
    assert resolution.recovery_max_minutes == 8.0
    assert resolution.hard_constraints_applied == ()
    assert "weakest work block first" in resolution.ranking_intent
    assert resolution.rationale == (
        "Uses the profile default of three 12 minute work blocks.",
        "Uses three repetitions with explicit 5 to 8 minute recoveries.",
        (
            "Adds no hidden surface, gradient, elevation, power-zone "
            "or interruption thresholds."
        ),
    )


def test_sweet_spot_climb_accepts_explicit_work_block_duration() -> None:
    resolution = resolve_cycling_training_profile(
        "sweet_spot_climb",
        work_duration_minutes=15.0,
    )

    assert resolution.training_durations_minutes == (15.0,)
    assert resolution.training_repetitions == 3
    assert resolution.rationale[0] == (
        "Uses the explicitly requested duration for each work block."
    )


def test_sweet_spot_climb_rejects_non_positive_work_duration() -> None:
    with pytest.raises(
        ValueError,
        match="work_duration_minutes must be greater than zero",
    ):
        resolve_cycling_training_profile(
            "sweet_spot_climb",
            work_duration_minutes=-1.0,
        )


def test_profile_resolver_rejects_unsupported_runtime_value() -> None:
    with pytest.raises(ValueError, match="unsupported cycling training profile"):
        resolve_cycling_training_profile("recovery_ride")  # type: ignore[arg-type]
