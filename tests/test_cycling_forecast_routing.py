"""Integration tests for optional cycling-route forecast context."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.tools.routing import (
    CyclingRouteCandidateSearchResult,
    ResolvedLocation,
)


def _config() -> ICUConfig:
    return ICUConfig(openrouteservice_api_key="test-key")


def _origin() -> ResolvedLocation:
    return ResolvedLocation(
        input_value="Start",
        source="coordinates",
        label="Start",
        original_longitude=2.63,
        original_latitude=39.59,
        longitude=2.63,
        latitude=39.59,
        snapped_distance_m=0.0,
    )


class _FakeClient:
    def __init__(self, _config: ICUConfig) -> None:
        pass

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        return None


async def test_departure_time_is_validated_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        departure_time="2026-08-13T08:00:00",
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "must include a UTC offset" in response["error"]["message"]


async def test_route_adds_optional_forecast_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    origin = _origin()
    route = SimpleNamespace(duration_s=5400.0)
    analysis = SimpleNamespace(candidate=SimpleNamespace(route=route))
    search = AsyncMock(
        return_value=CyclingRouteCandidateSearchResult(
            ranked=(analysis,),
            candidates_generated=2,
            candidates_after_distance_filter=1,
            candidates_after_duration_filter=1,
            candidates_after_deduplication=1,
        )
    )
    forecast_payload = {
        "timezone": "Europe/Madrid",
        "hourly": {
            "time": [
                "2026-08-13T08:00",
                "2026-08-13T09:00",
                "2026-08-13T10:00",
            ],
            "temperature_2m": [21.0, 23.0, 25.0],
            "apparent_temperature": [21.5, 24.0, 26.0],
            "precipitation_probability": [10, 70, 20],
            "precipitation": [0.0, 1.2, 0.1],
            "weather_code": [1, 95, 61],
            "wind_speed_10m": [8.0, 12.0, 10.0],
            "wind_direction_10m": [100, 110, 120],
            "wind_gusts_10m": [12.0, 22.0, 15.0],
        },
        "daily": {
            "time": ["2026-08-13"],
            "sunrise": ["2026-08-13T07:02"],
            "sunset": ["2026-08-13T20:48"],
        },
    }
    weather_forecast = AsyncMock(return_value=forecast_payload)

    class FakeWeatherClient(_FakeClient):
        forecast = weather_forecast

    monkeypatch.setattr(routing, "OpenRouteServiceClient", _FakeClient)
    monkeypatch.setattr(routing, "OpenMeteoClient", FakeWeatherClient)
    monkeypatch.setattr(
        routing,
        "resolve_location",
        AsyncMock(return_value=origin),
    )
    monkeypatch.setattr(
        routing,
        "find_cycling_training_route_candidates",
        search,
    )
    monkeypatch.setattr(
        routing,
        "serialize_cycling_route_candidate_analysis",
        lambda _analysis: {"candidate": "first"},
    )
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        candidate_count=2,
        departure_time="2026-08-13T08:30:00+02:00",
        ctx=ctx,
    )

    response = json.loads(result)
    external_context = response["data"]["external_context"]
    assert external_context["available"] is True
    assert external_context["estimated_finish_time"] == (
        "2026-08-13T10:00:00+02:00"
    )
    assert external_context["warnings"] == [
        "precipitation_forecast",
        "thunderstorm_forecast",
    ]
    assert response["metadata"]["departure_time_requested"] == (
        "2026-08-13T08:30:00+02:00"
    )
    weather_forecast.assert_awaited_once()
    assert weather_forecast.await_args.args == (39.59, 2.63)
    assert weather_forecast.await_args.kwargs["start_date"].isoformat() == (
        "2026-08-12"
    )
    assert weather_forecast.await_args.kwargs["end_date"].isoformat() == (
        "2026-08-14"
    )


async def test_route_without_departure_does_not_call_weather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        candidate=SimpleNamespace(route=SimpleNamespace(duration_s=3600.0))
    )
    monkeypatch.setattr(routing, "OpenRouteServiceClient", _FakeClient)
    monkeypatch.setattr(
        routing, "OpenMeteoClient", lambda _config: pytest.fail("unexpected")
    )
    monkeypatch.setattr(
        routing, "resolve_location", AsyncMock(return_value=_origin())
    )
    monkeypatch.setattr(
        routing,
        "find_cycling_training_route_candidates",
        AsyncMock(
            return_value=CyclingRouteCandidateSearchResult(
                ranked=(analysis,),
                candidates_generated=2,
                candidates_after_distance_filter=1,
                candidates_after_duration_filter=1,
                candidates_after_deduplication=1,
            )
        ),
    )
    monkeypatch.setattr(
        routing,
        "serialize_cycling_route_candidate_analysis",
        lambda _analysis: {"candidate": "first"},
    )
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))
    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        candidate_count=2,
        ctx=ctx,
    )
    assert json.loads(result)["data"]["external_context"] is None
