"""Contract tests for generic weekly sport TARGET upserts."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import Response

from intervals_icu_mcp.models import SportSettings
from intervals_icu_mcp.tools.weekly_targets import (
    _target_values_to_api,
    set_weekly_sport_target,
)

WEEK = "2026-08-24"
SPORT_SETTINGS = [
    {"id": 1, "types": ["Ride", "VirtualRide", "GravelRide"]},
    {"id": 2, "types": ["Run", "TrailRun"]},
    {"id": 3, "types": ["Swim", "OpenWaterSwim"]},
]


def _ctx(mock_config):
    ctx = MagicMock()
    ctx.get_state = AsyncMock(return_value=mock_config)
    return ctx


def _target(
    event_id: int = 9001,
    *,
    sport: str = "Ride",
    for_week: bool = True,
    load: int | None = 280,
    time: int | None = 18000,
    distance: float | None = 120000,
) -> dict:
    return {
        "id": event_id,
        "start_date_local": f"{WEEK}T00:00:00",
        "category": "TARGET",
        "type": sport,
        "for_week": for_week,
        "load_target": load,
        "time_target": time,
        "distance_target": distance,
        "name": "Must survive",
        "description": "Do not overwrite",
        "show_as_note": True,
    }


def _mock_reads(respx_mock, events):
    respx_mock.get("/athlete/i123456/sport-settings").mock(
        return_value=Response(200, json=SPORT_SETTINGS)
    )
    return respx_mock.get("/athlete/i123456/events").mock(
        return_value=Response(200, json=events)
    )


class TestWeeklyTargetHelpers:
    def test_sport_settings_preserves_all_raw_types(self):
        settings = SportSettings.model_validate(SPORT_SETTINGS[0])
        assert settings.type == "Ride"
        assert settings.types == ["Ride", "VirtualRide", "GravelRide"]

    def test_unit_conversions_are_centralized(self):
        assert _target_values_to_api(300, 360, 180) == {
            "load_target": 300,
            "time_target": 21600,
            "distance_target": 180000,
        }

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"load_target": 0}, "load_target"),
            ({"time_target_minutes": -1}, "time_target_minutes"),
            ({"distance_target_km": 0}, "distance_target_km"),
        ],
    )
    async def test_non_positive_values_are_rejected(
        self, mock_config, respx_mock, kwargs, field
    ):
        result = await set_weekly_sport_target(
            "Ride", WEEK, ctx=_ctx(mock_config), **kwargs
        )
        assert field in json.loads(result)["error"]["message"]
        assert not respx_mock.calls


class TestWeeklyTargetPreview:
    async def test_create_preview_is_read_only_and_generic(self, mock_config, respx_mock):
        events_route = _mock_reads(respx_mock, [])
        result = json.loads(
            await set_weekly_sport_target(
                "Swim",
                WEEK,
                time_target_minutes=120,
                distance_target_km=6,
                ctx=_ctx(mock_config),
            )
        )
        assert result["data"]["action"] == "created"
        assert result["data"]["sport_type"] == "Swim"
        assert result["data"]["proposed"] == {
            "load_target": None,
            "time_target_minutes": 120,
            "distance_target_km": 6,
        }
        assert result["data"]["requires_confirmation"] is True
        assert result["metadata"]["write"] is False
        assert events_route.called
        assert len(respx_mock.calls) == 2

    async def test_none_preserves_existing_values(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [_target()])
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=320, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["current"]["time_target_minutes"] == 300
        assert result["data"]["proposed"] == {
            "load_target": 320,
            "time_target_minutes": 300,
            "distance_target_km": 120,
        }

    async def test_conventional_target_is_ignored(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [_target(for_week=False)])
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=300, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["action"] == "created"
        assert result["data"]["target_id"] is None

    async def test_sports_same_week_remain_independent(self, mock_config, respx_mock):
        _mock_reads(
            respx_mock,
            [_target(sport="Ride"), _target(9002, sport="Swim", load=100)],
        )
        result = json.loads(
            await set_weekly_sport_target(
                "Swim", WEEK, distance_target_km=8, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["target_id"] == "9002"
        assert result["data"]["sport_type"] == "Swim"

    async def test_duplicate_weekly_targets_conflict(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [_target(), _target(9002)])
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=300, ctx=_ctx(mock_config)
            )
        )
        assert result["error"]["type"] == "conflict_error"
        assert len(respx_mock.calls) == 2

    async def test_invalid_sport_returns_native_values(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=SPORT_SETTINGS)
        )
        result = json.loads(
            await set_weekly_sport_target(
                "ride", WEEK, load_target=300, ctx=_ctx(mock_config)
            )
        )
        message = result["error"]["message"]
        assert "ride" in message and "GravelRide" in message and "OpenWaterSwim" in message
        assert len(respx_mock.calls) == 1

    async def test_non_monday_is_rejected_before_api(self, mock_config, respx_mock):
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", "2026-08-25", load_target=300, ctx=_ctx(mock_config)
            )
        )
        assert "Monday" in result["error"]["message"]
        assert not respx_mock.calls


class TestWeeklyTargetConfirmation:
    async def test_create_uses_minimal_payload_then_verifies(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [])
        post = respx_mock.post("/athlete/i123456/events").mock(
            return_value=Response(200, json=_target(load=300, time=None, distance=None))
        )
        get = respx_mock.get("/athlete/i123456/events/9001").mock(
            return_value=Response(200, json=_target(load=300, time=None, distance=None))
        )
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=300, confirm=True, ctx=_ctx(mock_config)
            )
        )
        payload = json.loads(post.calls[0].request.content)
        assert payload == {
            "category": "TARGET",
            "type": "Ride",
            "for_week": True,
            "start_date_local": WEEK,
            "load_target": 300,
        }
        assert get.called
        assert result["data"]["action"] == "created"
        assert result["data"]["verified"] is True

    async def test_update_is_partial_and_preserves_unmanaged_fields(
        self, mock_config, respx_mock
    ):
        _mock_reads(respx_mock, [_target()])
        put = respx_mock.put("/athlete/i123456/events/9001").mock(
            return_value=Response(200, json=_target(load=320))
        )
        respx_mock.get("/athlete/i123456/events/9001").mock(
            return_value=Response(200, json=_target(load=320))
        )
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=320, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert json.loads(put.calls[0].request.content) == {"load_target": 320}
        assert result["data"]["time_target_minutes"] == 300
        assert result["data"]["distance_target_km"] == 120

    async def test_unchanged_performs_zero_writes(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [_target()])
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=280, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["action"] == "unchanged"
        assert result["metadata"]["write"] is False
        assert len(respx_mock.calls) == 2

    async def test_verification_mismatch_is_error(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [_target()])
        respx_mock.put("/athlete/i123456/events/9001").mock(
            return_value=Response(200, json=_target(load=320))
        )
        respx_mock.get("/athlete/i123456/events/9001").mock(
            return_value=Response(200, json=_target(load=319))
        )
        result = json.loads(
            await set_weekly_sport_target(
                "Ride", WEEK, load_target=320, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["error"]["type"] == "verification_error"

    async def test_api_error_is_reported(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(500, json={})
        )
        result = json.loads(
            await set_weekly_sport_target(
                "Run", WEEK, load_target=100, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["error"]["type"] == "api_error"
