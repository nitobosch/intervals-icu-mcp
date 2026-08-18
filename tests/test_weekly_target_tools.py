"""Contract tests for generic weekly sport TARGET upserts."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import Response

from intervals_icu_mcp.models import SportSettings
from intervals_icu_mcp.tools.weekly_targets import (
    _target_values_to_api,
    delete_weekly_sport_target,
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
    start_date_local: str = f"{WEEK}T00:00:00",
) -> dict:
    return {
        "id": event_id,
        "start_date_local": start_date_local,
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

    async def test_datetime_input_is_rejected_before_api(self, mock_config, respx_mock):
        result = json.loads(
            await set_weekly_sport_target(
                "Ride",
                "2026-08-24T00:00:00",
                load_target=300,
                ctx=_ctx(mock_config),
            )
        )
        assert "YYYY-MM-DD" in result["error"]["message"]
        assert not respx_mock.calls


class TestWeeklyTargetConfirmation:
    async def test_create_uses_minimal_payload_then_verifies(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [])
        post = respx_mock.post("/athlete/i123456/events").mock(
            return_value=Response(200, json=_target(load=300, time=21600, distance=180000))
        )
        get = respx_mock.get("/athlete/i123456/events/9001").mock(
            return_value=Response(200, json=_target(load=300, time=21600, distance=180000))
        )
        result = json.loads(
            await set_weekly_sport_target(
                "Ride",
                WEEK,
                load_target=300,
                time_target_minutes=360,
                distance_target_km=180,
                confirm=True,
                ctx=_ctx(mock_config),
            )
        )
        payload = json.loads(post.calls[0].request.content)
        assert payload == {
            "category": "TARGET",
            "type": "Ride",
            "name": "Weekly",
            "for_week": True,
            "start_date_local": "2026-08-24T00:00:00",
            "load_target": 300,
            "time_target": 21600,
            "distance_target": 180000,
        }
        assert payload["start_date_local"].endswith("T00:00:00")
        assert "Z" not in payload["start_date_local"]
        assert "+" not in payload["start_date_local"]
        search_request = respx_mock.calls[1].request
        assert search_request.url.params["oldest"] == WEEK
        assert search_request.url.params["newest"] == WEEK
        assert get.called
        assert result["data"]["action"] == "created"
        assert result["data"]["week_start_date"] == WEEK
        assert result["data"]["verified"] is True

    async def test_update_matches_other_name_and_preserves_unmanaged_fields(
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
        update_payload = json.loads(put.calls[0].request.content)
        assert update_payload == {"load_target": 320}
        assert "name" not in update_payload
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


class TestDeleteWeeklySportTarget:
    async def test_preview_finds_realistic_ride_target_without_delete(
        self, mock_config, respx_mock
    ):
        _mock_reads(
            respx_mock,
            [_target(130144219, load=300, time=21600, distance=180000)],
        )
        result = json.loads(
            await delete_weekly_sport_target("Ride", WEEK, ctx=_ctx(mock_config))
        )
        assert result["data"] == {
            "week_start_date": WEEK,
            "sport_type": "Ride",
            "target_id": "130144219",
            "current": {
                "load_target": 300,
                "time_target_minutes": 360,
                "distance_target_km": 180,
            },
            "action": "delete",
            "requires_confirmation": True,
        }
        assert result["metadata"]["write"] is False
        assert len(respx_mock.calls) == 2

    async def test_confirm_deletes_exact_target_and_verifies_disappearance(
        self, mock_config, respx_mock
    ):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=SPORT_SETTINGS)
        )
        events = respx_mock.get("/athlete/i123456/events").mock(
            side_effect=[
                Response(200, json=[_target(130144219)]),
                Response(200, json=[]),
            ]
        )
        delete = respx_mock.delete("/athlete/i123456/events/130144219").mock(
            return_value=Response(204)
        )
        result = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert delete.call_count == 1
        assert events.call_count == 2
        assert result["data"] == {
            "week_start_date": WEEK,
            "sport_type": "Ride",
            "target_id": "130144219",
            "action": "deleted",
            "verified": True,
        }
        assert result["metadata"]["write"] is True

    async def test_repeated_confirm_is_idempotent(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=SPORT_SETTINGS)
        )
        respx_mock.get("/athlete/i123456/events").mock(
            side_effect=[
                Response(200, json=[_target(130144219)]),
                Response(200, json=[]),
                Response(200, json=[]),
            ]
        )
        delete = respx_mock.delete("/athlete/i123456/events/130144219").mock(
            return_value=Response(204)
        )
        first = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        second = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert first["data"]["action"] == "deleted"
        assert second["data"]["action"] == "not_found"
        assert second["data"]["verified"] is True
        assert delete.call_count == 1

    async def test_not_found_is_success_without_delete(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [])
        result = json.loads(
            await delete_weekly_sport_target(
                "Run", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["action"] == "not_found"
        assert result["data"]["requires_confirmation"] is False
        assert result["data"]["verified"] is True
        assert len(respx_mock.calls) == 2

    async def test_structurally_unrelated_events_are_never_deleted(
        self, mock_config, respx_mock
    ):
        events = [
            _target(1, for_week=False),
            _target(2, sport="Swim"),
            _target(3, start_date_local="2026-08-31T00:00:00"),
            {
                "id": 4,
                "start_date_local": f"{WEEK}T00:00:00",
                "category": "WORKOUT",
                "type": "Ride",
                "for_week": True,
            },
        ]
        _mock_reads(respx_mock, events)
        result = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["action"] == "not_found"
        assert len(respx_mock.calls) == 2

    async def test_ride_delete_does_not_delete_swim(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=SPORT_SETTINGS)
        )
        swim = _target(2, sport="Swim")
        respx_mock.get("/athlete/i123456/events").mock(
            side_effect=[
                Response(200, json=[_target(1), swim]),
                Response(200, json=[swim]),
            ]
        )
        delete = respx_mock.delete("/athlete/i123456/events/1").mock(
            return_value=Response(204)
        )
        result = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["action"] == "deleted"
        assert delete.called

    async def test_swim_delete_does_not_delete_ride(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=SPORT_SETTINGS)
        )
        ride = _target(1)
        respx_mock.get("/athlete/i123456/events").mock(
            side_effect=[
                Response(200, json=[ride, _target(2, sport="Swim")]),
                Response(200, json=[ride]),
            ]
        )
        delete = respx_mock.delete("/athlete/i123456/events/2").mock(
            return_value=Response(204)
        )
        result = json.loads(
            await delete_weekly_sport_target(
                "Swim", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["data"]["action"] == "deleted"
        assert delete.called

    async def test_duplicate_targets_conflict_without_delete(self, mock_config, respx_mock):
        _mock_reads(respx_mock, [_target(1), _target(2)])
        result = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["error"]["type"] == "conflict_error"
        assert len(respx_mock.calls) == 2

    async def test_invalid_sport_lists_available_types(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=SPORT_SETTINGS)
        )
        result = json.loads(
            await delete_weekly_sport_target("ride", WEEK, ctx=_ctx(mock_config))
        )
        assert result["error"]["type"] == "validation_error"
        assert "GravelRide" in result["error"]["message"]
        assert len(respx_mock.calls) == 1

    async def test_non_monday_is_rejected_before_api(self, mock_config, respx_mock):
        result = json.loads(
            await delete_weekly_sport_target(
                "Ride", "2026-08-25", ctx=_ctx(mock_config)
            )
        )
        assert "Monday" in result["error"]["message"]
        assert not respx_mock.calls

    @pytest.mark.parametrize("status", [401, 500])
    async def test_delete_api_error_is_reported(self, mock_config, respx_mock, status):
        _mock_reads(respx_mock, [_target(1)])
        respx_mock.delete("/athlete/i123456/events/1").mock(
            return_value=Response(status, json={})
        )
        result = json.loads(
            await delete_weekly_sport_target(
                "Ride", WEEK, confirm=True, ctx=_ctx(mock_config)
            )
        )
        assert result["error"]["type"] == "api_error"
