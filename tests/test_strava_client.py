"""Tests for the direct Strava API client."""

import json
import stat
import time

import respx
from httpx import Response

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.strava_client import StravaClient

EFFORT_ID = "9007199254740993"


def _config(tmp_path, *, refresh_token: str = "seed-refresh") -> ICUConfig:
    return ICUConfig(
        strava_client_id="12345",
        strava_client_secret="test-secret",
        strava_refresh_token=refresh_token,
        strava_token_store=str(tmp_path / "tokens.json"),
    )


def _persist_valid_token(config: ICUConfig) -> None:
    path = config.strava_token_store

    payload = {
        "access_token": "persisted-access",
        "refresh_token": "persisted-refresh",
        "expires_at": int(time.time()) + 21600,
    }

    from pathlib import Path

    token_path = Path(path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(payload))


async def test_refresh_persists_rotated_token_with_secure_permissions(tmp_path):
    config = _config(tmp_path)

    with respx.mock(
        base_url="https://www.strava.com",
        assert_all_called=False,
    ) as router:
        token_route = router.post("/oauth/token").mock(
            return_value=Response(
                200,
                json={
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_at": int(time.time()) + 21600,
                },
            )
        )

        starred_route = router.get("/api/v3/segments/starred").mock(
            side_effect=[
                Response(200, json=[{"id": 12345678}]),
                Response(200, json=[]),
            ]
        )

        async with StravaClient(config) as client:
            segments = await client.get_starred_segments(limit=500)

    assert segments == [{"id": 12345678}]
    assert token_route.called
    assert starred_route.call_count == 2

    from pathlib import Path

    token_path = Path(config.strava_token_store)
    persisted = json.loads(token_path.read_text())

    assert persisted["access_token"] == "new-access"
    assert persisted["refresh_token"] == "new-refresh"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


async def test_get_activity_requests_all_segment_efforts(tmp_path):
    config = _config(tmp_path, refresh_token="")
    _persist_valid_token(config)

    with respx.mock(
        base_url="https://www.strava.com",
        assert_all_called=True,
    ) as router:
        route = router.get("/api/v3/activities/9007199254740993").mock(
            return_value=Response(
                200,
                json={"id": 9007199254740993, "segment_efforts": []},
            )
        )

        async with StravaClient(config) as client:
            activity = await client.get_activity(
                "9007199254740993",
                include_all_efforts=True,
            )

    assert activity["id"] == 9007199254740993
    assert route.call_count == 1
    request = route.calls[0].request
    assert request.url.params["include_all_efforts"] == "true"


async def test_persisted_token_works_without_environment_refresh_token(tmp_path):
    config = _config(tmp_path, refresh_token="")
    _persist_valid_token(config)

    with respx.mock(
        base_url="https://www.strava.com",
        assert_all_called=False,
    ) as router:
        token_route = router.post("/oauth/token").mock(return_value=Response(500))

        starred_route = router.get("/api/v3/segments/starred").mock(
            side_effect=[
                Response(200, json=[{"id": 1}]),
                Response(200, json=[]),
            ]
        )

        async with StravaClient(config) as client:
            assert client.configured is True
            segments = await client.get_starred_segments(limit=500)

    assert segments == [{"id": 1}]
    assert token_route.called is False
    assert starred_route.call_count == 2


async def test_401_refreshes_once_and_retries_request(tmp_path):
    config = _config(tmp_path, refresh_token="")
    _persist_valid_token(config)

    with respx.mock(
        base_url="https://www.strava.com",
        assert_all_called=False,
    ) as router:
        token_route = router.post("/oauth/token").mock(
            return_value=Response(
                200,
                json={
                    "access_token": "refreshed-access",
                    "refresh_token": "refreshed-refresh",
                    "expires_at": int(time.time()) + 21600,
                },
            )
        )

        starred_route = router.get("/api/v3/segments/starred").mock(
            side_effect=[
                Response(401, json={"message": "Authorization Error"}),
                Response(200, json=[]),
            ]
        )

        async with StravaClient(config) as client:
            segments = await client.get_starred_segments(limit=500)

    assert segments == []
    assert token_route.call_count == 1
    assert starred_route.call_count == 2

    requests = [call.request for call in starred_route.calls]

    assert requests[0].headers["Authorization"] == "Bearer persisted-access"
    assert requests[1].headers["Authorization"] == "Bearer refreshed-access"


async def test_starred_pagination_continues_until_empty_page(tmp_path):
    config = _config(tmp_path, refresh_token="")
    _persist_valid_token(config)

    first_page = [{"id": index} for index in range(107)]

    with respx.mock(
        base_url="https://www.strava.com",
        assert_all_called=False,
    ) as router:
        route = router.get("/api/v3/segments/starred").mock(
            side_effect=[
                Response(200, json=first_page),
                Response(200, json=[]),
            ]
        )

        async with StravaClient(config) as client:
            segments = await client.get_starred_segments(limit=500)

    assert len(segments) == 107
    assert route.call_count == 2

    requests = [call.request for call in route.calls]

    assert requests[0].url.params["page"] == "1"
    assert requests[0].url.params["per_page"] == "200"
    assert requests[1].url.params["page"] == "2"


async def test_segment_effort_streams_use_official_endpoint_and_exact_id(tmp_path):
    config = _config(tmp_path, refresh_token="")
    _persist_valid_token(config)

    with respx.mock(
        base_url="https://www.strava.com",
        assert_all_called=False,
    ) as router:
        route = router.get(f"/api/v3/segment_efforts/{EFFORT_ID}/streams").mock(
            return_value=Response(
                200,
                json={
                    "time": {
                        "data": [0, 1, 2],
                        "original_size": 3,
                        "resolution": "high",
                        "series_type": "distance",
                    }
                },
            )
        )

        async with StravaClient(config) as client:
            streams = await client.get_segment_effort_streams(
                EFFORT_ID,
                keys=["time", "heartrate"],
            )

    assert "time" in streams
    assert route.call_count == 1

    request = route.calls[0].request

    assert EFFORT_ID in str(request.url)
    assert request.url.params["keys"] == "time,heartrate"
    assert request.url.params["key_by_type"] == "true"
