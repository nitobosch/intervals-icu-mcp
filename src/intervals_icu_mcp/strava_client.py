"""Async read-only client for the Strava API."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import httpx

from .auth import ICUConfig

STRAVA_API_BASE_URL = "https://www.strava.com/api/v3"

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"


STRAVA_TOKEN_REFRESH_WINDOW_SECONDS = 3600

_TOKEN_LOCK = asyncio.Lock()


class StravaAPIError(Exception):
    """Error returned while communicating with Strava."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:

        super().__init__(message)

        self.message = message

        self.status_code = status_code

        self.endpoint = endpoint


class StravaClient:
    """Small read-only Strava client with persistent OAuth refresh."""

    def __init__(self, config: ICUConfig) -> None:

        self.client_id = config.strava_client_id.strip()

        self.client_secret = config.strava_client_secret.strip()

        self.env_refresh_token = config.strava_refresh_token.strip()

        self.token_store = Path(config.strava_token_store)

        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> StravaClient:

        self._http = httpx.AsyncClient(timeout=30.0)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:

        if self._http is not None:
            await self._http.aclose()

            self._http = None

    def _load_tokens(self) -> dict[str, Any]:

        tokens: dict[str, Any] = {}

        if self.token_store.exists():
            try:
                parsed: Any = json.loads(self.token_store.read_text(encoding="utf-8"))

                if isinstance(parsed, dict):
                    tokens.update(cast(dict[str, Any], parsed))

            except (OSError, ValueError):
                pass

        if not tokens.get("refresh_token") and self.env_refresh_token:
            tokens["refresh_token"] = self.env_refresh_token

        return tokens

    @property
    def configured(self) -> bool:

        tokens = self._load_tokens()

        return bool(self.client_id and self.client_secret and tokens.get("refresh_token"))

    def _persist_tokens(self, tokens: dict[str, Any]) -> None:

        self.token_store.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )

        payload = {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "expires_at": tokens.get("expires_at"),
        }

        temp_path = self.token_store.with_suffix(self.token_store.suffix + ".tmp")

        temp_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        os.chmod(temp_path, 0o600)

        os.replace(temp_path, self.token_store)

    def _build_error(
        self,
        response: httpx.Response,
        *,
        endpoint: str,
    ) -> StravaAPIError:

        message = f"Strava API returned HTTP {response.status_code}"

        try:
            body: Any = response.json()

            if isinstance(body, dict):
                body_dict = cast(dict[str, Any], body)

                if body_dict.get("message"):
                    message += f": {body_dict['message']}"

                if body_dict.get("errors"):
                    message += f" ({body_dict['errors']})"

        except ValueError:
            if response.text:
                message += f": {response.text[:500]}"

        return StravaAPIError(
            message,
            status_code=response.status_code,
            endpoint=endpoint,
        )

    async def _refresh_access_token(
        self,
        *,
        force: bool = False,
    ) -> str:

        if self._http is None:
            raise RuntimeError("StravaClient must be used as an async context manager")

        async with _TOKEN_LOCK:
            tokens = self._load_tokens()

            access_token = str(tokens.get("access_token") or "")

            expires_at = int(tokens.get("expires_at") or 0)

            if (
                not force
                and access_token
                and expires_at > int(time.time()) + STRAVA_TOKEN_REFRESH_WINDOW_SECONDS
            ):
                return access_token

            refresh_token = str(tokens.get("refresh_token") or "")

            if not refresh_token:
                raise StravaAPIError("Strava refresh token is not configured.")

            response = await self._http.post(
                STRAVA_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )

            if response.status_code >= 400:
                raise self._build_error(
                    response,
                    endpoint="/oauth/token",
                )

            refreshed_raw: Any = response.json()

            if not isinstance(refreshed_raw, dict):
                raise StravaAPIError("Unexpected response from Strava OAuth token endpoint.")

            refreshed = cast(dict[str, Any], refreshed_raw)

            access_token = refreshed.get("access_token")

            refresh_token = refreshed.get("refresh_token")

            if not access_token or not refresh_token:
                raise StravaAPIError(
                    "Strava OAuth response did not contain both access_token and refresh_token."
                )

            if not refreshed.get("expires_at"):
                refreshed["expires_at"] = int(time.time()) + int(
                    refreshed.get("expires_in") or 21600
                )

            self._persist_tokens(refreshed)

            return str(access_token)

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:

        if self._http is None:
            raise RuntimeError("StravaClient must be used as an async context manager")

        token = await self._refresh_access_token()

        response = await self._http.request(
            method,
            f"{STRAVA_API_BASE_URL}{endpoint}",
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )

        if response.status_code == 401:
            token = await self._refresh_access_token(force=True)

            response = await self._http.request(
                method,
                f"{STRAVA_API_BASE_URL}{endpoint}",
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
            )

        if response.status_code >= 400:
            raise self._build_error(
                response,
                endpoint=endpoint,
            )

        payload: Any = response.json()
        return payload

    async def get_activity(
        self,
        strava_activity_id: str,
        *,
        include_all_efforts: bool = True,
    ) -> dict[str, Any]:
        """Return one detailed Strava activity with optional segment efforts."""

        result = await self._request(
            "GET",
            f"/activities/{strava_activity_id}",
            params={
                "include_all_efforts": (
                    "true" if include_all_efforts else "false"
                )
            },
        )

        if not isinstance(result, dict):
            raise StravaAPIError(
                f"Unexpected response from /activities/{strava_activity_id}."
            )

        return cast(dict[str, Any], result)

    async def get_starred_segments(
        self,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []

        page = 1

        per_page = min(200, limit)

        while len(result) < limit:
            batch = await self._request(
                "GET",
                "/segments/starred",
                params={
                    "page": page,
                    "per_page": per_page,
                },
            )

            if not isinstance(batch, list):
                raise StravaAPIError("Unexpected response from /segments/starred.")

            batch_items = cast(list[Any], batch)

            if not batch_items:
                break

            for segment_raw in batch_items:
                if not isinstance(segment_raw, dict):
                    raise StravaAPIError("Unexpected segment item from /segments/starred.")

                result.append(cast(dict[str, Any], segment_raw))

            page += 1

        return result[:limit]

    async def get_segment(
        self,
        segment_id: str,
    ) -> dict[str, Any]:
        """Return detailed information for a Strava segment."""

        result = await self._request(
            "GET",
            f"/segments/{segment_id}",
        )

        if not isinstance(result, dict):
            raise StravaAPIError(f"Unexpected response from /segments/{segment_id}.")

        return cast(dict[str, Any], result)

    async def get_segment_efforts(
        self,
        segment_id: str,
        *,
        limit: int = 200,
        start_date_local: str | None = None,
        end_date_local: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return segment efforts with defensive pagination."""

        result: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        page = 1
        per_page = min(200, limit)

        while len(result) < limit:
            params: dict[str, Any] = {
                "segment_id": segment_id,
                "page": page,
                "per_page": per_page,
            }

            if start_date_local:
                params["start_date_local"] = start_date_local

            if end_date_local:
                params["end_date_local"] = end_date_local

            batch = await self._request(
                "GET",
                "/segment_efforts",
                params=params,
            )

            if not isinstance(batch, list):
                raise StravaAPIError("Unexpected response from /segment_efforts.")

            batch_items = cast(list[Any], batch)

            if not batch_items:
                break

            new_items: list[dict[str, Any]] = []

            for effort_raw in batch_items:
                if not isinstance(effort_raw, dict):
                    raise StravaAPIError("Unexpected effort item from /segment_efforts.")

                effort = cast(dict[str, Any], effort_raw)
                effort_id = str(effort.get("id"))

                if effort_id not in seen_ids:
                    seen_ids.add(effort_id)
                    new_items.append(effort)

            # Defensive guard in case an API endpoint ignores the page
            # parameter and keeps returning the same data.
            if not new_items:
                break

            result.extend(new_items)
            page += 1

        return result[:limit]

    async def get_segment_effort(
        self,
        effort_id: str,
    ) -> dict[str, Any]:
        """Return detailed information for one segment effort."""

        result = await self._request(
            "GET",
            f"/segment_efforts/{effort_id}",
        )

        if not isinstance(result, dict):
            raise StravaAPIError(f"Unexpected response from /segment_efforts/{effort_id}.")

        return cast(dict[str, Any], result)

    async def get_segment_effort_streams(
        self,
        effort_id: str,
        *,
        keys: list[str],
    ) -> dict[str, Any]:
        """Return the official Strava streams for one segment effort."""

        result = await self._request(
            "GET",
            f"/segment_efforts/{effort_id}/streams",
            params={
                "keys": ",".join(keys),
                "key_by_type": "true",
            },
        )

        if not isinstance(result, dict):
            raise StravaAPIError("Unexpected response from segment effort streams.")

        return cast(dict[str, Any], result)
