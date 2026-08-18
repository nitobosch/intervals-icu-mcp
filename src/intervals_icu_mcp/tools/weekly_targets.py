"""Safe, sport-generic weekly target upserts for Intervals.icu."""

from datetime import date, datetime
from typing import Annotated, Any

from fastmcp import Context

from ..auth import ICUConfig
from ..client import ICUAPIError, ICUClient
from ..models import Event, SportSettings
from ..response_builder import ResponseBuilder


def _parse_week_start(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("week_start_date must use ISO YYYY-MM-DD format") from exc
    if parsed.weekday() != 0:
        raise ValueError("week_start_date must be a Monday; it is not normalized automatically")
    return parsed


def _target_values_to_api(
    load_target: float | None,
    time_target_minutes: int | None,
    distance_target_km: float | None,
) -> dict[str, int | float]:
    values = {
        "load_target": load_target,
        "time_target_minutes": time_target_minutes,
        "distance_target_km": distance_target_km,
    }
    if all(value is None for value in values.values()):
        raise ValueError("At least one target value must be provided")
    for name, value in values.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be greater than zero")

    payload: dict[str, int | float] = {}
    if load_target is not None:
        payload["load_target"] = load_target
    if time_target_minutes is not None:
        payload["time_target"] = time_target_minutes * 60
    if distance_target_km is not None:
        payload["distance_target"] = distance_target_km * 1000
    return payload


def _available_sport_types(settings: list[SportSettings]) -> list[str]:
    return sorted({sport_type for setting in settings for sport_type in setting.types})


def _event_week(event: Event) -> date:
    return datetime.fromisoformat(event.start_date_local).date()


def _matching_targets(
    events: list[Event], sport_type: str, week_start: date
) -> list[Event]:
    return [
        event
        for event in events
        if event.category == "TARGET"
        and event.for_week is True
        and event.type == sport_type
        and _event_week(event) == week_start
    ]


def _managed_state(event: Event | None) -> dict[str, int | float | None]:
    return {
        "load_target": event.load_target if event else None,
        "time_target_minutes": event.time_target / 60 if event and event.time_target is not None else None,
        "distance_target_km": event.distance_target / 1000
        if event and event.distance_target is not None
        else None,
    }


def _api_state(event: Event) -> dict[str, int | float | None]:
    return {
        "load_target": event.load_target,
        "time_target": event.time_target,
        "distance_target": event.distance_target,
    }


def _response_data(
    *,
    week_start: date,
    sport_type: str,
    event: Event | None,
    current: dict[str, int | float | None],
    proposed: dict[str, int | float | None],
    action: str,
    confirmed: bool,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "week_start_date": week_start.isoformat(),
        "sport_type": sport_type,
        "target_id": str(event.id) if event else None,
        "current": current,
        "proposed": proposed,
        "action": action,
    }
    if confirmed:
        data.update(proposed)
        data["verified"] = True
    else:
        data["requires_confirmation"] = True
    return data


def _verify_target(
    event: Event,
    *,
    week_start: date,
    sport_type: str,
    expected_api: dict[str, int | float | None],
) -> None:
    if (
        event.category != "TARGET"
        or event.for_week is not True
        or event.type != sport_type
        or _event_week(event) != week_start
        or _api_state(event) != expected_api
    ):
        raise ValueError("Weekly target verification failed: API state differs from requested state")


async def set_weekly_sport_target(
    sport_type: Annotated[str, "Exact native sport type from the athlete's sport settings"],
    week_start_date: Annotated[str, "Monday in ISO YYYY-MM-DD format"],
    load_target: Annotated[float | None, "Native Intervals.icu load target (> 0)"] = None,
    time_target_minutes: Annotated[int | None, "Weekly time target in minutes (> 0)"] = None,
    distance_target_km: Annotated[float | None, "Weekly distance target in km (> 0)"] = None,
    confirm: Annotated[bool, "False previews only; true performs and verifies the upsert"] = False,
    athlete_id: Annotated[str | None, "Athlete ID (for coaches managing multiple athletes)"] = None,
    ctx: Context | None = None,
) -> str:
    """Preview or confirm an idempotent weekly TARGET for any configured native sport."""
    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    try:
        week_start = _parse_week_start(week_start_date)
        requested_api = _target_values_to_api(
            load_target, time_target_minutes, distance_target_km
        )
    except ValueError as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="validation_error")

    try:
        async with ICUClient(config) as client:
            settings = await client.get_sport_settings(athlete_id=athlete_id)
            available = _available_sport_types(settings)
            if sport_type not in available:
                return ResponseBuilder.build_error_response(
                    f"Invalid sport_type {sport_type!r}. Available native types: {', '.join(available)}",
                    error_type="validation_error",
                )

            events = await client.get_events(
                athlete_id=athlete_id,
                oldest=week_start.isoformat(),
                newest=week_start.isoformat(),
            )
            matches = _matching_targets(events, sport_type, week_start)
            if len(matches) > 1:
                return ResponseBuilder.build_error_response(
                    "Multiple weekly TARGET events exist for this week and sport; refusing to choose one",
                    error_type="conflict_error",
                )

            existing = matches[0] if matches else None
            current = _managed_state(existing)
            proposed = dict(current)
            if "load_target" in requested_api:
                proposed["load_target"] = requested_api["load_target"]
            if "time_target" in requested_api:
                proposed["time_target_minutes"] = requested_api["time_target"] / 60
            if "distance_target" in requested_api:
                proposed["distance_target_km"] = requested_api["distance_target"] / 1000
            action = "created" if existing is None else ("unchanged" if proposed == current else "updated")

            if not confirm:
                return ResponseBuilder.build_response(
                    _response_data(
                        week_start=week_start,
                        sport_type=sport_type,
                        event=existing,
                        current=current,
                        proposed=proposed,
                        action=action,
                        confirmed=False,
                    ),
                    metadata={"write": False, "preview": True, "upsert": True, "source": "intervals.icu"},
                )

            if action == "unchanged":
                verified = existing
            elif existing is None:
                create_payload: dict[str, Any] = {
                    "category": "TARGET",
                    "type": sport_type,
                    "name": "Weekly",
                    "for_week": True,
                    "start_date_local": f"{week_start.isoformat()}T00:00:00",
                    **requested_api,
                }
                created = await client.create_event(create_payload, athlete_id=athlete_id)
                verified = await client.get_event(created.id, athlete_id=athlete_id)
            else:
                partial_payload = {
                    key: value
                    for key, value in requested_api.items()
                    if _api_state(existing)[key] != value
                }
                await client.update_event(existing.id, partial_payload, athlete_id=athlete_id)
                verified = await client.get_event(existing.id, athlete_id=athlete_id)

            assert verified is not None
            expected_api = {
                "load_target": proposed["load_target"],
                "time_target": int(proposed["time_target_minutes"] * 60)
                if proposed["time_target_minutes"] is not None
                else None,
                "distance_target": proposed["distance_target_km"] * 1000
                if proposed["distance_target_km"] is not None
                else None,
            }
            _verify_target(
                verified,
                week_start=week_start,
                sport_type=sport_type,
                expected_api=expected_api,
            )
            return ResponseBuilder.build_response(
                _response_data(
                    week_start=week_start,
                    sport_type=sport_type,
                    event=verified,
                    current=current,
                    proposed=proposed,
                    action=action,
                    confirmed=True,
                ),
                metadata={"write": action != "unchanged", "upsert": True, "source": "intervals.icu"},
            )
    except ICUAPIError as exc:
        return ResponseBuilder.build_error_response(exc.message, error_type="api_error")
    except ValueError as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="verification_error")
    except Exception as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="internal_error")


async def delete_weekly_sport_target(
    sport_type: Annotated[str, "Exact native sport type from the athlete's sport settings"],
    week_start_date: Annotated[str, "Monday in ISO YYYY-MM-DD format"],
    confirm: Annotated[bool, "False previews only; true deletes and verifies"] = False,
    athlete_id: Annotated[str | None, "Athlete ID (for coaches managing multiple athletes)"] = None,
    ctx: Context | None = None,
) -> str:
    """Preview or delete exactly one native weekly TARGET for a configured sport."""
    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    try:
        week_start = _parse_week_start(week_start_date)
    except ValueError as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="validation_error")

    try:
        async with ICUClient(config) as client:
            settings = await client.get_sport_settings(athlete_id=athlete_id)
            available = _available_sport_types(settings)
            if sport_type not in available:
                return ResponseBuilder.build_error_response(
                    f"Invalid sport_type {sport_type!r}. Available native types: {', '.join(available)}",
                    error_type="validation_error",
                )

            events = await client.get_events(
                athlete_id=athlete_id,
                oldest=week_start.isoformat(),
                newest=week_start.isoformat(),
            )
            matches = _matching_targets(events, sport_type, week_start)
            if len(matches) > 1:
                return ResponseBuilder.build_error_response(
                    "Multiple weekly TARGET events exist for this week and sport; refusing to choose one",
                    error_type="conflict_error",
                )

            existing = matches[0] if matches else None
            if existing is None:
                return ResponseBuilder.build_response(
                    {
                        "week_start_date": week_start.isoformat(),
                        "sport_type": sport_type,
                        "target_id": None,
                        "action": "not_found",
                        "requires_confirmation": False,
                        **({"verified": True} if confirm else {}),
                    },
                    metadata={
                        "write": False,
                        "delete": True,
                        "preview": not confirm,
                        "source": "intervals.icu",
                    },
                )

            if not confirm:
                return ResponseBuilder.build_response(
                    {
                        "week_start_date": week_start.isoformat(),
                        "sport_type": sport_type,
                        "target_id": str(existing.id),
                        "current": _managed_state(existing),
                        "action": "delete",
                        "requires_confirmation": True,
                    },
                    metadata={
                        "write": False,
                        "delete": True,
                        "preview": True,
                        "source": "intervals.icu",
                    },
                )

            await client.delete_event(existing.id, athlete_id=athlete_id)
            remaining_events = await client.get_events(
                athlete_id=athlete_id,
                oldest=week_start.isoformat(),
                newest=week_start.isoformat(),
            )
            if _matching_targets(remaining_events, sport_type, week_start):
                raise ValueError(
                    "Weekly target deletion verification failed: matching target still exists"
                )

            return ResponseBuilder.build_response(
                {
                    "week_start_date": week_start.isoformat(),
                    "sport_type": sport_type,
                    "target_id": str(existing.id),
                    "action": "deleted",
                    "verified": True,
                },
                metadata={
                    "write": True,
                    "delete": True,
                    "source": "intervals.icu",
                },
            )
    except ICUAPIError as exc:
        return ResponseBuilder.build_error_response(exc.message, error_type="api_error")
    except ValueError as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="verification_error")
    except Exception as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="internal_error")
