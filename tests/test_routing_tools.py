"""Tests for routing location helpers."""

from unittest.mock import AsyncMock

import pytest

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.openrouteservice_client import OpenRouteServiceClient
from intervals_icu_mcp.tools.routing import (
    GeocodeCandidate,
    LocationResolutionError,
    extract_geocode_candidates,
    geocode_location_candidates,
    name_token_coverage,
    normalize_location_text,
    parse_lat_lon,
    resolve_coordinate_location,
    resolve_location,
    resolve_named_location,
    select_geocode_candidate,
    snap_coordinate,
    split_location_query,
)


def _config() -> ICUConfig:
    return ICUConfig(
        openrouteservice_api_key="test-ors-key",
        openrouteservice_base_url="https://api.openrouteservice.org",
    )


def test_parse_lat_lon() -> None:
    assert parse_lat_lon("39.589985, 2.630108") == (
        39.589985,
        2.630108,
    )

    assert parse_lat_lon("Palma de Mallorca") is None
    assert parse_lat_lon("Son Moix, Palma") is None


def test_parse_lat_lon_rejects_invalid_ranges() -> None:
    with pytest.raises(
        ValueError,
        match="latitude must be between -90 and 90",
    ):
        parse_lat_lon("95,2.6")

    with pytest.raises(
        ValueError,
        match="longitude must be between -180 and 180",
    ):
        parse_lat_lon("39.5,200")


def test_extract_geocode_candidates_filters_invalid_features() -> None:
    candidates = extract_geocode_candidates(
        {
            "features": [
                {
                    "properties": {
                        "name": "Visit Mallorca Estadi",
                        "label": "Visit Mallorca Estadi, Palma, PM, Spain",
                        "layer": "venue",
                        "locality": "Palma",
                        "country": "Spain",
                    },
                    "geometry": {
                        "coordinates": [2.630108, 39.589985],
                    },
                },
                {
                    "properties": {},
                    "geometry": {
                        "coordinates": ["invalid", 39.0],
                    },
                },
            ]
        }
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.label == "Visit Mallorca Estadi, Palma, PM, Spain"
    assert candidate.longitude == 2.630108
    assert candidate.latitude == 39.589985
    assert candidate.layer == "venue"
    assert candidate.locality == "Palma"
    assert candidate.country == "Spain"


async def test_geocode_candidates_do_not_select_silently() -> None:
    geocode = AsyncMock(
        return_value={
            "features": [
                {
                    "properties": {
                        "name": "Candidate A",
                        "label": "Candidate A",
                        "layer": "venue",
                    },
                    "geometry": {
                        "coordinates": [2.63, 39.59],
                    },
                },
                {
                    "properties": {
                        "name": "Candidate B",
                        "label": "Candidate B",
                        "layer": "venue",
                    },
                    "geometry": {
                        "coordinates": [2.64, 39.60],
                    },
                },
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]

        candidates = await geocode_location_candidates(
            client,
            "Son Moix, Palma de Mallorca",
        )

    assert [candidate.label for candidate in candidates] == [
        "Candidate A",
        "Candidate B",
    ]

    geocode.assert_awaited_once_with(
        "Son Moix, Palma de Mallorca",
        size=5,
        country=None,
        focus_lon=None,
        focus_lat=None,
    )


async def test_snap_coordinate() -> None:
    snap = AsyncMock(
        return_value={
            "locations": [
                {
                    "location": [2.631246, 39.590265],
                    "snapped_distance": 102.39,
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.snap = snap  # type: ignore[method-assign]

        result = await snap_coordinate(
            client,
            longitude=2.630108,
            latitude=39.589985,
        )

    assert result.longitude == 2.631246
    assert result.latitude == 39.590265
    assert result.snapped_distance_m == 102.39

    snap.assert_awaited_once_with(
        [[2.630108, 39.589985]],
        profile="cycling-road",
        radius=350.0,
    )


async def test_snap_coordinate_handles_no_match() -> None:
    snap = AsyncMock(
        return_value={
            "locations": [None],
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.snap = snap  # type: ignore[method-assign]

        with pytest.raises(
            LocationResolutionError,
            match="No cycling-road network",
        ):
            await snap_coordinate(
                client,
                longitude=2.630108,
                latitude=39.589985,
            )


async def test_direct_coordinates_bypass_geocoder() -> None:
    snap = AsyncMock(
        return_value={
            "locations": [
                {
                    "location": [2.631246, 39.590265],
                    "snapped_distance": 102.39,
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.snap = snap  # type: ignore[method-assign]

        result = await resolve_coordinate_location(
            client,
            "39.589985,2.630108",
        )

    assert result is not None
    assert result.source == "coordinates"
    assert result.original_longitude == 2.630108
    assert result.original_latitude == 39.589985
    assert result.longitude == 2.631246
    assert result.latitude == 39.590265
    assert result.snapped_distance_m == 102.39


async def test_non_coordinate_input_is_not_resolved_as_coordinates() -> None:
    async with OpenRouteServiceClient(_config()) as client:
        result = await resolve_coordinate_location(
            client,
            "Son Moix, Palma de Mallorca",
        )

    assert result is None


def _candidate(
    name: str,
    *,
    label: str | None = None,
    layer: str = "venue",
    locality: str | None = None,
    localadmin: str | None = None,
) -> GeocodeCandidate:
    return GeocodeCandidate(
        name=name,
        label=label or name,
        longitude=2.65,
        latitude=39.60,
        layer=layer,
        locality=locality,
        localadmin=localadmin,
        region="Balearic Islands",
        country="Spain",
    )


def test_location_text_normalization() -> None:
    assert normalize_location_text("Alaró") == "alaro"
    assert normalize_location_text("Palma de Mallorca") == "palma mallorca"

    assert split_location_query(
        "Orient, Bunyola, Mallorca"
    ) == (
        "Orient",
        ("Bunyola", "Mallorca"),
    )

    assert name_token_coverage(
        "Estadi Mallorca Son Moix",
        "Visit Mallorca Estadi",
    ) == 0.5


def test_son_moix_false_matches_remain_ambiguous() -> None:
    candidates = [
        _candidate(
            "Son Costa - Son Fortesa",
            locality="Palma de Mallorca",
            localadmin="Palma",
        ),
        _candidate(
            "Son Fuster",
            locality="Palma de Mallorca",
            localadmin="Palma",
        ),
        _candidate(
            "Son Ametler",
            locality="Palma de Mallorca",
            localadmin="Palma",
        ),
    ]

    with pytest.raises(
        LocationResolutionError,
        match="ambiguous",
    ):
        select_geocode_candidate(
            "Son Moix, Palma de Mallorca",
            candidates,
        )


def test_clear_partial_name_match_can_be_selected() -> None:
    candidates = [
        _candidate(
            "Visit Mallorca Estadi",
            label="Visit Mallorca Estadi, Palma, PM, Spain",
            localadmin="Palma",
        ),
        _candidate(
            "Mallorca",
            layer="localadmin",
            localadmin="Mallorca",
        ),
        _candidate(
            "Palma de Mallorca",
            layer="locality",
            locality="Palma de Mallorca",
            localadmin="Palma",
        ),
        _candidate(
            "Estadi Balear",
            locality="Palma de Mallorca",
            localadmin="Palma",
        ),
    ]

    selected = select_geocode_candidate(
        "Estadi Mallorca Son Moix, Palma de Mallorca",
        candidates,
    )

    assert selected.name == "Visit Mallorca Estadi"


def test_exact_place_name_prefers_unique_locality() -> None:
    candidates = [
        _candidate(
            "Bunyola",
            layer="locality",
            locality="Bunyola",
            localadmin="Bunyola",
        ),
        _candidate(
            "Bunyola",
            layer="venue",
            localadmin="Bunyola",
        ),
        _candidate(
            "Carrer de Bunyola",
            layer="street",
            locality="Palma de Mallorca",
            localadmin="Palma",
        ),
    ]

    selected = select_geocode_candidate(
        "Bunyola, Mallorca",
        candidates,
    )

    assert selected.layer == "locality"


def test_orient_mallorca_remains_ambiguous() -> None:
    candidates = [
        _candidate(
            "Orient",
            locality="Cadenes Ses",
            localadmin="Palma",
        ),
        _candidate(
            "Orient",
            localadmin="Bunyola",
        ),
        _candidate(
            "Orient",
            localadmin="Palma",
        ),
    ]

    with pytest.raises(
        LocationResolutionError,
        match="ambiguous",
    ):
        select_geocode_candidate(
            "Orient, Mallorca",
            candidates,
        )


def test_admin_context_disambiguates_orient() -> None:
    candidates = [
        _candidate(
            "Orient",
            locality="Cadenes Ses",
            localadmin="Palma",
        ),
        _candidate(
            "Orient",
            label="Orient, Bunyola, PM, Spain",
            localadmin="Bunyola",
        ),
        _candidate(
            "Orient",
            localadmin="Palma",
        ),
    ]

    selected = select_geocode_candidate(
        "Orient, Bunyola, Mallorca",
        candidates,
    )

    assert selected.localadmin == "Bunyola"


async def test_resolve_named_location_geocodes_selects_and_snaps() -> None:
    geocode = AsyncMock(
        return_value={
            "features": [
                {
                    "properties": {
                        "name": "Bunyola",
                        "label": "Bunyola, PM, Spain",
                        "layer": "locality",
                        "locality": "Bunyola",
                        "localadmin": "Bunyola",
                        "region": "Balearic Islands",
                        "country": "Spain",
                    },
                    "geometry": {
                        "coordinates": [2.700963, 39.694825],
                    },
                },
                {
                    "properties": {
                        "name": "Carrer de Bunyola",
                        "label": "Carrer de Bunyola, Palma de Mallorca, PM, Spain",
                        "layer": "street",
                        "locality": "Palma de Mallorca",
                        "localadmin": "Palma",
                        "region": "Balearic Islands",
                        "country": "Spain",
                    },
                    "geometry": {
                        "coordinates": [2.656009, 39.578893],
                    },
                },
            ]
        }
    )

    snap = AsyncMock(
        return_value={
            "locations": [
                {
                    "location": [2.700683, 39.694754],
                    "snapped_distance": 25.24,
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        result = await resolve_named_location(
            client,
            "Bunyola, Mallorca",
            country="ES",
            focus_lon=2.65,
            focus_lat=39.58,
        )

    assert result.source == "geocode"
    assert result.label == "Bunyola, PM, Spain"
    assert result.original_longitude == 2.700963
    assert result.original_latitude == 39.694825
    assert result.longitude == 2.700683
    assert result.latitude == 39.694754
    assert result.snapped_distance_m == 25.24

    geocode.assert_awaited_once_with(
        "Bunyola, Mallorca",
        size=5,
        country="ES",
        focus_lon=2.65,
        focus_lat=39.58,
    )

    snap.assert_awaited_once_with(
        [[2.700963, 39.694825]],
        profile="cycling-road",
        radius=350.0,
    )


async def test_ambiguous_named_location_is_not_snapped() -> None:
    geocode = AsyncMock(
        return_value={
            "features": [
                {
                    "properties": {
                        "name": "Orient",
                        "label": "Orient, Cadenes Ses, PM, Spain",
                        "layer": "venue",
                        "locality": "Cadenes Ses",
                        "localadmin": "Palma",
                    },
                    "geometry": {
                        "coordinates": [2.752928, 39.514548],
                    },
                },
                {
                    "properties": {
                        "name": "Orient",
                        "label": "Orient, Bunyola, PM, Spain",
                        "layer": "venue",
                        "localadmin": "Bunyola",
                    },
                    "geometry": {
                        "coordinates": [2.761237, 39.733848],
                    },
                },
            ]
        }
    )

    snap = AsyncMock()

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        with pytest.raises(
            LocationResolutionError,
            match="ambiguous",
        ):
            await resolve_named_location(
                client,
                "Orient, Mallorca",
            )

    snap.assert_not_awaited()


async def test_resolve_location_direct_coordinates_bypass_geocoder() -> None:
    geocode = AsyncMock()

    snap = AsyncMock(
        return_value={
            "locations": [
                {
                    "location": [2.631246, 39.590265],
                    "snapped_distance": 102.39,
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        result = await resolve_location(
            client,
            "39.589985,2.630108",
        )

    assert result.source == "coordinates"
    assert result.longitude == 2.631246
    assert result.latitude == 39.590265

    geocode.assert_not_awaited()
