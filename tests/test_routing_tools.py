"""Tests for routing location helpers."""

import base64
import xml.etree.ElementTree as ET
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from intervals_icu_mcp.auth import ICUConfig
from intervals_icu_mcp.openrouteservice_client import OpenRouteServiceClient
from intervals_icu_mcp.tools.routing import (
    CyclingRoute,
    CyclingRouteCandidate,
    CyclingRouteCandidateAnalysis,
    CyclingRouteCandidateSearchResult,
    CyclingRouteQualityMetrics,
    CyclingSessionRequirements,
    GeocodeCandidate,
    LocationResolutionError,
    ResolvedLocation,
    RouteCoordinate,
    RouteExtraDistribution,
    RouteExtraValue,
    RouteParsingError,
    RouteQualityMetrics,
    RouteSessionSegmentAnalysis,
    RouteTimeline,
    RouteTimelinePoint,
    RouteTrainingWindow,
    RouteTrainingWindowRequirements,
    RouteWindowInterruptionMetrics,
    TrainingBlockSequenceAnalysis,
    TrainingBlockSpec,
    analyze_route_cooldown,
    analyze_route_warmup,
    analyze_training_block_sequence,
    analyze_training_window,
    build_cycling_route,
    calculate_cycling_route_quality_metrics,
    calculate_elevation_gain_loss,
    calculate_extra_distribution,
    calculate_extra_distribution_for_geometry_range,
    calculate_route_extra_distributions,
    calculate_route_geometry_overlap_percentage,
    calculate_route_quality_metrics,
    calculate_route_timeline,
    calculate_training_window,
    calculate_training_window_extra_distributions,
    calculate_training_window_interruption_metrics,
    calculate_training_window_quality_metrics,
    cycling_route_to_gpx,
    cycling_session_meets_requirements,
    deduplicate_cycling_route_candidates,
    evaluate_cycling_route_candidates,
    extract_geocode_candidates,
    extract_requested_house_number,
    filter_cycling_route_candidates_by_duration,
    find_cycling_training_route_candidates,
    find_training_block_sequences,
    generate_cycling_route_candidates,
    generate_training_window_candidates,
    geocode_location_candidates,
    name_token_coverage,
    normalize_location_text,
    parse_cycling_route_response,
    parse_lat_lon,
    rank_cycling_route_candidates,
    rank_training_block_sequences,
    resample_route_geometry,
    resolve_coordinate_location,
    resolve_location,
    resolve_named_location,
    select_geocode_candidate,
    serialize_training_block_sequence_analysis,
    snap_coordinate,
    split_location_query,
    timeline_point_at_time,
    validate_cycling_avoid_features,
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
                        "street": "Carrer de Blanquerna",
                        "housenumber": "44",
                        "confidence": 0.95,
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
    assert candidate.street == "Carrer de Blanquerna"
    assert candidate.house_number == "44"
    assert candidate.confidence == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Carrer de Blanquerna, 44, Palma", "44"),
        ("Carrer de Blanquerna 44, Palma", "44"),
        ("Carrer de Blanquerna 44", "44"),
        ("Carrer de Blanquerna, Palma", None),
        ("Ma-10", None),
        ("PM-1", None),
        ("Carretera Ma-10", None),
        ("Road MA-10", None),
        ("Carretera Ma-10, 14, Escorca", None),
        ("Carretera de Soller km 14", None),
        ("Carretera de Soller, km 14", None),
        ("Coll d'Honor", None),
    ],
)
def test_extract_requested_house_number_is_conservative(
    value: str,
    expected: str | None,
) -> None:
    assert extract_requested_house_number(value) == expected


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
    region: str = "Balearic Islands",
    distance_km: float | None = None,
) -> GeocodeCandidate:
    return GeocodeCandidate(
        name=name,
        label=label or name,
        longitude=2.65,
        latitude=39.60,
        layer=layer,
        locality=locality,
        localadmin=localadmin,
        region=region,
        country="Spain",
        distance_km=distance_km,
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


def _geocode_feature(
    *,
    name: str,
    label: str,
    layer: str,
    street: str | None = None,
    house_number: str | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "name": name,
        "label": label,
        "layer": layer,
        "locality": "Palma",
        "localadmin": "Palma",
        "region": "Balearic Islands",
        "country": "Spain",
    }

    if street is not None:
        properties["street"] = street

    if house_number is not None:
        properties["housenumber"] = house_number

    return {
        "properties": properties,
        "geometry": {"coordinates": [2.649397, 39.581232]},
    }


@pytest.mark.parametrize(
    "query",
    [
        "Carrer de Blanquerna, 44, Palma",
        "Carrer de Blanquerna 44, Palma",
        "Carrer de Blanquerna 44",
    ],
)
async def test_numbered_address_requires_and_accepts_exact_house_number(
    query: str,
) -> None:
    geocode = AsyncMock(
        return_value={
            "features": [
                _geocode_feature(
                    name="Carrer de Blanquerna 44",
                    label="Carrer de Blanquerna 44, Palma, PM, Spain",
                    layer="address",
                    street="Carrer de Blanquerna",
                    house_number="44",
                )
            ]
        }
    )
    snap = AsyncMock(
        return_value={
            "locations": [
                {
                    "location": [2.6494, 39.58123],
                    "snapped_distance": 0.4,
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        result = await resolve_named_location(client, query)

    assert result.label == "Carrer de Blanquerna 44, Palma, PM, Spain"
    assert result.original_longitude == pytest.approx(2.649397)
    assert result.original_latitude == pytest.approx(39.581232)
    snap.assert_awaited_once_with(
        [[2.649397, 39.581232]],
        profile="cycling-road",
        radius=350.0,
    )


async def test_numbered_address_rejects_generic_street_without_snapping() -> None:
    geocode = AsyncMock(
        return_value={
            "features": [
                _geocode_feature(
                    name="carrer de Saridakis",
                    label="carrer de Saridakis, Palma, PM, Spain",
                    layer="street",
                    street="carrer de Saridakis",
                )
            ]
        }
    )
    snap = AsyncMock()

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        with pytest.raises(
            LocationResolutionError,
            match="house number '44'",
        ):
            await resolve_named_location(
                client,
                "Carrer de Saridakis, 44, Palma",
            )

    snap.assert_not_awaited()


async def test_numbered_address_rejects_different_house_number() -> None:
    geocode = AsyncMock(
        return_value={
            "features": [
                _geocode_feature(
                    name="Carrer de Blanquerna 42",
                    label="Carrer de Blanquerna 42, Palma, PM, Spain",
                    layer="address",
                    street="Carrer de Blanquerna",
                    house_number="42",
                )
            ]
        }
    )
    snap = AsyncMock()

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        with pytest.raises(
            LocationResolutionError,
            match="house number '44'",
        ):
            await resolve_named_location(
                client,
                "Carrer de Blanquerna, 44, Palma",
            )

    snap.assert_not_awaited()


@pytest.mark.parametrize(
    ("query", "feature"),
    [
        (
            "Carrer de Blanquerna, Palma",
            _geocode_feature(
                name="Carrer de Blanquerna",
                label="Carrer de Blanquerna, Palma, PM, Spain",
                layer="street",
                street="Carrer de Blanquerna",
            ),
        ),
        (
            "Coll d'Honor, Bunyola",
            _geocode_feature(
                name="Coll d'Honor",
                label="Coll d'Honor, Bunyola, PM, Spain",
                layer="venue",
            ),
        ),
    ],
)
async def test_non_address_locations_keep_existing_resolution(
    query: str,
    feature: dict[str, object],
) -> None:
    geocode = AsyncMock(return_value={"features": [feature]})
    snap = AsyncMock(
        return_value={
            "locations": [
                {
                    "location": [2.6494, 39.58123],
                    "snapped_distance": 0.4,
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.geocode = geocode  # type: ignore[method-assign]
        client.snap = snap  # type: ignore[method-assign]

        result = await resolve_named_location(client, query)

    assert result.source == "geocode"
    snap.assert_awaited_once()


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


def test_flat_route_has_zero_smoothed_elevation_change() -> None:
    coordinates = (
        RouteCoordinate(2.60, 39.50, 100.0),
        RouteCoordinate(2.61, 39.50, 100.0),
        RouteCoordinate(2.62, 39.50, 100.0),
    )

    gain, loss = calculate_elevation_gain_loss(coordinates)

    assert gain == pytest.approx(0.0)
    assert loss == pytest.approx(0.0)


def test_sustained_climb_remains_positive_after_smoothing() -> None:
    coordinates = (
        RouteCoordinate(2.600, 39.500, 100.0),
        RouteCoordinate(2.605, 39.500, 150.0),
        RouteCoordinate(2.610, 39.500, 200.0),
        RouteCoordinate(2.615, 39.500, 250.0),
        RouteCoordinate(2.620, 39.500, 300.0),
    )

    gain, loss = calculate_elevation_gain_loss(coordinates)

    assert gain is not None
    assert loss is not None
    assert gain > 150.0
    assert loss == pytest.approx(0.0)


def test_elevation_smoothing_suppresses_short_noise() -> None:
    coordinates = tuple(
        RouteCoordinate(
            2.600 + index * 0.0003,
            39.500,
            100.0 if index % 2 == 0 else 110.0,
        )
        for index in range(20)
    )

    raw_gain = 90.0

    gain, _ = calculate_elevation_gain_loss(coordinates)

    assert gain is not None
    assert gain < raw_gain


def test_missing_elevation_returns_no_gain_or_loss() -> None:
    coordinates = (
        RouteCoordinate(2.60, 39.50, None),
        RouteCoordinate(2.61, 39.50, None),
    )

    assert calculate_elevation_gain_loss(
        coordinates
    ) == (None, None)


def test_parse_cycling_route_response() -> None:
    result = parse_cycling_route_response(
        {
            "features": [
                {
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [2.630000, 39.590000, 100.0],
                            [2.635000, 39.595000, 150.0],
                            [2.640000, 39.600000, 200.0],
                        ],
                    },
                    "properties": {
                        "summary": {
                            "distance": 12345.6,
                            "duration": 2345.7,
                            "ascent": 321.0,
                            "descent": 123.0,
                        },
                        "way_points": [0, 2],
                        "segments": [
                            {
                                "distance": 12345.6,
                                "duration": 2345.7,
                            }
                        ],
                        "extras": {
                            "surface": {
                                "values": [
                                    [0, 2, 3],
                                ]
                            }
                        },
                    },
                }
            ]
        }
    )

    assert isinstance(result, CyclingRoute)
    assert result.distance_m == pytest.approx(12345.6)
    assert result.duration_s == pytest.approx(2345.7)

    assert result.ors_ascent_m == pytest.approx(321.0)
    assert result.ors_descent_m == pytest.approx(123.0)

    assert result.elevation_gain_m is not None
    assert result.elevation_gain_m > 0
    assert result.elevation_loss_m == pytest.approx(0.0)

    assert len(result.geometry) == 3
    assert result.geometry[0].elevation_m == 100.0

    assert result.waypoint_indices == (0, 2)
    assert len(result.segments) == 1
    assert "surface" in result.extras


def test_parser_does_not_use_ors_ascent_as_computed_gain() -> None:
    result = parse_cycling_route_response(
        {
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [2.600, 39.500, 100.0],
                            [2.605, 39.500, 150.0],
                            [2.610, 39.500, 200.0],
                            [2.615, 39.500, 250.0],
                        ]
                    },
                    "properties": {
                        "summary": {
                            "distance": 2000.0,
                            "duration": 300.0,
                            "ascent": 9999.0,
                            "descent": 8888.0,
                        }
                    },
                }
            ]
        }
    )

    assert result.ors_ascent_m == 9999.0
    assert result.elevation_gain_m is not None
    assert result.elevation_gain_m < 9999.0


def test_parse_cycling_route_rejects_invalid_response() -> None:
    with pytest.raises(
        RouteParsingError,
        match="no features",
    ):
        parse_cycling_route_response({})


def _resolved_location(
    label: str,
    longitude: float,
    latitude: float,
) -> ResolvedLocation:
    return ResolvedLocation(
        input_value=label,
        source="geocode",
        label=label,
        original_longitude=longitude,
        original_latitude=latitude,
        longitude=longitude,
        latitude=latitude,
        snapped_distance_m=0.0,
    )


async def test_build_cycling_route_calls_directions_and_parses() -> None:
    start = _resolved_location(
        "Start",
        2.631246,
        39.590265,
    )
    waypoint = _resolved_location(
        "Waypoint",
        2.700683,
        39.694754,
    )

    directions = AsyncMock(
        return_value={
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [2.631246, 39.590265, 61.0],
                            [2.665000, 39.640000, 120.0],
                            [2.700683, 39.694754, 216.0],
                        ]
                    },
                    "properties": {
                        "summary": {
                            "distance": 15850.0,
                            "duration": 2382.0,
                            "ascent": 220.0,
                            "descent": 20.0,
                        },
                        "way_points": [0, 2],
                        "segments": [
                            {
                                "distance": 15850.0,
                                "duration": 2382.0,
                            }
                        ],
                        "extras": {
                            "surface": {"values": [[0, 2, 3]]},
                            "waytype": {"values": [[0, 2, 2]]},
                            "steepness": {"values": [[0, 2, 1]]},
                            "suitability": {"values": [[0, 2, 8]]},
                        },
                    },
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.directions = directions  # type: ignore[method-assign]

        route = await build_cycling_route(
            client,
            [start, waypoint],
        )

    assert route.distance_m == pytest.approx(15850.0)
    assert route.duration_s == pytest.approx(2382.0)
    assert route.waypoint_indices == (0, 2)

    directions.assert_awaited_once_with(
        [
            [2.631246, 39.590265],
            [2.700683, 39.694754],
        ],
        profile="cycling-road",
        elevation=True,
        instructions=True,
        extra_info=[
            "surface",
            "waytype",
            "steepness",
            "suitability",
        ],
    )


def test_resample_route_geometry_interpolates_endpoints() -> None:
    geometry = (
        RouteCoordinate(2.63, 39.59, 100.0),
        RouteCoordinate(2.633, 39.59, 130.0),
    )

    resampled = resample_route_geometry(geometry, spacing_m=100.0)

    assert resampled[0] == geometry[0]
    assert resampled[-1] == geometry[-1]
    assert len(resampled) > 2
    assert resampled[1].elevation_m is not None
    assert 100.0 < resampled[1].elevation_m < 130.0


def test_cycling_route_to_gpx_serializes_track_and_training_waypoints() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    training_window = calculate_training_window(
        route,
        timeline,
        start_time_s=20.0,
        duration_s=20.0,
    )

    content = cycling_route_to_gpx(
        route,
        name="Climb & tempo <session>",
        description="Road route & training block",
        training_window=training_window,
    )

    assert content.startswith(b"<?xml")
    root = ET.fromstring(content)
    namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}

    assert root.tag == "{http://www.topografix.com/GPX/1/1}gpx"
    assert root.attrib == {
        "version": "1.1",
        "creator": "intervals-icu-mcp",
    }
    assert root.findtext("gpx:metadata/gpx:name", namespaces=namespace) == (
        "Climb & tempo <session>"
    )
    assert root.findtext("gpx:metadata/gpx:desc", namespaces=namespace) == (
        "Road route & training block"
    )

    track_points = root.findall(
        "gpx:trk/gpx:trkseg/gpx:trkpt",
        namespace,
    )
    assert len(track_points) == len(route.geometry)
    assert track_points[0].attrib == {
        "lat": str(route.geometry[0].latitude),
        "lon": str(route.geometry[0].longitude),
    }
    assert track_points[0].find("gpx:ele", namespace) is not None

    waypoints = root.findall("gpx:wpt", namespace)
    assert [
        waypoint.findtext("gpx:name", namespaces=namespace)
        for waypoint in waypoints
    ] == ["TRAINING START", "TRAINING END"]


def test_cycling_route_to_gpx_omits_optional_values() -> None:
    base = _route_for_extra_tests({})
    route = CyclingRoute(
        distance_m=base.distance_m,
        duration_s=base.duration_s,
        elevation_gain_m=None,
        elevation_loss_m=None,
        ors_ascent_m=None,
        ors_descent_m=None,
        geometry=tuple(
            RouteCoordinate(point.longitude, point.latitude)
            for point in base.geometry
        ),
        waypoint_indices=base.waypoint_indices,
        segments=base.segments,
        extras=base.extras,
    )

    root = ET.fromstring(cycling_route_to_gpx(route))
    namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}

    assert root.find("gpx:metadata/gpx:desc", namespace) is None
    assert root.findall("gpx:wpt", namespace) == []
    assert root.findall("gpx:trk/gpx:trkseg/gpx:trkpt/gpx:ele", namespace) == []


def test_cycling_route_to_gpx_serializes_training_block_waypoints() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    sequence = find_training_block_sequences(
        route,
        timeline,
        TrainingBlockSpec(
            work_duration_s=20.0,
            repetitions=2,
            recovery_min_s=0.0,
            recovery_max_s=20.0,
        ),
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        step_s=20.0,
    )[0]

    root = ET.fromstring(
        cycling_route_to_gpx(
            route,
            training_block_sequence=sequence,
        )
    )
    namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}

    assert [
        waypoint.findtext("gpx:name", namespaces=namespace)
        for waypoint in root.findall("gpx:wpt", namespace)
    ] == [
        "BLOCK 1 START",
        "BLOCK 1 END",
        "BLOCK 2 START",
        "BLOCK 2 END",
    ]


def test_cycling_route_to_gpx_rejects_invalid_geometry() -> None:
    base = _route_for_extra_tests({})
    empty_route = CyclingRoute(
        distance_m=0.0,
        duration_s=0.0,
        elevation_gain_m=None,
        elevation_loss_m=None,
        ors_ascent_m=None,
        ors_descent_m=None,
        geometry=(),
        waypoint_indices=(),
        segments=(),
        extras={},
    )
    with pytest.raises(ValueError, match="empty route geometry"):
        cycling_route_to_gpx(empty_route)

    invalid_route = CyclingRoute(
        distance_m=base.distance_m,
        duration_s=base.duration_s,
        elevation_gain_m=base.elevation_gain_m,
        elevation_loss_m=base.elevation_loss_m,
        ors_ascent_m=base.ors_ascent_m,
        ors_descent_m=base.ors_descent_m,
        geometry=(RouteCoordinate(float("nan"), 39.59),),
        waypoint_indices=(),
        segments=(),
        extras={},
    )
    with pytest.raises(ValueError, match="must be finite"):
        cycling_route_to_gpx(invalid_route)


def test_deduplicate_cycling_route_candidates_uses_symmetric_overlap() -> None:
    base = _route_for_extra_tests({})

    def route(latitude_offset: float, point_count: int = 4) -> CyclingRoute:
        geometry = tuple(
            RouteCoordinate(
                longitude=2.63 + index * 0.001,
                latitude=39.59 + latitude_offset,
                elevation_m=100.0,
            )
            for index in range(point_count)
        )
        return CyclingRoute(
            distance_m=base.distance_m,
            duration_s=base.duration_s,
            elevation_gain_m=base.elevation_gain_m,
            elevation_loss_m=base.elevation_loss_m,
            ors_ascent_m=base.ors_ascent_m,
            ors_descent_m=base.ors_descent_m,
            geometry=geometry,
            waypoint_indices=base.waypoint_indices,
            segments=base.segments,
            extras=base.extras,
        )

    first_route = route(0.0)
    near_route = route(0.00005)
    far_route = route(0.01)
    partial_route = route(0.0, point_count=2)

    assert calculate_route_geometry_overlap_percentage(
        first_route,
        near_route,
        resample_spacing_m=50.0,
        proximity_m=20.0,
    ) == pytest.approx(100.0)
    assert calculate_route_geometry_overlap_percentage(
        first_route,
        partial_route,
        resample_spacing_m=50.0,
        proximity_m=20.0,
    ) < 90.0

    candidates = tuple(
        CyclingRouteCandidate(
            candidate_id=candidate_id,
            strategy="ors_round_trip",
            seed=index,
            target_distance_m=base.distance_m,
            route=candidate_route,
        )
        for index, (candidate_id, candidate_route) in enumerate(
            (
                ("first", first_route),
                ("near", near_route),
                ("far", far_route),
            )
        )
    )

    deduplicated = deduplicate_cycling_route_candidates(
        candidates,
        overlap_threshold_percentage=90.0,
        resample_spacing_m=50.0,
        proximity_m=20.0,
    )

    assert [candidate.candidate_id for candidate in deduplicated] == [
        "first",
        "far",
    ]


def test_deduplicate_cycling_route_candidates_validates_threshold() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        deduplicate_cycling_route_candidates(
            (),
            overlap_threshold_percentage=101.0,
        )

    with pytest.raises(ValueError, match="resample_spacing_m"):
        deduplicate_cycling_route_candidates((), resample_spacing_m=0.0)

    with pytest.raises(ValueError, match="proximity_m"):
        deduplicate_cycling_route_candidates((), proximity_m=0.0)


def test_filter_cycling_route_candidates_by_duration() -> None:
    base = _route_for_extra_tests({})

    def candidate(candidate_id: str, duration_s: float):
        route = CyclingRoute(
            distance_m=base.distance_m,
            duration_s=duration_s,
            elevation_gain_m=base.elevation_gain_m,
            elevation_loss_m=base.elevation_loss_m,
            ors_ascent_m=base.ors_ascent_m,
            ors_descent_m=base.ors_descent_m,
            geometry=base.geometry,
            waypoint_indices=base.waypoint_indices,
            segments=base.segments,
            extras=base.extras,
        )
        return CyclingRouteCandidate(
            candidate_id=candidate_id,
            strategy="ors_round_trip",
            seed=0,
            target_distance_m=base.distance_m,
            route=route,
            target_duration_s=3600.0,
        )

    close = candidate("close", 3780.0)
    far = candidate("far", 4500.0)

    assert filter_cycling_route_candidates_by_duration(
        (close, far),
        target_duration_s=3600.0,
        max_duration_deviation_percentage=10.0,
    ) == (close,)
    assert filter_cycling_route_candidates_by_duration(
        (close, far),
        target_duration_s=3600.0,
    ) == (close, far)

    with pytest.raises(ValueError, match="target_duration_s"):
        filter_cycling_route_candidates_by_duration((), target_duration_s=0.0)


async def test_generate_cycling_route_candidates_uses_deterministic_seeds() -> None:
    origin = _resolved_location(
        "Start",
        2.631246,
        39.590265,
    )

    def response(distance_m: float) -> dict[str, object]:
        return {
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [2.631246, 39.590265, 61.0],
                            [2.700683, 39.694754, 216.0],
                            [2.631246, 39.590265, 61.0],
                        ]
                    },
                    "properties": {
                        "summary": {
                            "distance": distance_m,
                            "duration": 5000.0,
                        },
                        "way_points": [0, 2],
                    },
                }
            ]
        }

    directions = AsyncMock(
        side_effect=[
            response(49_000.0),
            response(50_000.0),
            response(51_000.0),
        ]
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.directions = directions  # type: ignore[method-assign]

        candidates = await generate_cycling_route_candidates(
            client,
            origin,
            target_distance_m=50_000.0,
            candidate_count=3,
            seed_start=10,
        )

    assert all(
        isinstance(candidate, CyclingRouteCandidate)
        for candidate in candidates
    )
    assert [candidate.candidate_id for candidate in candidates] == [
        "round-trip-1",
        "round-trip-2",
        "round-trip-3",
    ]
    assert [candidate.seed for candidate in candidates] == [10, 11, 12]
    assert [candidate.route.distance_m for candidate in candidates] == [
        49_000.0,
        50_000.0,
        51_000.0,
    ]

    assert directions.await_count == 3

    for call, seed in zip(
        directions.await_args_list,
        (10, 11, 12),
        strict=True,
    ):
        assert call.args[0] == [[2.631246, 39.590265]]
        assert call.kwargs["options"] == {
            "round_trip": {
                "length": 50_000.0,
                "points": 2,
                "seed": seed,
            }
        }


async def test_generate_cycling_route_candidates_filters_distance_deviation() -> None:
    origin = _resolved_location("Start", 2.63, 39.59)

    def response(distance_m: float) -> dict[str, object]:
        return {
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [2.63, 39.59, 100.0],
                            [2.64, 39.60, 110.0],
                        ]
                    },
                    "properties": {
                        "summary": {
                            "distance": distance_m,
                            "duration": 5000.0,
                        }
                    },
                }
            ]
        }

    directions = AsyncMock(
        side_effect=[response(55_000.0), response(100_000.0)]
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.directions = directions  # type: ignore[method-assign]
        candidates = await generate_cycling_route_candidates(
            client,
            origin,
            target_distance_m=50_000.0,
            candidate_count=2,
            max_distance_deviation_percentage=50.0,
        )

    assert [candidate.seed for candidate in candidates] == [0]
    assert candidates[0].route.distance_m == 55_000.0


async def test_generate_cycling_route_candidates_forwards_avoid_features() -> None:
    origin = _resolved_location("Start", 2.63, 39.59)
    directions = AsyncMock(
        return_value={
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [2.63, 39.59, 100.0],
                            [2.64, 39.60, 110.0],
                        ]
                    },
                    "properties": {
                        "summary": {"distance": 10_000.0, "duration": 1_000.0}
                    },
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.directions = directions  # type: ignore[method-assign]
        await generate_cycling_route_candidates(
            client,
            origin,
            target_distance_m=10_000.0,
            candidate_count=1,
            avoid_features=("ferries", "steps"),
        )

    assert directions.await_args.kwargs["options"] == {
        "round_trip": {"length": 10_000.0, "points": 2, "seed": 0},
        "avoid_features": ["ferries", "steps"],
    }


@pytest.mark.parametrize(
    "avoid_features",
    [("highways",), ("steps", "steps")],
)
def test_validate_cycling_avoid_features_rejects_invalid_values(
    avoid_features: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        validate_cycling_avoid_features(avoid_features)


async def test_generate_cycling_route_candidates_validates_inputs() -> None:
    origin = _resolved_location("Start", 2.63, 39.59)

    async with OpenRouteServiceClient(_config()) as client:
        with pytest.raises(ValueError, match="target_distance_m"):
            await generate_cycling_route_candidates(
                client,
                origin,
                target_distance_m=0.0,
                candidate_count=1,
            )

        with pytest.raises(ValueError, match="candidate_count"):
            await generate_cycling_route_candidates(
                client,
                origin,
                target_distance_m=50_000.0,
                candidate_count=0,
            )

        with pytest.raises(ValueError, match="round_trip_points"):
            await generate_cycling_route_candidates(
                client,
                origin,
                target_distance_m=50_000.0,
                candidate_count=1,
                round_trip_points=0,
            )


async def test_build_cycling_route_supports_closed_route() -> None:
    start = _resolved_location(
        "Start",
        2.631246,
        39.590265,
    )
    waypoint = _resolved_location(
        "Waypoint",
        2.700683,
        39.694754,
    )

    directions = AsyncMock(
        return_value={
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [2.631246, 39.590265, 61.0],
                            [2.700683, 39.694754, 216.0],
                            [2.631246, 39.590265, 61.0],
                        ]
                    },
                    "properties": {
                        "summary": {
                            "distance": 30000.0,
                            "duration": 5000.0,
                        },
                        "way_points": [0, 1, 2],
                    },
                }
            ]
        }
    )

    async with OpenRouteServiceClient(_config()) as client:
        client.directions = directions  # type: ignore[method-assign]

        route = await build_cycling_route(
            client,
            [start, waypoint, start],
        )

    assert route.waypoint_indices == (0, 1, 2)

    called_coordinates = directions.await_args.args[0]

    assert called_coordinates[0] == called_coordinates[-1]


async def test_build_cycling_route_requires_two_locations() -> None:
    async with OpenRouteServiceClient(_config()) as client:
        with pytest.raises(
            ValueError,
            match="At least two resolved locations",
        ):
            await build_cycling_route(
                client,
                [
                    _resolved_location(
                        "Start",
                        2.631246,
                        39.590265,
                    )
                ],
            )


def test_exact_locality_can_use_strong_proximity_tiebreak() -> None:
    candidates = [
        _candidate(
            "Santa Maria del Camí",
            label="Santa Maria del Camí, PM, Spain",
            layer="locality",
            locality="Santa Maria del Camí",
            localadmin="Santa Maria del Camí",
            distance_km=13.946,
        ),
        _candidate(
            "Santa Maria del Camí",
            label="Santa Maria del Camí, PM, Spain",
            layer="venue",
            locality="Santa Maria del Camí",
            localadmin="Santa Maria del Camí",
            distance_km=14.134,
        ),
        _candidate(
            "Santa María del Camí",
            label="Santa María del Camí, CT, Spain",
            layer="locality",
            localadmin="Veciana",
            region="Barcelona",
            distance_km=247.077,
        ),
    ]

    selected = select_geocode_candidate(
        "Santa Maria del Camí, Mallorca",
        candidates,
    )

    assert selected.layer == "locality"
    assert selected.region == "Balearic Islands"
    assert selected.distance_km == pytest.approx(13.946)


def test_proximity_does_not_resolve_close_locality_matches() -> None:
    candidates = [
        _candidate(
            "Example",
            layer="locality",
            distance_km=12.0,
        ),
        _candidate(
            "Example",
            layer="locality",
            region="Another Region",
            distance_km=20.0,
        ),
    ]

    with pytest.raises(
        LocationResolutionError,
        match="ambiguous",
    ):
        select_geocode_candidate(
            "Example",
            candidates,
        )


def _route_for_extra_tests(
    extras: dict[str, object],
) -> CyclingRoute:
    return CyclingRoute(
        distance_m=2000.0,
        duration_s=300.0,
        elevation_gain_m=0.0,
        elevation_loss_m=0.0,
        ors_ascent_m=None,
        ors_descent_m=None,
        geometry=(
            RouteCoordinate(2.600, 39.500, 100.0),
            RouteCoordinate(2.610, 39.500, 100.0),
            RouteCoordinate(2.620, 39.500, 100.0),
        ),
        waypoint_indices=(0, 2),
        segments=(),
        extras=extras,
    )


def test_extra_distribution_uses_geometry_distance() -> None:
    route = _route_for_extra_tests(
        {
            "surface": {
                "values": [
                    [0, 1, 3],
                    [1, 2, 1],
                ]
            }
        }
    )

    result = calculate_extra_distribution(
        route,
        "surface",
    )

    assert isinstance(result, RouteExtraDistribution)
    assert result.unclassified_distance_m == pytest.approx(0.0)

    values = {
        item.value: item
        for item in result.values
    }

    assert isinstance(values[3], RouteExtraValue)

    assert values[3].percentage == pytest.approx(
        50.0,
        abs=0.1,
    )
    assert values[1].percentage == pytest.approx(
        50.0,
        abs=0.1,
    )


def test_extra_distribution_reports_unclassified_distance() -> None:
    route = _route_for_extra_tests(
        {
            "surface": {
                "values": [
                    [0, 1, 3],
                ]
            }
        }
    )

    result = calculate_extra_distribution(
        route,
        "surface",
    )

    assert result.classified_distance_m == pytest.approx(
        result.geometry_distance_m / 2.0,
        rel=0.01,
    )
    assert result.unclassified_distance_m == pytest.approx(
        result.geometry_distance_m / 2.0,
        rel=0.01,
    )


def test_missing_extra_is_fully_unclassified() -> None:
    route = _route_for_extra_tests({})

    result = calculate_extra_distribution(
        route,
        "surface",
    )

    assert result.values == ()
    assert result.classified_distance_m == 0.0
    assert result.unclassified_distance_m == pytest.approx(
        result.geometry_distance_m
    )


def test_extra_distribution_rejects_invalid_indices() -> None:
    route = _route_for_extra_tests(
        {
            "surface": {
                "values": [
                    [0, 99, 3],
                ]
            }
        }
    )

    with pytest.raises(
        RouteParsingError,
        match="out-of-range surface interval",
    ):
        calculate_extra_distribution(
            route,
            "surface",
        )


def test_calculate_all_route_extra_distributions() -> None:
    route = _route_for_extra_tests(
        {
            "surface": {
                "values": [[0, 2, 3]]
            },
            "waytype": {
                "values": [[0, 2, 2]]
            },
            "steepness": {
                "values": [[0, 2, 1]]
            },
            "suitability": {
                "values": [[0, 2, 8]]
            },
        }
    )

    distributions = calculate_route_extra_distributions(
        route
    )

    assert set(distributions) == {
        "surface",
        "waytype",
        "steepness",
        "suitability",
    }

    assert distributions["surface"].values[0].value == 3
    assert distributions["waytype"].values[0].value == 2
    assert distributions["steepness"].values[0].value == 1
    assert distributions["suitability"].values[0].value == 8


def test_calculate_route_quality_metrics() -> None:
    route = _route_for_extra_tests(
        {
            "surface": {
                "values": [
                    [0, 1, 3],
                    [1, 2, 0],
                ]
            },
            "waytype": {
                "values": [
                    [0, 1, 2],
                    [1, 2, 7],
                ]
            },
            "suitability": {
                "values": [
                    [0, 1, 8],
                    [1, 2, 6],
                ]
            },
            "steepness": {
                "values": [
                    [0, 1, 4],
                    [1, 2, -4],
                ]
            },
        }
    )

    metrics = calculate_route_quality_metrics(route)

    assert isinstance(metrics, RouteQualityMetrics)

    assert metrics.asphalt_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )
    assert metrics.unknown_surface_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )

    assert metrics.road_or_cycleway_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )
    assert metrics.footway_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )

    assert metrics.suitability_7_plus_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )
    assert metrics.suitability_8_plus_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )

    assert metrics.incline_7_plus_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )
    assert metrics.incline_10_plus_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )

    assert metrics.decline_7_plus_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )
    assert metrics.decline_10_plus_percentage == pytest.approx(
        50.0,
        abs=0.1,
    )


def _route_for_timeline_tests(
    steps: list[dict[str, object]],
) -> CyclingRoute:
    return CyclingRoute(
        distance_m=3000.0,
        duration_s=sum(
            float(step.get("duration", 0.0))
            for step in steps
        ),
        elevation_gain_m=30.0,
        elevation_loss_m=0.0,
        ors_ascent_m=None,
        ors_descent_m=None,
        geometry=(
            RouteCoordinate(2.600, 39.500, 100.0),
            RouteCoordinate(2.610, 39.500, 110.0),
            RouteCoordinate(2.620, 39.500, 120.0),
            RouteCoordinate(2.630, 39.500, 130.0),
        ),
        waypoint_indices=(0, 3),
        segments=(
            {
                "steps": steps,
            },
        ),
        extras={},
    )


def test_calculate_route_timeline_covers_geometry() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 10.0,
                "way_points": [0, 1],
            },
            {
                "duration": 20.0,
                "way_points": [1, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    assert isinstance(timeline, RouteTimeline)
    assert len(timeline.points) == len(route.geometry)

    assert timeline.points[0] == RouteTimelinePoint(
        geometry_index=0,
        distance_m=0.0,
        time_s=0.0,
        elevation_m=100.0,
    )

    assert timeline.points[1].time_s == pytest.approx(10.0)
    assert timeline.points[-1].time_s == pytest.approx(30.0)
    assert timeline.duration_s == pytest.approx(30.0)


def test_route_timeline_uses_step_duration() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 5.0,
                "way_points": [0, 1],
            },
            {
                "duration": 25.0,
                "way_points": [1, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    assert timeline.points[1].time_s == pytest.approx(5.0)
    assert timeline.points[-1].time_s == pytest.approx(30.0)


def test_route_timeline_interpolates_inside_step() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    assert timeline.points[1].time_s == pytest.approx(
        10.0,
        abs=0.1,
    )
    assert timeline.points[2].time_s == pytest.approx(
        20.0,
        abs=0.1,
    )


def test_route_timeline_rejects_non_contiguous_steps() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 10.0,
                "way_points": [0, 1],
            },
            {
                "duration": 10.0,
                "way_points": [2, 3],
            },
        ]
    )

    with pytest.raises(
        RouteParsingError,
        match="non-contiguous route steps",
    ):
        calculate_route_timeline(route)


def test_route_timeline_rejects_incomplete_geometry() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 10.0,
                "way_points": [0, 2],
            },
        ]
    )

    with pytest.raises(
        RouteParsingError,
        match="do not cover the full geometry",
    ):
        calculate_route_timeline(route)


def test_timeline_point_at_time_finds_nearest_point() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    point = timeline_point_at_time(
        timeline,
        19.0,
    )

    assert point.geometry_index == 2


def test_timeline_point_at_time_handles_boundaries() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    assert timeline_point_at_time(
        timeline,
        0.0,
    ).geometry_index == 0

    assert timeline_point_at_time(
        timeline,
        30.0,
    ).geometry_index == 3


def test_timeline_point_at_time_rejects_out_of_range_time() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        timeline_point_at_time(
            timeline,
            -1.0,
        )

    with pytest.raises(
        ValueError,
        match="exceeds route timeline duration",
    ):
        timeline_point_at_time(
            timeline,
            31.0,
        )


def test_calculate_training_window_uses_timeline_boundaries() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=10.0,
        duration_s=20.0,
    )

    assert isinstance(window, RouteTrainingWindow)
    assert window.start.geometry_index == 1
    assert window.end.geometry_index == 3

    assert window.duration_s == pytest.approx(20.0)
    assert window.distance_m == pytest.approx(
        timeline.points[3].distance_m
        - timeline.points[1].distance_m
    )

    assert window.net_elevation_gain_m == pytest.approx(
        20.0
    )


def test_calculate_training_window_reports_elevation_metrics() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=30.0,
    )

    assert window.elevation_gain_m is not None
    assert window.elevation_loss_m is not None
    assert window.elevation_gain_m > 0
    assert window.elevation_loss_m >= 0


def test_calculate_training_window_rejects_invalid_duration() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_training_window(
            route,
            timeline,
            start_time_s=0.0,
            duration_s=0.0,
        )


def test_calculate_training_window_rejects_window_past_route_end() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    with pytest.raises(
        ValueError,
        match="exceeds route timeline duration",
    ):
        calculate_training_window(
            route,
            timeline,
            start_time_s=20.0,
            duration_s=20.0,
        )


def test_extra_distribution_for_geometry_range_clips_values() -> None:
    route = _route_for_extra_tests(
        {
            "surface": {
                "values": [
                    [0, 1, 3],
                    [1, 2, 0],
                ]
            }
        }
    )

    result = calculate_extra_distribution_for_geometry_range(
        route,
        "surface",
        start_index=0,
        end_index=1,
    )

    assert result.geometry_distance_m > 0
    assert result.unclassified_distance_m == pytest.approx(0.0)
    assert len(result.values) == 1
    assert result.values[0].value == 3
    assert result.values[0].percentage == pytest.approx(100.0)


def test_extra_distribution_for_geometry_range_reports_missing_extra() -> None:
    route = _route_for_extra_tests({})

    result = calculate_extra_distribution_for_geometry_range(
        route,
        "surface",
        start_index=0,
        end_index=1,
    )

    assert result.values == ()
    assert result.classified_distance_m == 0.0
    assert result.unclassified_distance_m == pytest.approx(
        result.geometry_distance_m
    )


def test_training_window_extra_distributions_returns_all_extras() -> None:
    extras = {
        "surface": {"values": [[0, 3, 3]]},
        "waytype": {"values": [[0, 3, 2]]},
        "steepness": {"values": [[0, 3, 1]]},
        "suitability": {"values": [[0, 3, 8]]},
    }

    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    route = CyclingRoute(
        distance_m=route.distance_m,
        duration_s=route.duration_s,
        elevation_gain_m=route.elevation_gain_m,
        elevation_loss_m=route.elevation_loss_m,
        ors_ascent_m=route.ors_ascent_m,
        ors_descent_m=route.ors_descent_m,
        geometry=route.geometry,
        waypoint_indices=route.waypoint_indices,
        segments=route.segments,
        extras=extras,
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=30.0,
    )

    distributions = calculate_training_window_extra_distributions(
        route,
        window,
    )

    assert set(distributions) == {
        "surface",
        "waytype",
        "steepness",
        "suitability",
    }

    for distribution in distributions.values():
        assert distribution.unclassified_distance_m == pytest.approx(0.0)
        assert distribution.values[0].percentage == pytest.approx(100.0)


def test_training_window_quality_metrics() -> None:
    base_route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    route = CyclingRoute(
        distance_m=base_route.distance_m,
        duration_s=base_route.duration_s,
        elevation_gain_m=base_route.elevation_gain_m,
        elevation_loss_m=base_route.elevation_loss_m,
        ors_ascent_m=base_route.ors_ascent_m,
        ors_descent_m=base_route.ors_descent_m,
        geometry=base_route.geometry,
        waypoint_indices=base_route.waypoint_indices,
        segments=base_route.segments,
        extras={
            "surface": {"values": [[0, 3, 3]]},
            "waytype": {"values": [[0, 3, 2]]},
            "steepness": {"values": [[0, 3, 4]]},
            "suitability": {"values": [[0, 3, 8]]},
        },
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=30.0,
    )

    metrics = calculate_training_window_quality_metrics(
        route,
        window,
    )

    assert metrics.asphalt_percentage == pytest.approx(100.0)
    assert metrics.road_or_cycleway_percentage == pytest.approx(100.0)
    assert metrics.suitability_7_plus_percentage == pytest.approx(100.0)
    assert metrics.suitability_8_plus_percentage == pytest.approx(100.0)
    assert metrics.incline_7_plus_percentage == pytest.approx(100.0)
    assert metrics.incline_10_plus_percentage == pytest.approx(100.0)
    assert metrics.decline_7_plus_percentage == pytest.approx(0.0)


def test_training_window_interruption_metrics() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 10.0,
                "way_points": [0, 1],
                "type": 5,
            },
            {
                "duration": 10.0,
                "way_points": [1, 2],
                "type": 7,
            },
            {
                "duration": 10.0,
                "way_points": [2, 3],
                "type": 3,
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=30.0,
    )

    metrics = calculate_training_window_interruption_metrics(
        route,
        window,
    )

    assert isinstance(
        metrics,
        RouteWindowInterruptionMetrics,
    )

    # The type=5 maneuver starts exactly at the window boundary,
    # so it is deliberately excluded.
    assert metrics.maneuver_count == 2
    assert metrics.sharp_turn_count == 1
    assert metrics.roundabout_count == 1
    assert metrics.u_turn_count == 0


def test_training_window_interruption_metrics_ignores_non_maneuvers() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 10.0,
                "way_points": [0, 1],
                "type": 11,
            },
            {
                "duration": 10.0,
                "way_points": [1, 2],
                "type": 6,
            },
            {
                "duration": 10.0,
                "way_points": [2, 3],
                "type": 10,
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=30.0,
    )

    metrics = calculate_training_window_interruption_metrics(
        route,
        window,
    )

    assert metrics.maneuver_count == 0
    assert metrics.sharp_turn_count == 0
    assert metrics.roundabout_count == 0
    assert metrics.u_turn_count == 0


def test_training_window_interruption_metrics_counts_u_turn() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 10.0,
                "way_points": [0, 1],
                "type": 6,
            },
            {
                "duration": 10.0,
                "way_points": [1, 2],
                "type": 9,
            },
            {
                "duration": 10.0,
                "way_points": [2, 3],
                "type": 6,
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=30.0,
    )

    metrics = calculate_training_window_interruption_metrics(
        route,
        window,
    )

    assert metrics.maneuver_count == 1
    assert metrics.u_turn_count == 1


def test_generate_training_window_candidates() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 60.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    candidates = generate_training_window_candidates(
        route,
        timeline,
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        duration_s=20.0,
        step_s=20.0,
    )

    assert len(candidates) == 3

    assert candidates[0].start.time_s == pytest.approx(
        0.0,
        abs=0.1,
    )
    assert candidates[-1].start.time_s == pytest.approx(
        40.0,
        abs=0.1,
    )


def test_generate_training_window_candidates_skips_past_route_end() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    candidates = generate_training_window_candidates(
        route,
        timeline,
        start_time_min_s=0.0,
        start_time_max_s=20.0,
        duration_s=20.0,
        step_s=10.0,
    )

    assert len(candidates) == 2


def test_generate_training_window_candidates_rejects_invalid_step() -> None:
    route = _route_for_timeline_tests(
        [
            {
                "duration": 30.0,
                "way_points": [0, 3],
            },
        ]
    )

    timeline = calculate_route_timeline(route)

    with pytest.raises(
        ValueError,
        match="step_s must be greater than zero",
    ):
        generate_training_window_candidates(
            route,
            timeline,
            start_time_min_s=0.0,
            start_time_max_s=10.0,
            duration_s=10.0,
            step_s=0.0,
        )


def test_analyze_training_window_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = object()
    window_1 = object()
    window_2 = object()
    quality = object()
    interruptions = object()

    def fake_quality(
        _route: object,
        _window: object,
    ) -> object:
        return quality

    def fake_interruptions(
        _route: object,
        _window: object,
    ) -> object:
        return interruptions

    monkeypatch.setattr(
        routing,
        "calculate_training_window_quality_metrics",
        fake_quality,
    )
    monkeypatch.setattr(
        routing,
        "calculate_training_window_interruption_metrics",
        fake_interruptions,
    )

    analyses = routing.analyze_training_window_candidates(
        route,
        (window_1, window_2),
    )

    assert len(analyses) == 2

    assert analyses[0].window is window_1
    assert analyses[0].quality is quality
    assert analyses[0].interruptions is interruptions

    assert analyses[1].window is window_2
    assert analyses[1].quality is quality
    assert analyses[1].interruptions is interruptions


def test_rank_training_window_analyses_prefers_climbing_balance() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    def analysis(
        gain: float,
        loss: float,
        *,
        asphalt: float = 100.0,
        road: float = 100.0,
        suitability: float = 100.0,
        maneuvers: int = 0,
        roundabouts: int = 0,
    ) -> object:
        return SimpleNamespace(
            window=SimpleNamespace(
                elevation_gain_m=gain,
                elevation_loss_m=loss,
            ),
            quality=SimpleNamespace(
                asphalt_percentage=asphalt,
                road_or_cycleway_percentage=road,
                suitability_7_plus_percentage=suitability,
            ),
            interruptions=SimpleNamespace(
                maneuver_count=maneuvers,
                roundabout_count=roundabouts,
            ),
        )

    clean_climb = analysis(489.0, 4.6)
    later_window = analysis(482.7, 23.6)

    ranked = routing.rank_training_window_analyses(
        (later_window, clean_climb),
    )

    assert ranked == (clean_climb, later_window)


def test_rank_training_window_analyses_uses_quality_as_tiebreaker() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    def analysis(asphalt: float) -> object:
        return SimpleNamespace(
            window=SimpleNamespace(
                elevation_gain_m=400.0,
                elevation_loss_m=10.0,
            ),
            quality=SimpleNamespace(
                asphalt_percentage=asphalt,
                road_or_cycleway_percentage=100.0,
                suitability_7_plus_percentage=100.0,
            ),
            interruptions=SimpleNamespace(
                maneuver_count=0,
                roundabout_count=0,
            ),
        )

    lower_quality = analysis(95.0)
    higher_quality = analysis(99.0)

    ranked = routing.rank_training_window_analyses(
        (lower_quality, higher_quality),
    )

    assert ranked == (higher_quality, lower_quality)


def test_rank_training_window_analyses_requires_elevation() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        window=SimpleNamespace(
            elevation_gain_m=None,
            elevation_loss_m=None,
        ),
        quality=SimpleNamespace(
            asphalt_percentage=100.0,
            road_or_cycleway_percentage=100.0,
            suitability_7_plus_percentage=100.0,
        ),
        interruptions=SimpleNamespace(
            maneuver_count=0,
            roundabout_count=0,
        ),
    )

    with pytest.raises(
        routing.RouteParsingError,
        match="requires elevation gain and loss",
    ):
        routing.rank_training_window_analyses((analysis,))


def test_find_best_training_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = object()
    timeline = object()

    window_1 = object()
    window_2 = object()

    analysis_1 = object()
    analysis_2 = object()

    monkeypatch.setattr(
        routing,
        "generate_training_window_candidates",
        lambda *_args, **_kwargs: (window_1, window_2),
    )
    monkeypatch.setattr(
        routing,
        "analyze_training_window_candidates",
        lambda *_args, **_kwargs: (analysis_1, analysis_2),
    )
    monkeypatch.setattr(
        routing,
        "rank_training_window_analyses",
        lambda analyses: (analyses[1], analyses[0]),
    )

    best = routing.find_best_training_window(
        route,
        timeline,
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        duration_s=1800.0,
        step_s=60.0,
    )

    assert best is analysis_2


def test_find_best_training_window_returns_none_without_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    monkeypatch.setattr(
        routing,
        "generate_training_window_candidates",
        lambda *_args, **_kwargs: (),
    )

    best = routing.find_best_training_window(
        object(),
        object(),
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        duration_s=1800.0,
    )

    assert best is None


def test_find_best_training_windows_by_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = object()
    timeline = object()

    result_20 = object()
    result_30 = object()
    result_40 = object()

    results_by_duration = {
        1200.0: result_20,
        1800.0: result_30,
        2400.0: result_40,
    }

    def fake_find_best(
        _route: object,
        _timeline: object,
        *,
        start_time_min_s: float,
        start_time_max_s: float,
        duration_s: float,
        step_s: float,
    ) -> object:
        assert start_time_min_s == 1200.0
        assert start_time_max_s == 1800.0
        assert step_s == 60.0
        return results_by_duration[duration_s]

    monkeypatch.setattr(
        routing,
        "find_best_training_window",
        fake_find_best,
    )

    results = routing.find_best_training_windows_by_duration(
        route,
        timeline,
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1200.0, 1800.0, 2400.0),
        step_s=60.0,
    )

    assert results == (
        result_20,
        result_30,
        result_40,
    )


def test_find_best_training_windows_by_duration_skips_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    result_20 = object()
    calls = 0

    def fake_find_best(
        *_args: object,
        duration_s: float,
        **_kwargs: object,
    ) -> object | None:
        nonlocal calls
        calls += 1

        if duration_s == 1200.0:
            return result_20

        return None

    monkeypatch.setattr(
        routing,
        "find_best_training_window",
        fake_find_best,
    )

    results = routing.find_best_training_windows_by_duration(
        object(),
        object(),
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1200.0, 2400.0),
    )

    assert calls == 2
    assert results == (result_20,)


def test_find_best_training_windows_by_duration_empty() -> None:
    import intervals_icu_mcp.tools.routing as routing

    results = routing.find_best_training_windows_by_duration(
        object(),
        object(),
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(),
    )

    assert results == ()


def test_training_window_comparison_metrics() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        window=SimpleNamespace(
            elevation_gain_m=500.0,
            elevation_loss_m=20.0,
            duration_s=1800.0,
            distance_m=10000.0,
        ),
        interruptions=SimpleNamespace(
            maneuver_count=3,
        ),
    )

    metrics = routing.calculate_training_window_comparison_metrics(
        analysis,
    )

    assert metrics.elevation_gain_rate_m_per_hour == pytest.approx(
        1000.0
    )
    assert metrics.elevation_loss_rate_m_per_hour == pytest.approx(
        40.0
    )
    assert metrics.climbing_balance_rate_m_per_hour == pytest.approx(
        960.0
    )
    assert metrics.climbing_balance_gradient_percentage == pytest.approx(
        4.8
    )
    assert metrics.maneuvers_per_hour == pytest.approx(
        6.0
    )


def _route_for_session_segment_tests() -> CyclingRoute:
    base_route = _route_for_timeline_tests(
        [
            {
                "duration": 20.0,
                "way_points": [0, 1],
                "type": 6,
            },
            {
                "duration": 20.0,
                "way_points": [1, 2],
                "type": 2,
            },
            {
                "duration": 20.0,
                "way_points": [2, 3],
                "type": 7,
            },
        ]
    )

    return CyclingRoute(
        distance_m=base_route.distance_m,
        duration_s=base_route.duration_s,
        elevation_gain_m=base_route.elevation_gain_m,
        elevation_loss_m=base_route.elevation_loss_m,
        ors_ascent_m=base_route.ors_ascent_m,
        ors_descent_m=base_route.ors_descent_m,
        geometry=base_route.geometry,
        waypoint_indices=base_route.waypoint_indices,
        segments=base_route.segments,
        extras={
            "surface": {"values": [[0, 3, 3]]},
            "waytype": {"values": [[0, 3, 2]]},
            "steepness": {"values": [[0, 3, 1]]},
            "suitability": {"values": [[0, 3, 8]]},
        },
    )


def test_analyze_route_warmup_reuses_window_metrics() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    training_window = calculate_training_window(
        route,
        timeline,
        start_time_s=20.0,
        duration_s=20.0,
    )

    warmup = analyze_route_warmup(
        route,
        timeline,
        training_window,
    )

    assert isinstance(warmup, RouteSessionSegmentAnalysis)
    assert warmup.segment.start.time_s == pytest.approx(0.0)
    assert warmup.segment.end.time_s == pytest.approx(20.0)
    assert warmup.segment.duration_s == pytest.approx(20.0)
    assert warmup.segment.distance_m > 0
    assert warmup.segment.elevation_gain_m is not None
    assert warmup.segment.elevation_loss_m is not None
    assert warmup.quality.asphalt_percentage == pytest.approx(100.0)
    assert warmup.quality.road_or_cycleway_percentage == pytest.approx(100.0)
    assert warmup.quality.footway_percentage == pytest.approx(0.0)
    assert warmup.quality.suitability_7_plus_percentage == pytest.approx(100.0)
    assert warmup.comparison.elevation_gain_rate_m_per_hour >= 0
    assert warmup.comparison.elevation_loss_rate_m_per_hour >= 0
    assert warmup.comparison.maneuvers_per_hour >= 0


def test_analyze_route_warmup_returns_none_at_route_start() -> None:
    route = _route_for_timeline_tests(
        [{"duration": 60.0, "way_points": [0, 3]}]
    )
    timeline = calculate_route_timeline(route)
    training_window = calculate_training_window(
        route,
        timeline,
        start_time_s=0.0,
        duration_s=20.0,
    )

    assert analyze_route_warmup(
        route,
        timeline,
        training_window,
    ) is None


def test_analyze_route_cooldown_reuses_window_metrics() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    training_window = calculate_training_window(
        route,
        timeline,
        start_time_s=20.0,
        duration_s=20.0,
    )

    cooldown = analyze_route_cooldown(
        route,
        timeline,
        training_window,
    )

    assert isinstance(cooldown, RouteSessionSegmentAnalysis)
    assert cooldown.segment.start.time_s == pytest.approx(40.0)
    assert cooldown.segment.end.time_s == pytest.approx(60.0)
    assert cooldown.segment.duration_s == pytest.approx(20.0)
    assert cooldown.segment.distance_m > 0
    assert cooldown.segment.elevation_gain_m is not None
    assert cooldown.segment.elevation_loss_m is not None
    assert cooldown.quality.asphalt_percentage == pytest.approx(100.0)
    assert cooldown.quality.road_or_cycleway_percentage == pytest.approx(100.0)
    assert cooldown.quality.footway_percentage == pytest.approx(0.0)
    assert cooldown.quality.suitability_7_plus_percentage == pytest.approx(100.0)
    assert cooldown.comparison.elevation_gain_rate_m_per_hour >= 0
    assert cooldown.comparison.elevation_loss_rate_m_per_hour >= 0
    assert cooldown.comparison.maneuvers_per_hour >= 0


def test_analyze_route_cooldown_returns_none_at_route_end() -> None:
    route = _route_for_timeline_tests(
        [{"duration": 60.0, "way_points": [0, 3]}]
    )
    timeline = calculate_route_timeline(route)
    training_window = calculate_training_window(
        route,
        timeline,
        start_time_s=40.0,
        duration_s=20.0,
    )

    assert analyze_route_cooldown(
        route,
        timeline,
        training_window,
    ) is None


def test_calculate_cycling_route_quality_metrics() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)

    metrics = calculate_cycling_route_quality_metrics(route, timeline)

    assert isinstance(metrics, CyclingRouteQualityMetrics)
    assert metrics.total_distance_m == pytest.approx(route.distance_m)
    assert metrics.total_duration_s == pytest.approx(route.duration_s)
    assert metrics.elevation_gain_m == pytest.approx(route.elevation_gain_m)
    assert metrics.elevation_loss_m == pytest.approx(route.elevation_loss_m)
    assert metrics.asphalt_percentage == pytest.approx(100.0)
    assert metrics.unknown_surface_percentage == pytest.approx(0.0)
    assert metrics.road_or_cycleway_percentage == pytest.approx(100.0)
    assert metrics.footway_percentage == pytest.approx(0.0)
    assert metrics.suitability_7_plus_percentage == pytest.approx(100.0)
    assert metrics.suitability_8_plus_percentage == pytest.approx(100.0)
    assert metrics.incline_7_plus_percentage == pytest.approx(0.0)
    assert metrics.incline_10_plus_percentage == pytest.approx(0.0)
    assert metrics.decline_7_plus_percentage == pytest.approx(0.0)
    assert metrics.decline_10_plus_percentage == pytest.approx(0.0)
    assert metrics.maneuver_count == 2
    assert metrics.maneuver_rate_per_hour == pytest.approx(120.0)
    assert metrics.roundabout_count == 1
    assert metrics.sharp_turn_count == 1


def test_analyze_training_block_sequence_aggregates_work_and_recovery() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    work_blocks = tuple(
        analyze_training_window(
            route,
            calculate_training_window(
                route,
                timeline,
                start_time_s=start_time_s,
                duration_s=20.0,
            ),
        )
        for start_time_s in (0.0, 40.0)
    )
    spec = TrainingBlockSpec(
        work_duration_s=20.0,
        repetitions=2,
        recovery_min_s=19.0,
        recovery_max_s=21.0,
    )

    sequence = analyze_training_block_sequence(
        route,
        timeline,
        work_blocks,
        spec,
    )

    assert isinstance(sequence, TrainingBlockSequenceAnalysis)
    assert sequence.spec is spec
    assert sequence.work_blocks == work_blocks
    assert len(sequence.recoveries) == 1
    assert all(recovery is not None for recovery in sequence.recoveries)
    assert sequence.total_work_duration_s == pytest.approx(40.0)
    assert sequence.total_distance_m == pytest.approx(
        sum(block.window.distance_m for block in work_blocks)
    )
    assert sequence.total_elevation_gain_m == pytest.approx(
        sum(block.window.elevation_gain_m or 0.0 for block in work_blocks)
    )
    assert sequence.total_elevation_loss_m == pytest.approx(
        sum(block.window.elevation_loss_m or 0.0 for block in work_blocks)
    )
    assert sequence.climbing_balance_m == pytest.approx(
        sequence.total_elevation_gain_m - sequence.total_elevation_loss_m
    )
    assert sequence.total_maneuver_count == sum(
        block.interruptions.maneuver_count for block in work_blocks
    )
    assert sequence.elevation_gain_rate_range_m_per_hour >= 0.0


def test_analyze_training_block_sequence_validates_shape_and_recovery() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)

    def block(start_time_s: float, duration_s: float = 20.0):
        return analyze_training_window(
            route,
            calculate_training_window(
                route,
                timeline,
                start_time_s=start_time_s,
                duration_s=duration_s,
            ),
        )

    with pytest.raises(ValueError, match="count"):
        analyze_training_block_sequence(
            route,
            timeline,
            (block(10.0),),
            TrainingBlockSpec(work_duration_s=20.0, repetitions=2),
        )

    with pytest.raises(ValueError, match="duration does not match"):
        analyze_training_block_sequence(
            route,
            timeline,
            (block(0.0, 20.0),),
            TrainingBlockSpec(work_duration_s=60.0, repetitions=1),
        )

    with pytest.raises(ValueError, match="must not overlap"):
        analyze_training_block_sequence(
            route,
            timeline,
            (block(40.0), block(0.0)),
            TrainingBlockSpec(work_duration_s=20.0, repetitions=2),
        )

    with pytest.raises(ValueError, match="shorter"):
        analyze_training_block_sequence(
            route,
            timeline,
            (block(0.0), block(20.0)),
            TrainingBlockSpec(
                work_duration_s=20.0,
                repetitions=2,
                recovery_min_s=10.0,
            ),
        )

    with pytest.raises(ValueError, match="longer"):
        analyze_training_block_sequence(
            route,
            timeline,
            (block(0.0), block(40.0)),
            TrainingBlockSpec(
                work_duration_s=20.0,
                repetitions=2,
                recovery_max_s=10.0,
            ),
        )


def test_analyze_training_block_sequence_applies_work_eligibility() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    analysis = analyze_training_window(
        route,
        calculate_training_window(
            route,
            timeline,
            start_time_s=0.0,
            duration_s=20.0,
        ),
    )
    analysis = replace(
        analysis,
        quality=replace(analysis.quality, asphalt_percentage=50.0),
    )

    with pytest.raises(ValueError, match="eligibility requirements"):
        analyze_training_block_sequence(
            route,
            timeline,
            (analysis,),
            TrainingBlockSpec(work_duration_s=20.0, repetitions=1),
            requirements=RouteTrainingWindowRequirements(
                min_asphalt_percentage=90.0,
            ),
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (TrainingBlockSpec(0.0, 1), "work_duration_s"),
        (TrainingBlockSpec(10.0, 0), "repetitions"),
        (TrainingBlockSpec(10.0, 2, recovery_min_s=-1.0), "recovery_min_s"),
        (TrainingBlockSpec(10.0, 2, recovery_max_s=-1.0), "recovery_max_s"),
        (
            TrainingBlockSpec(
                10.0,
                2,
                recovery_min_s=20.0,
                recovery_max_s=10.0,
            ),
            "recovery_max_s",
        ),
    ],
)
def test_analyze_training_block_sequence_validates_spec(
    spec: TrainingBlockSpec,
    message: str,
) -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)

    with pytest.raises(ValueError, match=message):
        analyze_training_block_sequence(route, timeline, (), spec)


def test_find_training_block_sequences_uses_bounded_backtracking() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    spec = TrainingBlockSpec(
        work_duration_s=20.0,
        repetitions=2,
        recovery_min_s=0.0,
        recovery_max_s=20.0,
    )

    sequences = find_training_block_sequences(
        route,
        timeline,
        spec,
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        step_s=20.0,
        max_sequences=2,
    )

    assert len(sequences) == 2
    assert [
        tuple(block.window.start.time_s for block in sequence.work_blocks)
        for sequence in sequences
    ] == [(0.0, 20.0), (0.0, 40.0)]
    assert all(sequence.spec is spec for sequence in sequences)


def test_find_training_block_sequences_filters_ineligible_windows() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)

    ineligible_route = replace(
        route,
        extras={
            **route.extras,
            "surface": {"values": [[0, 3, 0]]},
        },
    )
    assert not find_training_block_sequences(
        ineligible_route,
        timeline,
        TrainingBlockSpec(work_duration_s=20.0, repetitions=2),
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        step_s=20.0,
        requirements=RouteTrainingWindowRequirements(
            min_asphalt_percentage=100.0,
        ),
    )

    assert find_training_block_sequences(
        route,
        timeline,
        TrainingBlockSpec(work_duration_s=20.0, repetitions=2),
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        step_s=20.0,
        requirements=RouteTrainingWindowRequirements(
            min_asphalt_percentage=90.0,
        ),
    )


def test_find_training_block_sequences_validates_limit() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)

    with pytest.raises(ValueError, match="max_sequences"):
        find_training_block_sequences(
            route,
            timeline,
            TrainingBlockSpec(work_duration_s=20.0, repetitions=2),
            start_time_min_s=0.0,
            start_time_max_s=40.0,
            max_sequences=0,
        )


def test_rank_training_block_sequences_prioritizes_weakest_block() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    spec = TrainingBlockSpec(
        work_duration_s=20.0,
        repetitions=2,
        recovery_min_s=0.0,
        recovery_max_s=20.0,
    )
    sequences = find_training_block_sequences(
        route,
        timeline,
        spec,
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        step_s=20.0,
    )
    assert len(sequences) >= 2

    stronger = sequences[0]
    weaker_block = replace(
        stronger.work_blocks[-1],
        quality=replace(
            stronger.work_blocks[-1].quality,
            asphalt_percentage=0.0,
        ),
    )
    weaker = replace(
        stronger,
        work_blocks=(*stronger.work_blocks[:-1], weaker_block),
    )

    assert rank_training_block_sequences((weaker, stronger))[0] is stronger


def test_cycling_session_requirements_filter_enabled_segments() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    training_window = calculate_training_window(
        route,
        timeline,
        start_time_s=20.0,
        duration_s=20.0,
    )
    warmup = analyze_route_warmup(route, timeline, training_window)
    cooldown = analyze_route_cooldown(route, timeline, training_window)

    assert cycling_session_meets_requirements(
        warmup,
        cooldown,
        CyclingSessionRequirements(
            max_warmup_maneuvers_per_hour=0.0,
            min_warmup_asphalt_percentage=100.0,
            max_cooldown_elevation_gain_rate_m_per_hour=10_000.0,
            max_cooldown_footway_percentage=0.0,
        ),
    )
    assert not cycling_session_meets_requirements(
        warmup,
        cooldown,
        CyclingSessionRequirements(max_warmup_gradient_percentage=-100.0),
    )


def test_cycling_session_requirements_reject_missing_enabled_segment() -> None:
    assert not cycling_session_meets_requirements(
        None,
        None,
        CyclingSessionRequirements(max_warmup_maneuvers_per_hour=5.0),
    )
    assert cycling_session_meets_requirements(
        None,
        None,
        CyclingSessionRequirements(),
    )


def test_cycling_session_requirements_validate_values() -> None:
    with pytest.raises(ValueError, match="percentage requirements"):
        cycling_session_meets_requirements(
            None,
            None,
            CyclingSessionRequirements(min_warmup_asphalt_percentage=101.0),
        )

    with pytest.raises(ValueError, match="rate requirements"):
        cycling_session_meets_requirements(
            None,
            None,
            CyclingSessionRequirements(max_cooldown_maneuvers_per_hour=-1.0),
        )


def test_training_window_comparison_metrics_requires_elevation() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        window=SimpleNamespace(
            elevation_gain_m=None,
            elevation_loss_m=None,
            duration_s=1800.0,
            distance_m=10000.0,
        ),
        interruptions=SimpleNamespace(
            maneuver_count=0,
        ),
    )

    with pytest.raises(
        routing.RouteParsingError,
        match="requires elevation gain and loss",
    ):
        routing.calculate_training_window_comparison_metrics(
            analysis,
        )


def test_rank_training_windows_across_durations_prefers_balance_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis_20 = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=100.0,
            suitability_7_plus_percentage=100.0,
        )
    )
    analysis_30 = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=100.0,
            suitability_7_plus_percentage=100.0,
        )
    )

    metrics = {
        id(analysis_20): SimpleNamespace(
            climbing_balance_rate_m_per_hour=930.0,
            climbing_balance_gradient_percentage=4.0,
            elevation_loss_rate_m_per_hour=10.0,
            maneuvers_per_hour=10.0,
        ),
        id(analysis_30): SimpleNamespace(
            climbing_balance_rate_m_per_hour=970.0,
            climbing_balance_gradient_percentage=3.8,
            elevation_loss_rate_m_per_hour=9.0,
            maneuvers_per_hour=12.0,
        ),
    }

    monkeypatch.setattr(
        routing,
        "calculate_training_window_comparison_metrics",
        lambda analysis: metrics[id(analysis)],
    )

    ranked = routing.rank_training_windows_across_durations(
        (analysis_20, analysis_30),
    )

    assert ranked == (analysis_30, analysis_20)


def test_rank_training_windows_across_durations_uses_gradient_tiebreaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    lower_gradient = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=100.0,
            suitability_7_plus_percentage=100.0,
        )
    )
    higher_gradient = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=100.0,
            suitability_7_plus_percentage=100.0,
        )
    )

    metrics = {
        id(lower_gradient): SimpleNamespace(
            climbing_balance_rate_m_per_hour=900.0,
            climbing_balance_gradient_percentage=3.0,
            elevation_loss_rate_m_per_hour=10.0,
            maneuvers_per_hour=5.0,
        ),
        id(higher_gradient): SimpleNamespace(
            climbing_balance_rate_m_per_hour=900.0,
            climbing_balance_gradient_percentage=4.0,
            elevation_loss_rate_m_per_hour=10.0,
            maneuvers_per_hour=5.0,
        ),
    }

    monkeypatch.setattr(
        routing,
        "calculate_training_window_comparison_metrics",
        lambda analysis: metrics[id(analysis)],
    )

    ranked = routing.rank_training_windows_across_durations(
        (lower_gradient, higher_gradient),
    )

    assert ranked == (higher_gradient, lower_gradient)


def test_find_best_training_window_across_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = object()
    timeline = object()

    analysis_20 = object()
    analysis_30 = object()
    analysis_40 = object()

    monkeypatch.setattr(
        routing,
        "find_best_training_windows_by_duration",
        lambda *_args, **_kwargs: (
            analysis_20,
            analysis_30,
            analysis_40,
        ),
    )

    monkeypatch.setattr(
        routing,
        "rank_training_windows_across_durations",
        lambda analyses: (
            analyses[1],
            analyses[0],
            analyses[2],
        ),
    )

    best = routing.find_best_training_window_across_durations(
        route,
        timeline,
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1200.0, 1800.0, 2400.0),
        step_s=60.0,
    )

    assert best is analysis_30


def test_find_best_training_window_across_durations_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    monkeypatch.setattr(
        routing,
        "find_best_training_windows_by_duration",
        lambda *_args, **_kwargs: (),
    )

    best = routing.find_best_training_window_across_durations(
        object(),
        object(),
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1200.0, 1800.0, 2400.0),
    )

    assert best is None


def test_evaluate_cycling_route_candidates_reuses_window_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = _route_for_extra_tests({})
    candidates = (
        CyclingRouteCandidate(
            candidate_id="round-trip-1",
            strategy="ors_round_trip",
            seed=0,
            target_distance_m=50_000.0,
            route=route,
        ),
        CyclingRouteCandidate(
            candidate_id="round-trip-2",
            strategy="ors_round_trip",
            seed=1,
            target_distance_m=50_000.0,
            route=route,
        ),
    )
    window = object()
    analysis = Mock(window=window)
    warmup = object()
    cooldown = object()
    route_quality = object()
    timelines: list[CyclingRoute] = []
    timeline = object()

    monkeypatch.setattr(
        routing,
        "calculate_route_timeline",
        lambda candidate_route: timelines.append(candidate_route) or timeline,
    )
    find_mock = Mock(
        side_effect=[(analysis,), ()]
    )
    monkeypatch.setattr(
        routing,
        "find_best_training_windows_by_duration",
        find_mock,
    )
    monkeypatch.setattr(
        routing,
        "rank_training_windows_across_durations",
        lambda analyses: analyses,
    )
    analyze_warmup = Mock(return_value=warmup)
    analyze_cooldown = Mock(return_value=cooldown)
    calculate_quality = Mock(return_value=route_quality)
    session_requirements = object()
    session_meets_requirements = Mock(return_value=True)
    monkeypatch.setattr(routing, "analyze_route_warmup", analyze_warmup)
    monkeypatch.setattr(routing, "analyze_route_cooldown", analyze_cooldown)
    monkeypatch.setattr(
        routing,
        "calculate_cycling_route_quality_metrics",
        calculate_quality,
    )
    monkeypatch.setattr(
        routing,
        "cycling_session_meets_requirements",
        session_meets_requirements,
    )

    results = evaluate_cycling_route_candidates(
        candidates,
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1800.0,),
        session_requirements=session_requirements,
    )

    assert len(results) == 1
    assert isinstance(results[0], CyclingRouteCandidateAnalysis)
    assert results[0].candidate is candidates[0]
    assert results[0].best_training_window is analysis
    assert results[0].best_by_duration == (analysis,)
    assert results[0].warmup is warmup
    assert results[0].cooldown is cooldown
    assert results[0].route_quality is route_quality
    assert timelines == [route, route]
    analyze_warmup.assert_called_once_with(route, timeline, window)
    analyze_cooldown.assert_called_once_with(route, timeline, window)
    calculate_quality.assert_called_once_with(route, timeline)
    session_meets_requirements.assert_called_once_with(
        warmup,
        cooldown,
        session_requirements,
    )


def test_evaluate_cycling_route_candidates_uses_training_block_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = _route_for_extra_tests({})
    candidate = CyclingRouteCandidate(
        candidate_id="round-trip-1",
        strategy="ors_round_trip",
        seed=0,
        target_distance_m=2_000.0,
        route=route,
    )
    timeline = object()
    first_window = Mock(window=object())
    last_window = Mock(window=object())
    sequence = Mock(work_blocks=(first_window, last_window))
    spec = TrainingBlockSpec(
        work_duration_s=480.0,
        repetitions=2,
        recovery_min_s=120.0,
        recovery_max_s=240.0,
    )
    warmup = object()
    cooldown = object()

    monkeypatch.setattr(
        routing,
        "calculate_route_timeline",
        Mock(return_value=timeline),
    )
    find_sequences = Mock(return_value=(sequence,))
    monkeypatch.setattr(
        routing,
        "find_training_block_sequences",
        find_sequences,
    )
    monkeypatch.setattr(
        routing,
        "rank_training_block_sequences",
        lambda sequences: sequences,
    )
    find_continuous = Mock()
    monkeypatch.setattr(
        routing,
        "find_best_training_windows_by_duration",
        find_continuous,
    )
    analyze_warmup = Mock(return_value=warmup)
    analyze_cooldown = Mock(return_value=cooldown)
    monkeypatch.setattr(routing, "analyze_route_warmup", analyze_warmup)
    monkeypatch.setattr(routing, "analyze_route_cooldown", analyze_cooldown)
    monkeypatch.setattr(
        routing,
        "calculate_cycling_route_quality_metrics",
        Mock(return_value=object()),
    )

    result = evaluate_cycling_route_candidates(
        (candidate,),
        start_time_min_s=1_200.0,
        start_time_max_s=1_800.0,
        durations_s=(1_800.0,),
        training_block_spec=spec,
    )

    assert result[0].best_training_block_sequence is sequence
    assert result[0].best_training_window is first_window
    assert result[0].best_by_duration == (first_window, last_window)
    find_continuous.assert_not_called()
    find_sequences.assert_called_once_with(
        route,
        timeline,
        spec,
        start_time_min_s=1_200.0,
        start_time_max_s=1_800.0,
        step_s=60.0,
        requirements=None,
    )
    analyze_warmup.assert_called_once_with(
        route,
        timeline,
        first_window.window,
    )
    analyze_cooldown.assert_called_once_with(
        route,
        timeline,
        last_window.window,
    )


def test_rank_cycling_route_candidates_prioritizes_best_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = _route_for_extra_tests({})
    stronger_window = object()
    closer_window = object()
    stronger = CyclingRouteCandidateAnalysis(
        candidate=CyclingRouteCandidate(
            candidate_id="round-trip-1",
            strategy="ors_round_trip",
            seed=0,
            target_distance_m=10_000.0,
            route=route,
        ),
        best_training_window=stronger_window,
        best_by_duration=(),
    )
    closer = CyclingRouteCandidateAnalysis(
        candidate=CyclingRouteCandidate(
            candidate_id="round-trip-2",
            strategy="ors_round_trip",
            seed=1,
            target_distance_m=2_000.0,
            route=route,
        ),
        best_training_window=closer_window,
        best_by_duration=(),
    )
    keys = {
        id(stronger_window): (900.0, 4.0, -10.0, 95.0, 90.0, -8.0),
        id(closer_window): (800.0, 4.0, -10.0, 95.0, 90.0, -8.0),
    }
    monkeypatch.setattr(
        routing,
        "_cross_duration_training_window_rank_key",
        lambda analysis: keys[id(analysis)],
    )

    assert rank_cycling_route_candidates((closer, stronger)) == (
        stronger,
        closer,
    )


def test_rank_cycling_route_candidates_uses_training_block_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = _route_for_extra_tests({})
    common_window = object()
    stronger_sequence = object()
    weaker_sequence = object()

    def analyzed(
        candidate_id: str,
        sequence: object,
    ) -> CyclingRouteCandidateAnalysis:
        return CyclingRouteCandidateAnalysis(
            candidate=CyclingRouteCandidate(
                candidate_id=candidate_id,
                strategy="ors_round_trip",
                seed=0,
                target_distance_m=2_000.0,
                route=route,
            ),
            best_training_window=common_window,
            best_by_duration=(),
            best_training_block_sequence=sequence,
        )

    stronger = analyzed("stronger", stronger_sequence)
    weaker = analyzed("weaker", weaker_sequence)
    keys = {
        id(stronger_sequence): ((900.0, 4.0), -10.0),
        id(weaker_sequence): ((800.0, 4.0), -10.0),
    }
    monkeypatch.setattr(
        routing,
        "_training_block_sequence_rank_key",
        lambda sequence: keys[id(sequence)],
    )
    continuous_rank = Mock()
    monkeypatch.setattr(
        routing,
        "_cross_duration_training_window_rank_key",
        continuous_rank,
    )

    assert rank_cycling_route_candidates((weaker, stronger)) == (
        stronger,
        weaker,
    )
    continuous_rank.assert_not_called()


def test_rank_cycling_route_candidates_uses_distance_tiebreaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    common_window = object()
    exact_route = _route_for_extra_tests({})
    farther_route = CyclingRoute(
        distance_m=2_500.0,
        duration_s=exact_route.duration_s,
        elevation_gain_m=exact_route.elevation_gain_m,
        elevation_loss_m=exact_route.elevation_loss_m,
        ors_ascent_m=exact_route.ors_ascent_m,
        ors_descent_m=exact_route.ors_descent_m,
        geometry=exact_route.geometry,
        waypoint_indices=exact_route.waypoint_indices,
        segments=exact_route.segments,
        extras=exact_route.extras,
    )

    def analyzed(candidate_id: str, seed: int, route: CyclingRoute):
        return CyclingRouteCandidateAnalysis(
            candidate=CyclingRouteCandidate(
                candidate_id=candidate_id,
                strategy="ors_round_trip",
                seed=seed,
                target_distance_m=2_000.0,
                route=route,
            ),
            best_training_window=common_window,
            best_by_duration=(),
        )

    exact = analyzed("round-trip-1", 0, exact_route)
    farther = analyzed("round-trip-2", 1, farther_route)
    monkeypatch.setattr(
        routing,
        "_cross_duration_training_window_rank_key",
        lambda _analysis: (800.0, 4.0, -10.0, 95.0, 90.0, -8.0),
    )

    assert rank_cycling_route_candidates((farther, exact)) == (
        exact,
        farther,
    )


def test_rank_cycling_route_candidates_uses_duration_before_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    base = _route_for_extra_tests({})
    common_window = object()

    def analyzed(
        candidate_id: str,
        duration_s: float,
        distance_m: float,
    ) -> CyclingRouteCandidateAnalysis:
        route = CyclingRoute(
            distance_m=distance_m,
            duration_s=duration_s,
            elevation_gain_m=base.elevation_gain_m,
            elevation_loss_m=base.elevation_loss_m,
            ors_ascent_m=base.ors_ascent_m,
            ors_descent_m=base.ors_descent_m,
            geometry=base.geometry,
            waypoint_indices=base.waypoint_indices,
            segments=base.segments,
            extras=base.extras,
        )
        return CyclingRouteCandidateAnalysis(
            candidate=CyclingRouteCandidate(
                candidate_id=candidate_id,
                strategy="ors_round_trip",
                seed=0,
                target_distance_m=2000.0,
                route=route,
                target_duration_s=3600.0,
            ),
            best_training_window=common_window,
            best_by_duration=(),
        )

    duration_fit = analyzed("duration-fit", 3600.0, 2500.0)
    distance_fit = analyzed("distance-fit", 4200.0, 2000.0)
    monkeypatch.setattr(
        routing,
        "_cross_duration_training_window_rank_key",
        lambda _analysis: (800.0, 4.0, -10.0, 95.0, 90.0, -8.0),
    )

    assert rank_cycling_route_candidates((distance_fit, duration_fit)) == (
        duration_fit,
        distance_fit,
    )


def test_rank_cycling_route_candidates_uses_session_quality_tiebreakers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    route = _route_for_extra_tests({})
    common_window = object()

    def segment(
        maneuvers_per_hour: float,
        elevation_gain_rate_m_per_hour: float = 100.0,
    ):
        return SimpleNamespace(
            quality=SimpleNamespace(
                asphalt_percentage=100.0,
                suitability_7_plus_percentage=100.0,
                footway_percentage=0.0,
            ),
            comparison=SimpleNamespace(
                elevation_gain_rate_m_per_hour=elevation_gain_rate_m_per_hour,
                maneuvers_per_hour=maneuvers_per_hour,
            ),
        )

    def analyzed(
        candidate_id: str,
        *,
        warmup_maneuvers: float,
        cooldown_climbing: float = 100.0,
        route_asphalt: float = 100.0,
    ):
        return CyclingRouteCandidateAnalysis(
            candidate=CyclingRouteCandidate(
                candidate_id=candidate_id,
                strategy="ors_round_trip",
                seed=0,
                target_distance_m=route.distance_m,
                route=route,
            ),
            best_training_window=common_window,
            best_by_duration=(),
            warmup=segment(warmup_maneuvers),
            cooldown=segment(0.0, cooldown_climbing),
            route_quality=SimpleNamespace(
                asphalt_percentage=route_asphalt,
                suitability_7_plus_percentage=100.0,
                road_or_cycleway_percentage=100.0,
                footway_percentage=0.0,
                unknown_surface_percentage=0.0,
                maneuver_rate_per_hour=0.0,
                sharp_turn_count=0,
                roundabout_count=0,
            ),
        )

    cleaner = analyzed("cleaner", warmup_maneuvers=2.0)
    interrupted = analyzed("interrupted", warmup_maneuvers=8.0)
    monkeypatch.setattr(
        routing,
        "_cross_duration_training_window_rank_key",
        lambda _analysis: (800.0, 4.0, -10.0, 95.0, 90.0, -8.0),
    )

    assert rank_cycling_route_candidates((interrupted, cleaner)) == (
        cleaner,
        interrupted,
    )

    easier_cooldown = analyzed(
        "easier-cooldown",
        warmup_maneuvers=2.0,
        cooldown_climbing=50.0,
    )
    harder_cooldown = analyzed(
        "harder-cooldown",
        warmup_maneuvers=2.0,
        cooldown_climbing=200.0,
    )
    assert rank_cycling_route_candidates(
        (harder_cooldown, easier_cooldown)
    ) == (easier_cooldown, harder_cooldown)

    better_route = analyzed(
        "better-route",
        warmup_maneuvers=2.0,
        route_asphalt=95.0,
    )
    worse_route = analyzed(
        "worse-route",
        warmup_maneuvers=2.0,
        route_asphalt=80.0,
    )
    assert rank_cycling_route_candidates((worse_route, better_route)) == (
        better_route,
        worse_route,
    )


async def test_find_cycling_training_route_candidates_orchestrates_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    origin = _resolved_location("Start", 2.63, 39.59)
    generated = (object(), object(), object())
    deduplicated = generated[:2]
    evaluated = object()
    ranked = object()
    generate = AsyncMock(return_value=generated)
    evaluate = Mock(return_value=evaluated)
    rank = Mock(return_value=ranked)
    deduplicate = Mock(return_value=deduplicated)
    monkeypatch.setattr(routing, "generate_cycling_route_candidates", generate)
    monkeypatch.setattr(routing, "evaluate_cycling_route_candidates", evaluate)
    monkeypatch.setattr(routing, "rank_cycling_route_candidates", rank)
    monkeypatch.setattr(
        routing,
        "deduplicate_cycling_route_candidates",
        deduplicate,
    )

    async with OpenRouteServiceClient(_config()) as client:
        result = await find_cycling_training_route_candidates(
            client,
            origin,
            target_distance_m=50_000.0,
            candidate_count=3,
            start_time_min_s=1200.0,
            start_time_max_s=1800.0,
            durations_s=(1200.0, 1800.0),
            requirements=None,
        )

    assert result.ranked is ranked
    assert result.candidates_generated == 3
    assert result.candidates_after_distance_filter == 3
    assert result.candidates_after_duration_filter == 3
    assert result.candidates_after_deduplication == 2
    generate.assert_awaited_once_with(
        client,
        origin,
        target_distance_m=50_000.0,
        candidate_count=3,
        round_trip_points=2,
        seed_start=0,
        max_distance_deviation_percentage=50.0,
        target_duration_s=None,
        avoid_features=(),
    )
    evaluate.assert_called_once_with(
        deduplicated,
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1200.0, 1800.0),
        step_s=60.0,
        requirements=None,
        session_requirements=None,
        training_block_spec=None,
    )
    rank.assert_called_once_with(evaluated)
    deduplicate.assert_called_once_with(
        generated,
        overlap_threshold_percentage=90.0,
        resample_spacing_m=100.0,
        proximity_m=50.0,
    )


def test_serialize_cycling_route_candidate_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = _route_for_extra_tests({})
    best = object()
    other = object()
    warmup = object()
    cooldown = object()
    route_quality = object()
    block_sequence = object()
    analysis = CyclingRouteCandidateAnalysis(
        candidate=CyclingRouteCandidate(
            candidate_id="round-trip-2",
            strategy="ors_round_trip",
            seed=7,
            target_distance_m=2_500.0,
            route=route,
            target_duration_s=route.duration_s + 60.0,
        ),
        best_training_window=best,
        best_by_duration=(best, other),
        warmup=warmup,
        cooldown=cooldown,
        route_quality=route_quality,
        best_training_block_sequence=block_sequence,
    )
    serialize_window = Mock(
        side_effect=lambda _route, window: {"window": id(window)}
    )
    monkeypatch.setattr(
        routing,
        "_serialize_training_window_analysis",
        serialize_window,
    )
    serialize_segment = Mock(
        side_effect=lambda _route, segment: {"segment": id(segment)}
    )
    serialize_quality = Mock(return_value={"quality": "route"})
    serialize_sequence = Mock(return_value={"sequence": "blocks"})
    monkeypatch.setattr(
        routing,
        "_serialize_route_session_segment_analysis",
        serialize_segment,
    )
    monkeypatch.setattr(
        routing,
        "_serialize_cycling_route_quality_metrics",
        serialize_quality,
    )
    monkeypatch.setattr(
        routing,
        "serialize_training_block_sequence_analysis",
        serialize_sequence,
    )

    result = routing.serialize_cycling_route_candidate_analysis(analysis)

    assert result["candidate_id"] == "round-trip-2"
    assert result["generation"] == {
        "strategy": "ors_round_trip",
        "seed": 7,
        "target_distance_meters": 2_500.0,
        "distance_deviation_meters": -500.0,
        "distance_deviation_percentage": -20.0,
        "target_duration_seconds": route.duration_s + 60.0,
        "target_duration_minutes": (route.duration_s + 60.0) / 60.0,
        "duration_deviation_seconds": -60.0,
        "duration_deviation_percentage": (
            -60.0 / (route.duration_s + 60.0) * 100.0
        ),
    }
    assert result["route"]["geometry"] == {
        "type": "LineString",
        "coordinates": [
            [2.6, 39.5, 100.0],
            [2.61, 39.5, 100.0],
            [2.62, 39.5, 100.0],
        ],
    }
    assert result["best_training_window"] == {"window": id(best)}
    assert result["best_by_duration"] == [
        {"window": id(best)},
        {"window": id(other)},
    ]
    assert result["warmup"] == {"segment": id(warmup)}
    assert result["cooldown"] == {"segment": id(cooldown)}
    assert result["route_quality"] == {"quality": "route"}
    assert result["training_block_sequence"] == {"sequence": "blocks"}
    assert serialize_segment.call_count == 2
    serialize_quality.assert_called_once_with(route_quality)
    serialize_sequence.assert_called_once_with(route, block_sequence)


def test_serialize_training_block_sequence_analysis() -> None:
    route = _route_for_session_segment_tests()
    timeline = calculate_route_timeline(route)
    spec = TrainingBlockSpec(
        work_duration_s=20.0,
        repetitions=2,
        recovery_min_s=0.0,
        recovery_max_s=20.0,
    )
    sequence = find_training_block_sequences(
        route,
        timeline,
        spec,
        start_time_min_s=0.0,
        start_time_max_s=40.0,
        step_s=20.0,
    )[0]

    result = serialize_training_block_sequence_analysis(route, sequence)

    assert result["spec"] == {
        "work_duration_s": 20.0,
        "repetitions": 2,
        "recovery_min_s": 0.0,
        "recovery_max_s": 20.0,
    }
    assert result["summary"] == {
        "total_work_duration_seconds": sequence.total_work_duration_s,
        "total_work_duration_minutes": sequence.total_work_duration_s / 60.0,
        "total_distance_meters": sequence.total_distance_m,
        "total_elevation_gain_meters": sequence.total_elevation_gain_m,
        "total_elevation_loss_meters": sequence.total_elevation_loss_m,
        "climbing_balance_meters": sequence.climbing_balance_m,
        "total_maneuver_count": sequence.total_maneuver_count,
        "elevation_gain_rate_range_meters_per_hour": (
            sequence.elevation_gain_rate_range_m_per_hour
        ),
    }
    assert len(result["work_blocks"]) == 2
    assert len(result["recoveries"]) == 1
    assert result["work_blocks"][0]["start"]["time_seconds"] == (
        sequence.work_blocks[0].window.start.time_s
    )
    assert result["recoveries"] == [None]

    sequence_with_recovery = next(
        candidate
        for candidate in find_training_block_sequences(
            route,
            timeline,
            spec,
            start_time_min_s=0.0,
            start_time_max_s=40.0,
            step_s=20.0,
        )
        if candidate.recoveries[0] is not None
    )
    recovered_result = serialize_training_block_sequence_analysis(
        route,
        sequence_with_recovery,
    )
    recovery = sequence_with_recovery.recoveries[0]
    assert recovery is not None
    assert recovered_result["recoveries"][0]["duration_seconds"] == (
        recovery.segment.duration_s
    )


async def test_find_best_cycling_training_window_requires_ors_config() -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    ctx = SimpleNamespace(
        get_state=AsyncMock(
            return_value=ICUConfig(
                openrouteservice_api_key="",
            )
        )
    )

    result = await routing.find_best_cycling_training_window(
        locations=["A", "B"],
        ctx=ctx,
    )

    response = json.loads(result)

    assert response["error"]["type"] == "configuration_error"


async def test_find_best_cycling_training_window_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    config = _config()

    ctx = SimpleNamespace(
        get_state=AsyncMock(return_value=config)
    )

    resolved_a = ResolvedLocation(
        input_value="A",
        source="geocode",
        label="A",
        original_longitude=2.6,
        original_latitude=39.5,
        longitude=2.6,
        latitude=39.5,
        snapped_distance_m=1.0,
    )
    resolved_b = ResolvedLocation(
        input_value="B",
        source="geocode",
        label="B",
        original_longitude=2.7,
        original_latitude=39.6,
        longitude=2.7,
        latitude=39.6,
        snapped_distance_m=1.0,
    )

    route = CyclingRoute(
        distance_m=10000.0,
        duration_s=3600.0,
        elevation_gain_m=500.0,
        elevation_loss_m=20.0,
        ors_ascent_m=500.0,
        ors_descent_m=20.0,
        geometry=(
            RouteCoordinate(2.6, 39.5, 100.0),
            RouteCoordinate(2.7, 39.6, 500.0),
        ),
        waypoint_indices=(0, 1),
        segments=(),
        extras={},
    )

    window = RouteTrainingWindow(
        start=RouteTimelinePoint(
            geometry_index=0,
            distance_m=0.0,
            time_s=1500.0,
            elevation_m=100.0,
        ),
        end=RouteTimelinePoint(
            geometry_index=1,
            distance_m=10000.0,
            time_s=3300.0,
            elevation_m=500.0,
        ),
        distance_m=10000.0,
        duration_s=1800.0,
        elevation_gain_m=500.0,
        elevation_loss_m=20.0,
        net_elevation_gain_m=400.0,
    )

    quality = RouteQualityMetrics(
        asphalt_percentage=99.0,
        unknown_surface_percentage=1.0,
        paving_stones_percentage=0.0,
        road_or_cycleway_percentage=100.0,
        footway_percentage=0.0,
        suitability_7_plus_percentage=100.0,
        suitability_8_plus_percentage=80.0,
        incline_7_plus_percentage=10.0,
        incline_10_plus_percentage=5.0,
        decline_7_plus_percentage=0.0,
        decline_10_plus_percentage=0.0,
    )

    interruptions = RouteWindowInterruptionMetrics(
        maneuver_count=2,
        sharp_turn_count=0,
        roundabout_count=0,
        u_turn_count=0,
    )

    analysis = routing.RouteTrainingWindowAnalysis(
        window=window,
        quality=quality,
        interruptions=interruptions,
    )

    class FakeClient:
        def __init__(self, _config: ICUConfig) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            return None

    resolve = AsyncMock(
        side_effect=[resolved_a, resolved_b]
    )
    build = AsyncMock(return_value=route)

    monkeypatch.setattr(
        routing,
        "OpenRouteServiceClient",
        FakeClient,
    )
    monkeypatch.setattr(
        routing,
        "resolve_location",
        resolve,
    )
    monkeypatch.setattr(
        routing,
        "build_cycling_route",
        build,
    )
    monkeypatch.setattr(
        routing,
        "calculate_route_timeline",
        lambda _route: object(),
    )
    monkeypatch.setattr(
        routing,
        "find_best_training_windows_by_duration",
        lambda *_args, **_kwargs: (analysis,),
    )
    monkeypatch.setattr(
        routing,
        "rank_training_windows_across_durations",
        lambda analyses: analyses,
    )

    result = await routing.find_best_cycling_training_window(
        locations=["A", "B"],
        durations_minutes=[30.0],
        ctx=ctx,
    )

    response = json.loads(result)
    data = response["data"]

    assert data["route"]["distance_meters"] == 10000.0

    best = data["best_training_window"]

    assert best["duration_minutes"] == pytest.approx(30.0)
    assert best["elevation_gain_meters"] == 500.0
    assert best["elevation_loss_meters"] == 20.0
    assert best["start"]["latitude"] == 39.5
    assert best["end"]["latitude"] == 39.6

    assert len(data["best_by_duration"]) == 1
    assert response["metadata"]["profile"] == "cycling-road"

    assert resolve.await_count == 2
    build.assert_awaited_once()


def test_training_window_requirements_allow_all_by_default() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=10.0,
            road_or_cycleway_percentage=20.0,
            suitability_7_plus_percentage=30.0,
            footway_percentage=70.0,
        ),
        interruptions=SimpleNamespace(
            maneuver_count=100,
        ),
        window=SimpleNamespace(
            elevation_gain_m=100.0,
            elevation_loss_m=100.0,
            duration_s=1800.0,
            distance_m=1000.0,
        ),
    )

    requirements = routing.RouteTrainingWindowRequirements()

    assert routing.training_window_meets_requirements(
        analysis,
        requirements,
    )


def test_training_window_requirements_reject_low_route_quality() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=80.0,
            road_or_cycleway_percentage=95.0,
            suitability_7_plus_percentage=90.0,
            footway_percentage=5.0,
        ),
    )

    requirements = routing.RouteTrainingWindowRequirements(
        min_asphalt_percentage=90.0,
        min_road_or_cycleway_percentage=90.0,
        min_suitability_7_plus_percentage=80.0,
        max_footway_percentage=10.0,
    )

    assert not routing.training_window_meets_requirements(
        analysis,
        requirements,
    )


def test_training_window_requirements_reject_excessive_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    analysis = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=100.0,
            road_or_cycleway_percentage=100.0,
            suitability_7_plus_percentage=100.0,
            footway_percentage=0.0,
        ),
    )

    monkeypatch.setattr(
        routing,
        "calculate_training_window_comparison_metrics",
        lambda _analysis: SimpleNamespace(
            elevation_loss_rate_m_per_hour=120.0,
            maneuvers_per_hour=15.0,
        ),
    )

    requirements = routing.RouteTrainingWindowRequirements(
        max_elevation_loss_rate_m_per_hour=100.0,
        max_maneuvers_per_hour=10.0,
    )

    assert not routing.training_window_meets_requirements(
        analysis,
        requirements,
    )


def test_filter_training_window_analyses() -> None:
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    good = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=99.0,
            road_or_cycleway_percentage=100.0,
            suitability_7_plus_percentage=100.0,
            footway_percentage=0.0,
        ),
    )
    bad = SimpleNamespace(
        quality=SimpleNamespace(
            asphalt_percentage=70.0,
            road_or_cycleway_percentage=100.0,
            suitability_7_plus_percentage=100.0,
            footway_percentage=0.0,
        ),
    )

    requirements = routing.RouteTrainingWindowRequirements(
        min_asphalt_percentage=90.0,
    )

    assert routing.filter_training_window_analyses(
        (bad, good),
        requirements,
    ) == (good,)


def test_training_window_requirements_validate_ranges() -> None:
    import intervals_icu_mcp.tools.routing as routing

    analysis = object()

    with pytest.raises(
        ValueError,
        match="percentage requirements",
    ):
        routing.training_window_meets_requirements(
            analysis,
            routing.RouteTrainingWindowRequirements(
                min_asphalt_percentage=101.0,
            ),
        )

    with pytest.raises(
        ValueError,
        match="rate requirements",
    ):
        routing.training_window_meets_requirements(
            analysis,
            routing.RouteTrainingWindowRequirements(
                max_maneuvers_per_hour=-1.0,
            ),
        )


def test_find_best_training_window_filters_ineligible_analyses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    route = object()
    timeline = object()

    window_1 = object()
    window_2 = object()

    rejected = object()
    eligible = object()

    requirements = routing.RouteTrainingWindowRequirements(
        min_asphalt_percentage=90.0,
    )

    monkeypatch.setattr(
        routing,
        "generate_training_window_candidates",
        lambda *_args, **_kwargs: (window_1, window_2),
    )
    monkeypatch.setattr(
        routing,
        "analyze_training_window_candidates",
        lambda *_args, **_kwargs: (rejected, eligible),
    )
    monkeypatch.setattr(
        routing,
        "filter_training_window_analyses",
        lambda analyses, supplied_requirements: (
            eligible,
        )
        if supplied_requirements is requirements
        else analyses,
    )
    monkeypatch.setattr(
        routing,
        "rank_training_window_analyses",
        lambda analyses: analyses,
    )

    best = routing.find_best_training_window(
        route,
        timeline,
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        duration_s=1800.0,
        requirements=requirements,
    )

    assert best is eligible


def test_find_best_training_window_returns_none_when_all_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    monkeypatch.setattr(
        routing,
        "generate_training_window_candidates",
        lambda *_args, **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        routing,
        "analyze_training_window_candidates",
        lambda *_args, **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        routing,
        "filter_training_window_analyses",
        lambda *_args, **_kwargs: (),
    )

    best = routing.find_best_training_window(
        object(),
        object(),
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        duration_s=1800.0,
        requirements=routing.RouteTrainingWindowRequirements(
            min_asphalt_percentage=90.0,
        ),
    )

    assert best is None


def test_find_best_training_windows_by_duration_passes_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intervals_icu_mcp.tools.routing as routing

    requirements = routing.RouteTrainingWindowRequirements(
        min_asphalt_percentage=90.0,
    )

    calls: list[object] = []

    def fake_find_best(*_args: object, **kwargs: object) -> object:
        calls.append(kwargs["requirements"])
        return object()

    monkeypatch.setattr(
        routing,
        "find_best_training_window",
        fake_find_best,
    )

    routing.find_best_training_windows_by_duration(
        object(),
        object(),
        start_time_min_s=1200.0,
        start_time_max_s=1800.0,
        durations_s=(1200.0, 1800.0, 2400.0),
        requirements=requirements,
    )

    assert calls == [
        requirements,
        requirements,
        requirements,
    ]


async def test_find_best_cycling_training_window_passes_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    config = _config()
    ctx = SimpleNamespace(
        get_state=AsyncMock(return_value=config)
    )

    resolved = ResolvedLocation(
        input_value="A",
        source="coordinates",
        label="A",
        original_longitude=2.6,
        original_latitude=39.5,
        longitude=2.6,
        latitude=39.5,
        snapped_distance_m=0.0,
    )

    route = CyclingRoute(
        distance_m=1000.0,
        duration_s=600.0,
        elevation_gain_m=100.0,
        elevation_loss_m=0.0,
        ors_ascent_m=100.0,
        ors_descent_m=0.0,
        geometry=(
            RouteCoordinate(2.6, 39.5, 100.0),
            RouteCoordinate(2.7, 39.6, 200.0),
        ),
        waypoint_indices=(0, 1),
        segments=(),
        extras={},
    )

    class FakeClient:
        def __init__(self, _config: ICUConfig) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            return None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        routing,
        "OpenRouteServiceClient",
        FakeClient,
    )
    monkeypatch.setattr(
        routing,
        "resolve_location",
        AsyncMock(return_value=resolved),
    )
    monkeypatch.setattr(
        routing,
        "build_cycling_route",
        AsyncMock(return_value=route),
    )
    monkeypatch.setattr(
        routing,
        "calculate_route_timeline",
        lambda _route: object(),
    )

    def fake_find(*_args: object, **kwargs: object) -> tuple[()]:
        captured["requirements"] = kwargs["requirements"]
        return ()

    monkeypatch.setattr(
        routing,
        "find_best_training_windows_by_duration",
        fake_find,
    )

    result = await routing.find_best_cycling_training_window(
        locations=["A", "B"],
        min_asphalt_percentage=90.0,
        max_elevation_loss_rate_m_per_hour=20.0,
        max_maneuvers_per_hour=12.0,
        ctx=ctx,
    )

    response = json.loads(result)

    assert response["error"]["type"] == "not_found"

    requirements = captured["requirements"]

    assert isinstance(
        requirements,
        routing.RouteTrainingWindowRequirements,
    )
    assert requirements.min_asphalt_percentage == 90.0
    assert (
        requirements.max_elevation_loss_rate_m_per_hour
        == 20.0
    )
    assert requirements.max_maneuvers_per_hour == 12.0


async def test_find_best_cycling_training_window_validates_requirements_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    ctx = SimpleNamespace(
        get_state=AsyncMock(return_value=_config())
    )

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError(
                "ORS client must not be created for invalid requirements"
            )

    monkeypatch.setattr(
        routing,
        "OpenRouteServiceClient",
        UnexpectedClient,
    )

    result = await routing.find_best_cycling_training_window(
        locations=["A", "B"],
        min_asphalt_percentage=120.0,
        ctx=ctx,
    )

    response = json.loads(result)

    assert response["error"]["type"] == "validation_error"
    assert (
        "percentage requirements"
        in response["error"]["message"]
    )


async def test_find_cycling_training_route_validates_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        candidate_count=1,
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "between 2 and 10" in response["error"]["message"]


async def test_find_cycling_training_route_validates_session_requirements_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        max_warmup_maneuvers_per_hour=-1.0,
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "rate requirements" in response["error"]["message"]


async def test_find_cycling_training_route_validates_deduplication_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        deduplication_overlap_threshold_percentage=101.0,
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "between 0 and 100" in response["error"]["message"]


async def test_find_cycling_training_route_validates_duration_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        target_duration_minutes=0.0,
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "target_duration_minutes" in response["error"]["message"]


async def test_find_cycling_training_route_validates_avoid_features_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        avoid_features=["steps", "steps"],
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "duplicates" in response["error"]["message"]


async def test_find_cycling_training_route_validates_multiblock_duration_before_ors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    class UnexpectedClient:
        def __init__(self, _config: ICUConfig) -> None:
            raise AssertionError("ORS client must not be created")

    monkeypatch.setattr(routing, "OpenRouteServiceClient", UnexpectedClient)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        training_repetitions=3,
        training_durations_minutes=[8.0, 12.0],
        ctx=ctx,
    )

    response = json.loads(result)
    assert response["error"]["type"] == "validation_error"
    assert "exactly one work-block duration" in response["error"]["message"]


async def test_find_cycling_training_route_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    import intervals_icu_mcp.tools.routing as routing

    origin = _resolved_location("Start", 2.63, 39.59)
    first = SimpleNamespace(
        candidate=SimpleNamespace(
            candidate_id="round-trip-1",
            route=object(),
        ),
        best_training_window=SimpleNamespace(window=object()),
        best_training_block_sequence=object(),
    )
    second = object()
    search = AsyncMock(
        return_value=CyclingRouteCandidateSearchResult(
            ranked=(first, second),
            candidates_generated=2,
            candidates_after_distance_filter=2,
            candidates_after_duration_filter=2,
            candidates_after_deduplication=2,
        )
    )

    class FakeClient:
        def __init__(self, _config: ICUConfig) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            return None

    monkeypatch.setattr(routing, "OpenRouteServiceClient", FakeClient)
    resolve = AsyncMock(return_value=origin)
    monkeypatch.setattr(routing, "resolve_location", resolve)
    monkeypatch.setattr(
        routing,
        "find_cycling_training_route_candidates",
        search,
    )
    monkeypatch.setattr(
        routing,
        "serialize_cycling_route_candidate_analysis",
        lambda analysis: {"candidate": "first" if analysis is first else "second"},
    )
    gpx_content = b"<?xml version='1.0'?><gpx/>"
    serialize_gpx = Mock(return_value=gpx_content)
    monkeypatch.setattr(routing, "cycling_route_to_gpx", serialize_gpx)
    ctx = SimpleNamespace(get_state=AsyncMock(return_value=_config()))

    result = await routing.find_cycling_training_route(
        start_location="Start",
        target_distance_km=50.0,
        target_duration_minutes=90.0,
        training_durations_minutes=[30.0],
        training_repetitions=3,
        recovery_min_minutes=5.0,
        recovery_max_minutes=10.0,
        candidate_count=2,
        max_duration_deviation_percentage=20.0,
        avoid_features=["ferries", "fords", "steps"],
        include_gpx=True,
        max_warmup_maneuvers_per_hour=8.0,
        ctx=ctx,
    )

    from intervals_icu_mcp.gpx_delivery import response_text_and_resources

    response_text, resources = response_text_and_resources(result)
    response = json.loads(response_text)
    assert len(resources) == 1
    assert response["data"]["best_route"] == {
        "candidate": "first",
        "gpx": {
            "format": "GPX 1.1",
            "encoding": "base64",
            "size_bytes": len(gpx_content),
            "content_base64": base64.b64encode(gpx_content).decode("ascii"),
        },
    }
    assert response["data"]["alternatives"] == [{"candidate": "second"}]
    assert response["metadata"]["candidate_count_eligible"] == 2
    assert response["metadata"]["candidates_generated"] == 2
    assert response["metadata"]["candidates_after_distance_filter"] == 2
    assert response["metadata"]["candidates_after_duration_filter"] == 2
    assert response["metadata"]["candidates_after_deduplication"] == 2
    assert response["metadata"]["deduplication"] == {
        "overlap_threshold_percentage": 90.0,
        "resample_spacing_m": 100.0,
        "proximity_m": 50.0,
    }
    assert response["metadata"]["target_duration_minutes"] == 90.0
    assert response["metadata"]["max_duration_deviation_percentage"] == 20.0
    assert response["metadata"]["gpx_included"] is True
    assert response["metadata"]["avoid_features"] == [
        "ferries",
        "fords",
        "steps",
    ]
    assert response["metadata"]["training_block_spec"] == {
        "work_duration_s": 1800.0,
        "repetitions": 3,
        "recovery_min_s": 300.0,
        "recovery_max_s": 600.0,
    }
    assert response["metadata"]["session_eligibility_requirements"] == {
        "max_warmup_elevation_gain_rate_m_per_hour": None,
        "max_warmup_gradient_percentage": None,
        "max_warmup_maneuvers_per_hour": 8.0,
        "min_warmup_asphalt_percentage": None,
        "max_warmup_footway_percentage": None,
        "max_cooldown_elevation_gain_rate_m_per_hour": None,
        "max_cooldown_gradient_percentage": None,
        "max_cooldown_maneuvers_per_hour": None,
        "min_cooldown_asphalt_percentage": None,
        "max_cooldown_footway_percentage": None,
    }
    resolve.assert_awaited_once()
    search.assert_awaited_once()
    assert search.await_args.kwargs["target_distance_m"] == 50_000.0
    assert search.await_args.kwargs["durations_s"] == (1800.0,)
    assert search.await_args.kwargs["avoid_features"] == (
        "ferries",
        "fords",
        "steps",
    )
    assert search.await_args.kwargs["training_block_spec"] == TrainingBlockSpec(
        work_duration_s=1800.0,
        repetitions=3,
        recovery_min_s=300.0,
        recovery_max_s=600.0,
    )
    assert search.await_args.kwargs["target_duration_s"] == 5400.0
    assert search.await_args.kwargs["max_duration_deviation_percentage"] == 20.0
    assert (
        search.await_args.kwargs[
            "deduplication_overlap_threshold_percentage"
        ]
        == 90.0
    )
    assert (
        search.await_args.kwargs[
            "session_requirements"
        ].max_warmup_maneuvers_per_hour
        == 8.0
    )
    serialize_gpx.assert_called_once_with(
        first.candidate.route,
        name="Cycling training route round-trip-1",
        description="Generated by intervals-icu-mcp with training markers.",
        training_window=None,
        training_block_sequence=first.best_training_block_sequence,
    )
