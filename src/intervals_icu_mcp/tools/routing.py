"""Routing helpers for cycling route generation."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..openrouteservice_client import OpenRouteServiceClient


class LocationResolutionError(Exception):
    """Raised when a routing location cannot be resolved safely."""


@dataclass(frozen=True)
class GeocodeCandidate:
    """One candidate returned by the OpenRouteService geocoder."""

    name: str
    label: str
    longitude: float
    latitude: float
    layer: str | None = None
    locality: str | None = None
    localadmin: str | None = None
    region: str | None = None
    country: str | None = None
    distance_km: float | None = None


@dataclass(frozen=True)
class SnappedLocation:
    """Coordinate adjusted to the routable cycling-road network."""

    longitude: float
    latitude: float
    snapped_distance_m: float


@dataclass(frozen=True)
class ResolvedLocation:
    """Location ready to be used by the routing engine."""

    input_value: str
    source: Literal["coordinates", "geocode"]
    label: str
    original_longitude: float
    original_latitude: float
    longitude: float
    latitude: float
    snapped_distance_m: float


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None



_LOCATION_STOPWORDS = frozenset(
    {
        "de",
        "del",
        "la",
        "las",
        "el",
        "los",
        "the",
        "of",
        "and",
    }
)


def normalize_location_text(value: str) -> str:
    """Normalize location text for conservative name comparison."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())

    characters: list[str] = []

    for character in decomposed:
        if unicodedata.combining(character):
            continue

        if character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")

    tokens = [
        token
        for token in "".join(characters).split()
        if token not in _LOCATION_STOPWORDS
    ]

    return " ".join(tokens)


def _location_tokens(value: str) -> set[str]:
    normalized = normalize_location_text(value)

    if not normalized:
        return set()

    return set(normalized.split())


def split_location_query(value: str) -> tuple[str, tuple[str, ...]]:
    """Split 'name, context...' into primary name and context parts."""

    parts = tuple(
        part.strip()
        for part in value.split(",")
        if part.strip()
    )

    if not parts:
        return "", ()

    return parts[0], parts[1:]


def name_token_coverage(
    query_name: str,
    candidate_name: str,
) -> float:
    """Fraction of significant query-name tokens present in candidate name."""

    query_tokens = _location_tokens(query_name)

    if not query_tokens:
        return 0.0

    candidate_tokens = _location_tokens(candidate_name)

    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _name_token_overlap_count(
    query_name: str,
    candidate_name: str,
) -> int:
    return len(
        _location_tokens(query_name)
        & _location_tokens(candidate_name)
    )


def _candidate_context_tokens(
    candidate: GeocodeCandidate,
) -> set[str]:
    tokens: set[str] = set()

    for value in (
        candidate.locality,
        candidate.localadmin,
        candidate.region,
        candidate.country,
    ):
        if value:
            tokens.update(_location_tokens(value))

    return tokens


def _context_overlap_count(
    candidate: GeocodeCandidate,
    context_parts: tuple[str, ...],
) -> int:
    context_tokens: set[str] = set()

    for part in context_parts:
        context_tokens.update(_location_tokens(part))

    return len(
        context_tokens
        & _candidate_context_tokens(candidate)
    )


def _ambiguous_location_error(
    query: str,
    candidates: list[GeocodeCandidate],
) -> LocationResolutionError:
    labels = "; ".join(
        candidate.label
        for candidate in candidates[:5]
    )

    return LocationResolutionError(
        f"Location '{query}' is ambiguous. "
        f"Candidates: {labels}"
    )


def select_geocode_candidate(
    query: str,
    candidates: list[GeocodeCandidate],
) -> GeocodeCandidate:
    """Select a geocoder candidate only when the result is sufficiently clear."""

    if not candidates:
        raise LocationResolutionError(
            f"No geocoding candidates were found for '{query}'."
        )

    primary_name, context_parts = split_location_query(query)

    if not primary_name:
        raise LocationResolutionError(
            "Location query must not be empty."
        )

    normalized_primary = normalize_location_text(primary_name)

    exact_matches = [
        candidate
        for candidate in candidates
        if normalize_location_text(candidate.name)
        == normalized_primary
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        if context_parts:
            context_scores = {
                index: _context_overlap_count(
                    candidate,
                    context_parts,
                )
                for index, candidate in enumerate(exact_matches)
            }

            best_score = max(context_scores.values())

            if best_score > 0:
                best = [
                    exact_matches[index]
                    for index, score in context_scores.items()
                    if score == best_score
                ]

                if len(best) == 1:
                    return best[0]

        locality_matches = [
            candidate
            for candidate in exact_matches
            if candidate.layer == "locality"
        ]

        if len(locality_matches) == 1:
            return locality_matches[0]

        raise _ambiguous_location_error(
            query,
            exact_matches,
        )

    overlap_scores = {
        index: _name_token_overlap_count(
            primary_name,
            candidate.name,
        )
        for index, candidate in enumerate(candidates)
    }

    best_overlap = max(overlap_scores.values())

    # A single shared token such as "Son" or "Mallorca" is not enough.
    if best_overlap < 2:
        raise _ambiguous_location_error(
            query,
            candidates,
        )

    best_candidates = [
        candidates[index]
        for index, score in overlap_scores.items()
        if score == best_overlap
    ]

    sufficiently_matching = [
        candidate
        for candidate in best_candidates
        if name_token_coverage(
            primary_name,
            candidate.name,
        )
        >= 0.5
    ]

    if len(sufficiently_matching) == 1:
        return sufficiently_matching[0]

    if len(sufficiently_matching) > 1 and context_parts:
        context_scores = {
            index: _context_overlap_count(
                candidate,
                context_parts,
            )
            for index, candidate in enumerate(
                sufficiently_matching
            )
        }

        best_context = max(context_scores.values())

        if best_context > 0:
            best = [
                sufficiently_matching[index]
                for index, score in context_scores.items()
                if score == best_context
            ]

            if len(best) == 1:
                return best[0]

    raise _ambiguous_location_error(
        query,
        best_candidates,
    )


def parse_lat_lon(value: str) -> tuple[float, float] | None:
    """Parse user-facing coordinates in 'latitude,longitude' order."""

    parts = [part.strip() for part in value.strip().split(",")]

    if len(parts) != 2:
        return None

    try:
        latitude = float(parts[0])
        longitude = float(parts[1])
    except ValueError:
        return None

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")

    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")

    return latitude, longitude


def extract_geocode_candidates(
    data: dict[str, Any],
) -> list[GeocodeCandidate]:
    """Extract valid candidates from an ORS geocoding response."""

    raw_features = data.get("features")

    if not isinstance(raw_features, list):
        return []

    features = cast(list[Any], raw_features)
    candidates: list[GeocodeCandidate] = []

    for raw_feature in features:
        if not isinstance(raw_feature, dict):
            continue

        feature = cast(dict[str, Any], raw_feature)
        raw_geometry = feature.get("geometry")
        raw_properties = feature.get("properties")

        if not isinstance(raw_geometry, dict):
            continue

        if not isinstance(raw_properties, dict):
            continue

        geometry = cast(dict[str, Any], raw_geometry)
        properties = cast(dict[str, Any], raw_properties)

        raw_coordinates = geometry.get("coordinates")

        if not isinstance(raw_coordinates, list):
            continue

        coordinates = cast(list[Any], raw_coordinates)

        if len(coordinates) < 2:
            continue

        try:
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
        except (TypeError, ValueError):
            continue

        if not -180 <= longitude <= 180:
            continue

        if not -90 <= latitude <= 90:
            continue

        name = str(properties.get("name") or "").strip()
        label = str(properties.get("label") or name).strip()

        if not name or not label:
            continue

        raw_distance = properties.get("distance")

        try:
            distance_km = (
                float(raw_distance)
                if raw_distance is not None
                else None
            )
        except (TypeError, ValueError):
            distance_km = None

        candidates.append(
            GeocodeCandidate(
                name=name,
                label=label,
                longitude=longitude,
                latitude=latitude,
                layer=_optional_string(properties.get("layer")),
                locality=_optional_string(properties.get("locality")),
                localadmin=_optional_string(properties.get("localadmin")),
                region=_optional_string(properties.get("region")),
                country=_optional_string(properties.get("country")),
                distance_km=distance_km,
            )
        )

    return candidates


async def geocode_location_candidates(
    client: OpenRouteServiceClient,
    value: str,
    *,
    size: int = 5,
    country: str | None = None,
    focus_lon: float | None = None,
    focus_lat: float | None = None,
) -> list[GeocodeCandidate]:
    """Return geocoder candidates without silently selecting one."""

    data = await client.geocode(
        value,
        size=size,
        country=country,
        focus_lon=focus_lon,
        focus_lat=focus_lat,
    )

    return extract_geocode_candidates(data)


async def snap_coordinate(
    client: OpenRouteServiceClient,
    *,
    longitude: float,
    latitude: float,
    radius_m: float = 350.0,
) -> SnappedLocation:
    """Snap one coordinate to the cycling-road network."""

    data = await client.snap(
        [[longitude, latitude]],
        profile="cycling-road",
        radius=radius_m,
    )

    raw_locations = data.get("locations")

    if not isinstance(raw_locations, list) or not raw_locations:
        raise LocationResolutionError(
            "OpenRouteService returned no snapping result for the location."
        )

    locations = cast(list[Any], raw_locations)
    raw_location = locations[0]

    if raw_location is None:
        raise LocationResolutionError(
            f"No cycling-road network was found within {radius_m:g} m "
            "of the location."
        )

    if not isinstance(raw_location, dict):
        raise LocationResolutionError(
            "OpenRouteService returned an unexpected snapping response."
        )

    location = cast(dict[str, Any], raw_location)
    raw_coordinate = location.get("location")

    if not isinstance(raw_coordinate, list):
        raise LocationResolutionError(
            "OpenRouteService returned an invalid snapped coordinate."
        )

    coordinate = cast(list[Any], raw_coordinate)

    if len(coordinate) < 2:
        raise LocationResolutionError(
            "OpenRouteService returned an invalid snapped coordinate."
        )

    try:
        snapped_longitude = float(coordinate[0])
        snapped_latitude = float(coordinate[1])
        snapped_distance = float(
            location.get("snapped_distance") or 0.0
        )
    except (TypeError, ValueError) as exc:
        raise LocationResolutionError(
            "OpenRouteService returned an invalid snapping result."
        ) from exc

    return SnappedLocation(
        longitude=snapped_longitude,
        latitude=snapped_latitude,
        snapped_distance_m=snapped_distance,
    )


async def resolve_coordinate_location(
    client: OpenRouteServiceClient,
    value: str,
    *,
    snap_radius_m: float = 350.0,
) -> ResolvedLocation | None:
    """Resolve direct 'lat,lon' input without using the geocoder."""

    parsed = parse_lat_lon(value)

    if parsed is None:
        return None

    latitude, longitude = parsed

    snapped = await snap_coordinate(
        client,
        longitude=longitude,
        latitude=latitude,
        radius_m=snap_radius_m,
    )

    return ResolvedLocation(
        input_value=value,
        source="coordinates",
        label=value.strip(),
        original_longitude=longitude,
        original_latitude=latitude,
        longitude=snapped.longitude,
        latitude=snapped.latitude,
        snapped_distance_m=snapped.snapped_distance_m,
    )

async def resolve_named_location(
    client: OpenRouteServiceClient,
    value: str,
    *,
    geocode_size: int = 5,
    country: str | None = None,
    focus_lon: float | None = None,
    focus_lat: float | None = None,
    snap_radius_m: float = 350.0,
) -> ResolvedLocation:
    """Resolve a named place conservatively and snap it to cycling-road."""

    query = value.strip()

    if not query:
        raise LocationResolutionError(
            "Location must not be empty."
        )

    candidates = await geocode_location_candidates(
        client,
        query,
        size=geocode_size,
        country=country,
        focus_lon=focus_lon,
        focus_lat=focus_lat,
    )

    selected = select_geocode_candidate(
        query,
        candidates,
    )

    snapped = await snap_coordinate(
        client,
        longitude=selected.longitude,
        latitude=selected.latitude,
        radius_m=snap_radius_m,
    )

    return ResolvedLocation(
        input_value=value,
        source="geocode",
        label=selected.label,
        original_longitude=selected.longitude,
        original_latitude=selected.latitude,
        longitude=snapped.longitude,
        latitude=snapped.latitude,
        snapped_distance_m=snapped.snapped_distance_m,
    )


async def resolve_location(
    client: OpenRouteServiceClient,
    value: str,
    *,
    geocode_size: int = 5,
    country: str | None = None,
    focus_lon: float | None = None,
    focus_lat: float | None = None,
    snap_radius_m: float = 350.0,
) -> ResolvedLocation:
    """Resolve direct coordinates or a named place for routing."""

    coordinate_location = await resolve_coordinate_location(
        client,
        value,
        snap_radius_m=snap_radius_m,
    )

    if coordinate_location is not None:
        return coordinate_location

    return await resolve_named_location(
        client,
        value,
        geocode_size=geocode_size,
        country=country,
        focus_lon=focus_lon,
        focus_lat=focus_lat,
        snap_radius_m=snap_radius_m,
    )
