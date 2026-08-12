"""Routing helpers for cycling route generation."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, cast

from fastmcp import Context

from ..auth import ICUConfig
from ..openrouteservice_client import (
    OpenRouteServiceAPIError,
    OpenRouteServiceClient,
)
from ..response_builder import ResponseBuilder


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


def _select_strongly_nearest(
    candidates: list[GeocodeCandidate],
) -> GeocodeCandidate | None:
    """Select by proximity only when one candidate is overwhelmingly nearer."""

    if len(candidates) < 2:
        return candidates[0] if candidates else None

    if any(candidate.distance_km is None for candidate in candidates):
        return None

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.distance_km
            if candidate.distance_km is not None
            else float("inf")
        ),
    )

    nearest = ordered[0]
    second = ordered[1]

    assert nearest.distance_km is not None
    assert second.distance_km is not None

    distance_gap_km = second.distance_km - nearest.distance_km

    if (
        distance_gap_km >= 50.0
        and second.distance_km >= nearest.distance_km * 3.0
    ):
        return nearest

    return None


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

        if len(locality_matches) > 1:
            strongly_nearest = _select_strongly_nearest(
                locality_matches
            )

            if strongly_nearest is not None:
                return strongly_nearest

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


class RouteParsingError(Exception):
    """Raised when an ORS directions response cannot be parsed safely."""


@dataclass(frozen=True)
class RouteCoordinate:
    """One point in a routed geometry."""

    longitude: float
    latitude: float
    elevation_m: float | None = None


@dataclass(frozen=True)
class CyclingRoute:
    """Normalized cycling route returned by the routing engine."""

    distance_m: float
    duration_s: float
    elevation_gain_m: float | None
    elevation_loss_m: float | None
    ors_ascent_m: float | None
    ors_descent_m: float | None
    geometry: tuple[RouteCoordinate, ...]
    waypoint_indices: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    extras: dict[str, Any]


_EARTH_RADIUS_M = 6_371_008.8
_ELEVATION_RESAMPLE_M = 25.0
_ELEVATION_SMOOTHING_WINDOW_M = 200.0


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_distance_m(
    first: RouteCoordinate,
    second: RouteCoordinate,
) -> float:
    lat1 = math.radians(first.latitude)
    lat2 = math.radians(second.latitude)

    delta_lat = lat2 - lat1
    delta_lon = math.radians(
        second.longitude - first.longitude
    )

    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2.0) ** 2
    )

    return (
        2.0
        * _EARTH_RADIUS_M
        * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    )


def _geometry_cumulative_distances(
    coordinates: tuple[RouteCoordinate, ...],
) -> list[float]:
    distances = [0.0]

    for previous, current in zip(
        coordinates,
        coordinates[1:],
        strict=False,
    ):
        distances.append(
            distances[-1]
            + _haversine_distance_m(previous, current)
        )

    return distances


def _resample_elevations(
    coordinates: tuple[RouteCoordinate, ...],
    *,
    spacing_m: float = _ELEVATION_RESAMPLE_M,
) -> list[float]:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be greater than zero")

    if len(coordinates) < 2:
        return []

    elevations = [
        coordinate.elevation_m
        for coordinate in coordinates
    ]

    if any(elevation is None for elevation in elevations):
        return []

    cumulative = _geometry_cumulative_distances(coordinates)
    total_distance = cumulative[-1]

    if total_distance <= 0:
        return []

    numeric_elevations = [
        float(elevation)
        for elevation in elevations
        if elevation is not None
    ]

    targets: list[float] = []
    target = 0.0

    while target < total_distance:
        targets.append(target)
        target += spacing_m

    if not targets or targets[-1] < total_distance:
        targets.append(total_distance)

    result: list[float] = []
    segment_index = 0

    for target_distance in targets:
        while (
            segment_index + 1 < len(cumulative)
            and cumulative[segment_index + 1]
            < target_distance
        ):
            segment_index += 1

        if segment_index + 1 >= len(cumulative):
            result.append(numeric_elevations[-1])
            continue

        start_distance = cumulative[segment_index]
        end_distance = cumulative[segment_index + 1]

        start_elevation = numeric_elevations[segment_index]
        end_elevation = numeric_elevations[segment_index + 1]

        segment_distance = end_distance - start_distance

        if segment_distance <= 0:
            result.append(start_elevation)
            continue

        fraction = (
            target_distance - start_distance
        ) / segment_distance

        result.append(
            start_elevation
            + fraction * (end_elevation - start_elevation)
        )

    return result


def _smooth_elevations(
    elevations: list[float],
    *,
    spacing_m: float = _ELEVATION_RESAMPLE_M,
    window_m: float = _ELEVATION_SMOOTHING_WINDOW_M,
) -> list[float]:
    if not elevations:
        return []

    if spacing_m <= 0:
        raise ValueError("spacing_m must be greater than zero")

    if window_m <= 0:
        raise ValueError("window_m must be greater than zero")

    radius_samples = max(
        0,
        round((window_m / 2.0) / spacing_m),
    )

    prefix = [0.0]

    for elevation in elevations:
        prefix.append(prefix[-1] + elevation)

    smoothed: list[float] = []

    for index in range(len(elevations)):
        start = max(0, index - radius_samples)
        end = min(
            len(elevations),
            index + radius_samples + 1,
        )

        total = prefix[end] - prefix[start]
        smoothed.append(total / (end - start))

    return smoothed


def calculate_elevation_gain_loss(
    coordinates: tuple[RouteCoordinate, ...],
    *,
    resample_m: float = _ELEVATION_RESAMPLE_M,
    smoothing_window_m: float = _ELEVATION_SMOOTHING_WINDOW_M,
) -> tuple[float | None, float | None]:
    """Calculate deterministic smoothed elevation gain and loss."""

    resampled = _resample_elevations(
        coordinates,
        spacing_m=resample_m,
    )

    if len(resampled) < 2:
        return None, None

    smoothed = _smooth_elevations(
        resampled,
        spacing_m=resample_m,
        window_m=smoothing_window_m,
    )

    gain = 0.0
    loss = 0.0

    for previous, current in zip(
        smoothed,
        smoothed[1:],
        strict=False,
    ):
        difference = current - previous

        if difference > 0:
            gain += difference
        elif difference < 0:
            loss -= difference

    return gain, loss


def _parse_route_geometry(
    feature: dict[str, Any],
) -> tuple[RouteCoordinate, ...]:
    raw_geometry = feature.get("geometry")

    if not isinstance(raw_geometry, dict):
        raise RouteParsingError(
            "OpenRouteService response has no valid geometry."
        )

    geometry = cast(dict[str, Any], raw_geometry)
    raw_coordinates = geometry.get("coordinates")

    if not isinstance(raw_coordinates, list):
        raise RouteParsingError(
            "OpenRouteService response has no route coordinates."
        )

    coordinate_items = cast(list[Any], raw_coordinates)
    coordinates: list[RouteCoordinate] = []

    for raw_coordinate in coordinate_items:
        if not isinstance(raw_coordinate, list):
            raise RouteParsingError(
                "OpenRouteService returned an invalid route coordinate."
            )

        coordinate = cast(list[Any], raw_coordinate)

        if len(coordinate) < 2:
            raise RouteParsingError(
                "OpenRouteService returned an incomplete route coordinate."
            )

        longitude = _numeric_value(coordinate[0])
        latitude = _numeric_value(coordinate[1])

        if longitude is None or latitude is None:
            raise RouteParsingError(
                "OpenRouteService returned a non-numeric route coordinate."
            )

        if not -180 <= longitude <= 180:
            raise RouteParsingError(
                "OpenRouteService returned an invalid longitude."
            )

        if not -90 <= latitude <= 90:
            raise RouteParsingError(
                "OpenRouteService returned an invalid latitude."
            )

        elevation = (
            _numeric_value(coordinate[2])
            if len(coordinate) >= 3
            else None
        )

        coordinates.append(
            RouteCoordinate(
                longitude=longitude,
                latitude=latitude,
                elevation_m=elevation,
            )
        )

    if len(coordinates) < 2:
        raise RouteParsingError(
            "OpenRouteService route must contain at least two coordinates."
        )

    return tuple(coordinates)


def _parse_waypoint_indices(
    properties: dict[str, Any],
    *,
    geometry_size: int,
) -> tuple[int, ...]:
    raw_waypoints = properties.get("way_points")

    if not isinstance(raw_waypoints, list):
        return ()

    waypoint_items = cast(list[Any], raw_waypoints)
    indices: list[int] = []

    for value in waypoint_items:
        numeric = _numeric_value(value)

        if numeric is None or not numeric.is_integer():
            raise RouteParsingError(
                "OpenRouteService returned an invalid waypoint index."
            )

        index = int(numeric)

        if index < 0 or index >= geometry_size:
            raise RouteParsingError(
                "OpenRouteService returned an out-of-range waypoint index."
            )

        indices.append(index)

    return tuple(indices)


def _parse_segments(
    properties: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_segments = properties.get("segments")

    if not isinstance(raw_segments, list):
        return ()

    segment_items = cast(list[Any], raw_segments)
    segments: list[dict[str, Any]] = []

    for raw_segment in segment_items:
        if not isinstance(raw_segment, dict):
            raise RouteParsingError(
                "OpenRouteService returned an invalid route segment."
            )

        segments.append(
            dict(cast(dict[str, Any], raw_segment))
        )

    return tuple(segments)


def parse_cycling_route_response(
    data: dict[str, Any],
) -> CyclingRoute:
    """Normalize one ORS GeoJSON directions response."""

    raw_features = data.get("features")

    if not isinstance(raw_features, list):
        raise RouteParsingError(
            "OpenRouteService directions response has no features."
        )

    features = cast(list[Any], raw_features)

    if not features:
        raise RouteParsingError(
            "OpenRouteService directions response is empty."
        )

    raw_feature = features[0]

    if not isinstance(raw_feature, dict):
        raise RouteParsingError(
            "OpenRouteService returned an invalid route feature."
        )

    feature = cast(dict[str, Any], raw_feature)
    raw_properties = feature.get("properties")

    if not isinstance(raw_properties, dict):
        raise RouteParsingError(
            "OpenRouteService route has no valid properties."
        )

    properties = cast(dict[str, Any], raw_properties)
    raw_summary = properties.get("summary")

    if not isinstance(raw_summary, dict):
        raise RouteParsingError(
            "OpenRouteService route has no valid summary."
        )

    summary = cast(dict[str, Any], raw_summary)

    distance_m = _numeric_value(summary.get("distance"))
    duration_s = _numeric_value(summary.get("duration"))

    if distance_m is None or distance_m < 0:
        raise RouteParsingError(
            "OpenRouteService returned an invalid route distance."
        )

    if duration_s is None or duration_s < 0:
        raise RouteParsingError(
            "OpenRouteService returned an invalid route duration."
        )

    geometry = _parse_route_geometry(feature)

    elevation_gain_m, elevation_loss_m = (
        calculate_elevation_gain_loss(geometry)
    )

    raw_extras = properties.get("extras")

    extras = (
        dict(cast(dict[str, Any], raw_extras))
        if isinstance(raw_extras, dict)
        else {}
    )

    return CyclingRoute(
        distance_m=distance_m,
        duration_s=duration_s,
        elevation_gain_m=elevation_gain_m,
        elevation_loss_m=elevation_loss_m,
        ors_ascent_m=_numeric_value(summary.get("ascent")),
        ors_descent_m=_numeric_value(summary.get("descent")),
        geometry=geometry,
        waypoint_indices=_parse_waypoint_indices(
            properties,
            geometry_size=len(geometry),
        ),
        segments=_parse_segments(properties),
        extras=extras,
    )


_ROUTING_EXTRA_INFO = (
    "surface",
    "waytype",
    "steepness",
    "suitability",
)


async def build_cycling_route(
    client: OpenRouteServiceClient,
    locations: list[ResolvedLocation],
) -> CyclingRoute:
    """Build one road-cycling route through already resolved locations."""

    if len(locations) < 2:
        raise ValueError(
            "At least two resolved locations are required to build a route."
        )

    coordinates = [
        [
            location.longitude,
            location.latitude,
        ]
        for location in locations
    ]

    data = await client.directions(
        coordinates,
        profile="cycling-road",
        elevation=True,
        instructions=True,
        extra_info=list(_ROUTING_EXTRA_INFO),
    )

    return parse_cycling_route_response(data)


@dataclass(frozen=True)
class RouteExtraValue:
    """Distance covered by one ORS extra-info value."""

    value: int
    distance_m: float
    percentage: float


@dataclass(frozen=True)
class RouteExtraDistribution:
    """Distance-weighted distribution for one ORS extra-info category."""

    name: str
    geometry_distance_m: float
    classified_distance_m: float
    unclassified_distance_m: float
    values: tuple[RouteExtraValue, ...]


def calculate_extra_distribution(
    route: CyclingRoute,
    extra_name: str,
) -> RouteExtraDistribution:
    """Calculate an ORS extra-info distribution using real geometry distance."""

    cumulative = _geometry_cumulative_distances(
        route.geometry
    )

    geometry_distance_m = (
        cumulative[-1]
        if cumulative
        else 0.0
    )

    raw_extra = route.extras.get(extra_name)

    if not isinstance(raw_extra, dict):
        return RouteExtraDistribution(
            name=extra_name,
            geometry_distance_m=geometry_distance_m,
            classified_distance_m=0.0,
            unclassified_distance_m=geometry_distance_m,
            values=(),
        )

    extra = cast(dict[str, Any], raw_extra)
    raw_values = extra.get("values")

    if not isinstance(raw_values, list):
        return RouteExtraDistribution(
            name=extra_name,
            geometry_distance_m=geometry_distance_m,
            classified_distance_m=0.0,
            unclassified_distance_m=geometry_distance_m,
            values=(),
        )

    value_ranges = cast(list[Any], raw_values)

    parsed_ranges: list[tuple[int, int, int]] = []

    for raw_range in value_ranges:
        if not isinstance(raw_range, list):
            raise RouteParsingError(
                f"OpenRouteService returned an invalid {extra_name} range."
            )

        range_items = cast(list[Any], raw_range)

        if len(range_items) < 3:
            raise RouteParsingError(
                f"OpenRouteService returned an incomplete {extra_name} range."
            )

        start_value = _numeric_value(range_items[0])
        end_value = _numeric_value(range_items[1])
        category_value = _numeric_value(range_items[2])

        if (
            start_value is None
            or end_value is None
            or category_value is None
            or not start_value.is_integer()
            or not end_value.is_integer()
            or not category_value.is_integer()
        ):
            raise RouteParsingError(
                f"OpenRouteService returned a non-integer {extra_name} range."
            )

        start_index = int(start_value)
        end_index = int(end_value)
        value = int(category_value)

        if (
            start_index < 0
            or end_index < start_index
            or end_index >= len(route.geometry)
        ):
            raise RouteParsingError(
                f"OpenRouteService returned an out-of-range {extra_name} interval."
            )

        parsed_ranges.append(
            (
                start_index,
                end_index,
                value,
            )
        )

    parsed_ranges.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    previous_end: int | None = None

    for start_index, end_index, _ in parsed_ranges:
        if (
            previous_end is not None
            and start_index < previous_end
        ):
            raise RouteParsingError(
                f"OpenRouteService returned overlapping {extra_name} ranges."
            )

        previous_end = end_index

    distances_by_value: dict[int, float] = {}
    classified_distance_m = 0.0

    for start_index, end_index, value in parsed_ranges:
        distance_m = (
            cumulative[end_index]
            - cumulative[start_index]
        )

        classified_distance_m += distance_m

        distances_by_value[value] = (
            distances_by_value.get(value, 0.0)
            + distance_m
        )

    values = tuple(
        RouteExtraValue(
            value=value,
            distance_m=distance_m,
            percentage=(
                distance_m / geometry_distance_m * 100.0
                if geometry_distance_m > 0
                else 0.0
            ),
        )
        for value, distance_m in sorted(
            distances_by_value.items()
        )
    )

    unclassified_distance_m = max(
        0.0,
        geometry_distance_m - classified_distance_m,
    )

    return RouteExtraDistribution(
        name=extra_name,
        geometry_distance_m=geometry_distance_m,
        classified_distance_m=classified_distance_m,
        unclassified_distance_m=unclassified_distance_m,
        values=values,
    )


def calculate_route_extra_distributions(
    route: CyclingRoute,
) -> dict[str, RouteExtraDistribution]:
    """Calculate distributions for all routing extra-info categories."""

    return {
        extra_name: calculate_extra_distribution(
            route,
            extra_name,
        )
        for extra_name in _ROUTING_EXTRA_INFO
    }


@dataclass(frozen=True)
class RouteQualityMetrics:
    """Semantic routing metrics derived from ORS extra-info."""

    asphalt_percentage: float
    unknown_surface_percentage: float
    paving_stones_percentage: float

    road_or_cycleway_percentage: float
    footway_percentage: float

    suitability_7_plus_percentage: float
    suitability_8_plus_percentage: float

    incline_7_plus_percentage: float
    incline_10_plus_percentage: float

    decline_7_plus_percentage: float
    decline_10_plus_percentage: float


def _percentage_for_extra_values(
    distribution: RouteExtraDistribution,
    accepted_values: set[int],
) -> float:
    return sum(
        item.percentage
        for item in distribution.values
        if item.value in accepted_values
    )


def calculate_route_quality_metrics(
    route: CyclingRoute,
) -> RouteQualityMetrics:
    """Calculate semantic quality metrics for road-cycling route ranking."""

    distributions = calculate_route_extra_distributions(
        route
    )

    surface = distributions["surface"]
    waytype = distributions["waytype"]
    suitability = distributions["suitability"]
    steepness = distributions["steepness"]

    return RouteQualityMetrics(
        asphalt_percentage=_percentage_for_extra_values(
            surface,
            {3},
        ),
        unknown_surface_percentage=_percentage_for_extra_values(
            surface,
            {0},
        ),
        paving_stones_percentage=_percentage_for_extra_values(
            surface,
            {14},
        ),
        road_or_cycleway_percentage=_percentage_for_extra_values(
            waytype,
            {1, 2, 3, 6},
        ),
        footway_percentage=_percentage_for_extra_values(
            waytype,
            {7},
        ),
        suitability_7_plus_percentage=_percentage_for_extra_values(
            suitability,
            {7, 8, 9, 10},
        ),
        suitability_8_plus_percentage=_percentage_for_extra_values(
            suitability,
            {8, 9, 10},
        ),
        incline_7_plus_percentage=_percentage_for_extra_values(
            steepness,
            {3, 4, 5},
        ),
        incline_10_plus_percentage=_percentage_for_extra_values(
            steepness,
            {4, 5},
        ),
        decline_7_plus_percentage=_percentage_for_extra_values(
            steepness,
            {-3, -4, -5},
        ),
        decline_10_plus_percentage=_percentage_for_extra_values(
            steepness,
            {-4, -5},
        ),
    )


@dataclass(frozen=True)
class RouteTimelinePoint:
    """Distance, time and elevation at one route geometry index."""

    geometry_index: int
    distance_m: float
    time_s: float
    elevation_m: float | None


@dataclass(frozen=True)
class RouteTimeline:
    """Cumulative distance/time mapping across route geometry."""

    points: tuple[RouteTimelinePoint, ...]
    geometry_distance_m: float
    duration_s: float


def calculate_route_timeline(
    route: CyclingRoute,
) -> RouteTimeline:
    """Build a geometry timeline using ORS step durations."""

    if not route.geometry:
        raise RouteParsingError(
            "Cannot build a timeline for an empty route geometry."
        )

    cumulative_distance = _geometry_cumulative_distances(
        route.geometry
    )

    times: list[float | None] = [
        None
        for _ in route.geometry
    ]

    times[0] = 0.0

    elapsed_s = 0.0
    previous_end: int | None = None
    usable_steps = 0

    for segment in route.segments:
        raw_steps = segment.get("steps")

        if not isinstance(raw_steps, list):
            raise RouteParsingError(
                "OpenRouteService route segment is missing steps."
            )

        steps = cast(list[Any], raw_steps)

        for raw_step in steps:
            if not isinstance(raw_step, dict):
                raise RouteParsingError(
                    "OpenRouteService returned an invalid route step."
                )

            step = cast(dict[str, Any], raw_step)

            raw_way_points = step.get("way_points")

            if not isinstance(raw_way_points, list):
                raise RouteParsingError(
                    "OpenRouteService returned invalid step waypoints."
                )

            way_points = cast(list[Any], raw_way_points)

            if len(way_points) < 2:
                raise RouteParsingError(
                    "OpenRouteService returned invalid step waypoints."
                )

            start_value = _numeric_value(way_points[0])
            end_value = _numeric_value(way_points[1])
            duration_value = _numeric_value(step.get("duration"))

            if (
                start_value is None
                or end_value is None
                or duration_value is None
                or not start_value.is_integer()
                or not end_value.is_integer()
            ):
                raise RouteParsingError(
                    "OpenRouteService returned invalid step timing data."
                )

            start_index = int(start_value)
            end_index = int(end_value)
            duration_s = float(duration_value)

            if (
                start_index < 0
                or end_index < start_index
                or end_index >= len(route.geometry)
                or duration_s < 0
            ):
                raise RouteParsingError(
                    "OpenRouteService returned an out-of-range route step."
                )

            if previous_end is None:
                if start_index != 0:
                    raise RouteParsingError(
                        "OpenRouteService route steps do not start at geometry index 0."
                    )
            elif start_index != previous_end:
                raise RouteParsingError(
                    "OpenRouteService returned non-contiguous route steps."
                )

            if start_index == end_index:
                previous_end = end_index
                continue

            start_distance = cumulative_distance[start_index]
            end_distance = cumulative_distance[end_index]
            step_geometry_distance = (
                end_distance - start_distance
            )

            for geometry_index in range(
                start_index,
                end_index + 1,
            ):
                if step_geometry_distance > 0:
                    fraction = (
                        cumulative_distance[geometry_index]
                        - start_distance
                    ) / step_geometry_distance
                else:
                    fraction = (
                        geometry_index - start_index
                    ) / (
                        end_index - start_index
                    )

                times[geometry_index] = (
                    elapsed_s
                    + duration_s * fraction
                )

            elapsed_s += duration_s
            previous_end = end_index
            usable_steps += 1

    if usable_steps == 0:
        raise RouteParsingError(
            "OpenRouteService route contains no usable steps."
        )

    if previous_end != len(route.geometry) - 1:
        raise RouteParsingError(
            "OpenRouteService route steps do not cover the full geometry."
        )

    if any(value is None for value in times):
        raise RouteParsingError(
            "OpenRouteService route steps left geometry points without timing."
        )

    points = tuple(
        RouteTimelinePoint(
            geometry_index=index,
            distance_m=cumulative_distance[index],
            time_s=cast(float, times[index]),
            elevation_m=coordinate.elevation_m,
        )
        for index, coordinate in enumerate(route.geometry)
    )

    return RouteTimeline(
        points=points,
        geometry_distance_m=cumulative_distance[-1],
        duration_s=elapsed_s,
    )


def timeline_point_at_time(
    timeline: RouteTimeline,
    time_s: float,
) -> RouteTimelinePoint:
    """Return the geometry point nearest to a requested elapsed route time."""

    if not timeline.points:
        raise RouteParsingError(
            "Cannot query an empty route timeline."
        )

    if time_s < 0:
        raise ValueError("time_s must be non-negative.")

    if time_s > timeline.duration_s:
        raise ValueError(
            "time_s exceeds route timeline duration."
        )

    low = 0
    high = len(timeline.points) - 1

    while low < high:
        mid = (low + high) // 2

        if timeline.points[mid].time_s < time_s:
            low = mid + 1
        else:
            high = mid

    after_index = low

    if after_index == 0:
        return timeline.points[0]

    before = timeline.points[after_index - 1]
    after = timeline.points[after_index]

    if (
        time_s - before.time_s
        <= after.time_s - time_s
    ):
        return before

    return after


@dataclass(frozen=True)
class RouteTrainingWindow:
    """Metrics for a time-bounded training window on a route."""

    start: RouteTimelinePoint
    end: RouteTimelinePoint

    distance_m: float
    duration_s: float

    elevation_gain_m: float | None
    elevation_loss_m: float | None
    net_elevation_gain_m: float | None


def calculate_training_window(
    route: CyclingRoute,
    timeline: RouteTimeline,
    *,
    start_time_s: float,
    duration_s: float,
) -> RouteTrainingWindow:
    """Calculate metrics for a training window defined by route time."""

    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero.")

    end_time_s = start_time_s + duration_s

    if end_time_s > timeline.duration_s:
        raise ValueError(
            "training window exceeds route timeline duration."
        )

    start = timeline_point_at_time(
        timeline,
        start_time_s,
    )
    end = timeline_point_at_time(
        timeline,
        end_time_s,
    )

    if end.geometry_index <= start.geometry_index:
        raise RouteParsingError(
            "Training window does not span route geometry."
        )

    window_geometry = route.geometry[
        start.geometry_index : end.geometry_index + 1
    ]

    elevation_gain_m, elevation_loss_m = (
        calculate_elevation_gain_loss(
            window_geometry
        )
    )

    start_elevation = start.elevation_m
    end_elevation = end.elevation_m

    net_elevation_gain_m = (
        end_elevation - start_elevation
        if (
            start_elevation is not None
            and end_elevation is not None
        )
        else None
    )

    return RouteTrainingWindow(
        start=start,
        end=end,
        distance_m=end.distance_m - start.distance_m,
        duration_s=end.time_s - start.time_s,
        elevation_gain_m=elevation_gain_m,
        elevation_loss_m=elevation_loss_m,
        net_elevation_gain_m=net_elevation_gain_m,
    )


def calculate_extra_distribution_for_geometry_range(
    route: CyclingRoute,
    extra_name: str,
    *,
    start_index: int,
    end_index: int,
) -> RouteExtraDistribution:
    """Calculate an ORS extra distribution clipped to a geometry range."""

    if (
        start_index < 0
        or end_index <= start_index
        or end_index >= len(route.geometry)
    ):
        raise ValueError("Invalid geometry range.")

    cumulative = _geometry_cumulative_distances(
        route.geometry
    )

    geometry_distance_m = (
        cumulative[end_index]
        - cumulative[start_index]
    )

    raw_extra = route.extras.get(extra_name)

    if not isinstance(raw_extra, dict):
        return RouteExtraDistribution(
            name=extra_name,
            geometry_distance_m=geometry_distance_m,
            classified_distance_m=0.0,
            unclassified_distance_m=geometry_distance_m,
            values=(),
        )

    extra = cast(dict[str, Any], raw_extra)
    raw_values = extra.get("values")

    if not isinstance(raw_values, list):
        return RouteExtraDistribution(
            name=extra_name,
            geometry_distance_m=geometry_distance_m,
            classified_distance_m=0.0,
            unclassified_distance_m=geometry_distance_m,
            values=(),
        )

    value_ranges = cast(list[Any], raw_values)

    distances_by_value: dict[int, float] = {}
    classified_distance_m = 0.0

    for raw_range in value_ranges:
        if not isinstance(raw_range, list):
            raise RouteParsingError(
                f"OpenRouteService returned an invalid {extra_name} range."
            )

        range_items = cast(list[Any], raw_range)

        if len(range_items) < 3:
            raise RouteParsingError(
                f"OpenRouteService returned an incomplete {extra_name} range."
            )

        start_value = _numeric_value(range_items[0])
        end_value = _numeric_value(range_items[1])
        category_value = _numeric_value(range_items[2])

        if (
            start_value is None
            or end_value is None
            or category_value is None
            or not start_value.is_integer()
            or not end_value.is_integer()
            or not category_value.is_integer()
        ):
            raise RouteParsingError(
                f"OpenRouteService returned a non-integer {extra_name} range."
            )

        range_start = int(start_value)
        range_end = int(end_value)
        value = int(category_value)

        if (
            range_start < 0
            or range_end < range_start
            or range_end >= len(route.geometry)
        ):
            raise RouteParsingError(
                f"OpenRouteService returned an out-of-range {extra_name} interval."
            )

        overlap_start = max(
            start_index,
            range_start,
        )
        overlap_end = min(
            end_index,
            range_end,
        )

        if overlap_end <= overlap_start:
            continue

        distance_m = (
            cumulative[overlap_end]
            - cumulative[overlap_start]
        )

        classified_distance_m += distance_m

        distances_by_value[value] = (
            distances_by_value.get(value, 0.0)
            + distance_m
        )

    values = tuple(
        RouteExtraValue(
            value=value,
            distance_m=distance_m,
            percentage=(
                distance_m / geometry_distance_m * 100.0
                if geometry_distance_m > 0
                else 0.0
            ),
        )
        for value, distance_m in sorted(
            distances_by_value.items()
        )
    )

    return RouteExtraDistribution(
        name=extra_name,
        geometry_distance_m=geometry_distance_m,
        classified_distance_m=classified_distance_m,
        unclassified_distance_m=max(
            0.0,
            geometry_distance_m - classified_distance_m,
        ),
        values=values,
    )


def calculate_training_window_extra_distributions(
    route: CyclingRoute,
    window: RouteTrainingWindow,
) -> dict[str, RouteExtraDistribution]:
    """Calculate ORS extra distributions inside a training window."""

    return {
        extra_name: calculate_extra_distribution_for_geometry_range(
            route,
            extra_name,
            start_index=window.start.geometry_index,
            end_index=window.end.geometry_index,
        )
        for extra_name in _ROUTING_EXTRA_INFO
    }


def calculate_training_window_quality_metrics(
    route: CyclingRoute,
    window: RouteTrainingWindow,
) -> RouteQualityMetrics:
    """Calculate semantic road-cycling quality metrics inside a training window."""

    distributions = calculate_training_window_extra_distributions(
        route,
        window,
    )

    surface = distributions["surface"]
    waytype = distributions["waytype"]
    suitability = distributions["suitability"]
    steepness = distributions["steepness"]

    return RouteQualityMetrics(
        asphalt_percentage=_percentage_for_extra_values(
            surface,
            {3},
        ),
        unknown_surface_percentage=_percentage_for_extra_values(
            surface,
            {0},
        ),
        paving_stones_percentage=_percentage_for_extra_values(
            surface,
            {14},
        ),
        road_or_cycleway_percentage=_percentage_for_extra_values(
            waytype,
            {1, 2, 3, 6},
        ),
        footway_percentage=_percentage_for_extra_values(
            waytype,
            {7},
        ),
        suitability_7_plus_percentage=_percentage_for_extra_values(
            suitability,
            {7, 8, 9, 10},
        ),
        suitability_8_plus_percentage=_percentage_for_extra_values(
            suitability,
            {8, 9, 10},
        ),
        incline_7_plus_percentage=_percentage_for_extra_values(
            steepness,
            {3, 4, 5},
        ),
        incline_10_plus_percentage=_percentage_for_extra_values(
            steepness,
            {4, 5},
        ),
        decline_7_plus_percentage=_percentage_for_extra_values(
            steepness,
            {-3, -4, -5},
        ),
        decline_10_plus_percentage=_percentage_for_extra_values(
            steepness,
            {-4, -5},
        ),
    )


_MANEUVER_INSTRUCTION_TYPES = frozenset({
    0,   # left
    1,   # right
    2,   # sharp left
    3,   # sharp right
    4,   # slight left
    5,   # slight right
    7,   # enter roundabout
    8,   # exit roundabout
    9,   # U-turn
    12,  # keep left
    13,  # keep right
})

_SHARP_TURN_INSTRUCTION_TYPES = frozenset({2, 3})


@dataclass(frozen=True)
class RouteWindowInterruptionMetrics:
    """Maneuver interruptions inside a route training window."""

    maneuver_count: int
    sharp_turn_count: int
    roundabout_count: int
    u_turn_count: int


def calculate_training_window_interruption_metrics(
    route: CyclingRoute,
    window: RouteTrainingWindow,
) -> RouteWindowInterruptionMetrics:
    """Count meaningful ORS maneuvers whose start lies inside the window."""

    maneuver_count = 0
    sharp_turn_count = 0
    roundabout_count = 0
    u_turn_count = 0

    for segment in route.segments:
        raw_steps = segment.get("steps")

        if not isinstance(raw_steps, list):
            raise RouteParsingError(
                "OpenRouteService route segment is missing steps."
            )

        steps = cast(list[Any], raw_steps)

        for raw_step in steps:
            if not isinstance(raw_step, dict):
                raise RouteParsingError(
                    "OpenRouteService returned an invalid route step."
                )

            step = cast(dict[str, Any], raw_step)

            raw_way_points = step.get("way_points")

            if not isinstance(raw_way_points, list):
                raise RouteParsingError(
                    "OpenRouteService returned invalid step waypoints."
                )

            way_points = cast(list[Any], raw_way_points)

            if len(way_points) < 2:
                raise RouteParsingError(
                    "OpenRouteService returned invalid step waypoints."
                )

            start_value = _numeric_value(way_points[0])
            type_value = _numeric_value(step.get("type"))

            if (
                start_value is None
                or type_value is None
                or not start_value.is_integer()
                or not type_value.is_integer()
            ):
                raise RouteParsingError(
                    "OpenRouteService returned invalid step instruction data."
                )

            start_index = int(start_value)
            instruction_type = int(type_value)

            if (
                start_index <= window.start.geometry_index
                or start_index >= window.end.geometry_index
            ):
                continue

            if instruction_type not in _MANEUVER_INSTRUCTION_TYPES:
                continue

            maneuver_count += 1

            if instruction_type in _SHARP_TURN_INSTRUCTION_TYPES:
                sharp_turn_count += 1

            if instruction_type == 7:
                roundabout_count += 1

            if instruction_type == 9:
                u_turn_count += 1

    return RouteWindowInterruptionMetrics(
        maneuver_count=maneuver_count,
        sharp_turn_count=sharp_turn_count,
        roundabout_count=roundabout_count,
        u_turn_count=u_turn_count,
    )


def generate_training_window_candidates(
    route: CyclingRoute,
    timeline: RouteTimeline,
    *,
    start_time_min_s: float,
    start_time_max_s: float,
    duration_s: float,
    step_s: float = 60.0,
) -> tuple[RouteTrainingWindow, ...]:
    """Generate fixed-duration training windows across a start-time range."""

    if start_time_min_s < 0:
        raise ValueError("start_time_min_s must be non-negative.")

    if start_time_max_s < start_time_min_s:
        raise ValueError(
            "start_time_max_s must be greater than or equal to start_time_min_s."
        )

    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero.")

    if step_s <= 0:
        raise ValueError("step_s must be greater than zero.")

    candidates: list[RouteTrainingWindow] = []

    start_time_s = start_time_min_s

    while start_time_s <= start_time_max_s:
        if start_time_s + duration_s <= timeline.duration_s:
            candidates.append(
                calculate_training_window(
                    route,
                    timeline,
                    start_time_s=start_time_s,
                    duration_s=duration_s,
                )
            )

        start_time_s += step_s

    return tuple(candidates)


@dataclass(frozen=True)
class RouteTrainingWindowAnalysis:
    """Complete analysis of one candidate training window."""

    window: RouteTrainingWindow
    quality: RouteQualityMetrics
    interruptions: RouteWindowInterruptionMetrics


def analyze_training_window(
    route: CyclingRoute,
    window: RouteTrainingWindow,
) -> RouteTrainingWindowAnalysis:
    """Calculate all evaluation metrics for one training window."""

    return RouteTrainingWindowAnalysis(
        window=window,
        quality=calculate_training_window_quality_metrics(
            route,
            window,
        ),
        interruptions=calculate_training_window_interruption_metrics(
            route,
            window,
        ),
    )


def analyze_training_window_candidates(
    route: CyclingRoute,
    windows: tuple[RouteTrainingWindow, ...],
) -> tuple[RouteTrainingWindowAnalysis, ...]:
    """Analyze a sequence of candidate training windows."""

    return tuple(
        analyze_training_window(route, window)
        for window in windows
    )


def _training_window_rank_key(
    analysis: RouteTrainingWindowAnalysis,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return a deterministic lexicographic ranking key for climb windows."""

    elevation_gain_m = analysis.window.elevation_gain_m
    elevation_loss_m = analysis.window.elevation_loss_m

    if elevation_gain_m is None or elevation_loss_m is None:
        raise RouteParsingError(
            "Training window ranking requires elevation gain and loss."
        )

    climbing_balance_m = elevation_gain_m - elevation_loss_m

    return (
        climbing_balance_m,
        elevation_gain_m,
        -elevation_loss_m,
        analysis.quality.asphalt_percentage,
        analysis.quality.road_or_cycleway_percentage,
        analysis.quality.suitability_7_plus_percentage,
        -float(analysis.interruptions.maneuver_count),
        -float(analysis.interruptions.roundabout_count),
    )


def rank_training_window_analyses(
    analyses: tuple[RouteTrainingWindowAnalysis, ...],
) -> tuple[RouteTrainingWindowAnalysis, ...]:
    """Rank climb-window analyses without combining metrics into weighted scores."""

    return tuple(
        sorted(
            analyses,
            key=_training_window_rank_key,
            reverse=True,
        )
    )


def find_best_training_window(
    route: CyclingRoute,
    timeline: RouteTimeline,
    *,
    start_time_min_s: float,
    start_time_max_s: float,
    duration_s: float,
    step_s: float = 60.0,
) -> RouteTrainingWindowAnalysis | None:
    """Find the best climb-oriented training window in a start-time range."""

    windows = generate_training_window_candidates(
        route,
        timeline,
        start_time_min_s=start_time_min_s,
        start_time_max_s=start_time_max_s,
        duration_s=duration_s,
        step_s=step_s,
    )

    if not windows:
        return None

    analyses = analyze_training_window_candidates(
        route,
        windows,
    )

    ranked = rank_training_window_analyses(analyses)

    return ranked[0]


def find_best_training_windows_by_duration(
    route: CyclingRoute,
    timeline: RouteTimeline,
    *,
    start_time_min_s: float,
    start_time_max_s: float,
    durations_s: tuple[float, ...],
    step_s: float = 60.0,
) -> tuple[RouteTrainingWindowAnalysis, ...]:
    """Find the best training window independently for each duration."""

    if not durations_s:
        return ()

    results: list[RouteTrainingWindowAnalysis] = []

    for duration_s in durations_s:
        best = find_best_training_window(
            route,
            timeline,
            start_time_min_s=start_time_min_s,
            start_time_max_s=start_time_max_s,
            duration_s=duration_s,
            step_s=step_s,
        )

        if best is not None:
            results.append(best)

    return tuple(results)


@dataclass(frozen=True)
class RouteTrainingWindowComparisonMetrics:
    """Duration-normalized metrics for comparing training windows."""

    elevation_gain_rate_m_per_hour: float
    elevation_loss_rate_m_per_hour: float
    climbing_balance_rate_m_per_hour: float
    climbing_balance_gradient_percentage: float
    maneuvers_per_hour: float


def calculate_training_window_comparison_metrics(
    analysis: RouteTrainingWindowAnalysis,
) -> RouteTrainingWindowComparisonMetrics:
    """Calculate duration-neutral metrics for a training window."""

    window = analysis.window

    gain = window.elevation_gain_m
    loss = window.elevation_loss_m

    if gain is None or loss is None:
        raise RouteParsingError(
            "Training window comparison requires elevation gain and loss."
        )

    if window.duration_s <= 0:
        raise RouteParsingError(
            "Training window comparison requires positive duration."
        )

    if window.distance_m <= 0:
        raise RouteParsingError(
            "Training window comparison requires positive distance."
        )

    duration_hours = window.duration_s / 3600.0
    climbing_balance_m = gain - loss

    return RouteTrainingWindowComparisonMetrics(
        elevation_gain_rate_m_per_hour=gain / duration_hours,
        elevation_loss_rate_m_per_hour=loss / duration_hours,
        climbing_balance_rate_m_per_hour=(
            climbing_balance_m / duration_hours
        ),
        climbing_balance_gradient_percentage=(
            climbing_balance_m / window.distance_m * 100.0
        ),
        maneuvers_per_hour=(
            analysis.interruptions.maneuver_count / duration_hours
        ),
    )


def _cross_duration_training_window_rank_key(
    analysis: RouteTrainingWindowAnalysis,
) -> tuple[float, float, float, float, float, float]:
    """Return a duration-neutral ranking key for training windows."""

    comparison = calculate_training_window_comparison_metrics(
        analysis,
    )

    return (
        comparison.climbing_balance_rate_m_per_hour,
        comparison.climbing_balance_gradient_percentage,
        -comparison.elevation_loss_rate_m_per_hour,
        analysis.quality.asphalt_percentage,
        analysis.quality.suitability_7_plus_percentage,
        -comparison.maneuvers_per_hour,
    )


def rank_training_windows_across_durations(
    analyses: tuple[RouteTrainingWindowAnalysis, ...],
) -> tuple[RouteTrainingWindowAnalysis, ...]:
    """Rank training windows of different durations using normalized metrics."""

    return tuple(
        sorted(
            analyses,
            key=_cross_duration_training_window_rank_key,
            reverse=True,
        )
    )


def find_best_training_window_across_durations(
    route: CyclingRoute,
    timeline: RouteTimeline,
    *,
    start_time_min_s: float,
    start_time_max_s: float,
    durations_s: tuple[float, ...],
    step_s: float = 60.0,
) -> RouteTrainingWindowAnalysis | None:
    """Find the best training window across multiple candidate durations."""

    analyses = find_best_training_windows_by_duration(
        route,
        timeline,
        start_time_min_s=start_time_min_s,
        start_time_max_s=start_time_max_s,
        durations_s=durations_s,
        step_s=step_s,
    )

    if not analyses:
        return None

    ranked = rank_training_windows_across_durations(
        analyses,
    )

    return ranked[0]


def _serialize_training_window_analysis(
    route: CyclingRoute,
    analysis: RouteTrainingWindowAnalysis,
) -> dict[str, Any]:
    """Serialize one analyzed training window for the public MCP response."""

    window = analysis.window
    comparison = calculate_training_window_comparison_metrics(
        analysis,
    )

    start_coordinate = route.geometry[
        window.start.geometry_index
    ]
    end_coordinate = route.geometry[
        window.end.geometry_index
    ]

    return {
        "duration_seconds": window.duration_s,
        "duration_minutes": window.duration_s / 60.0,
        "start": {
            "time_seconds": window.start.time_s,
            "time_minutes": window.start.time_s / 60.0,
            "distance_meters": window.start.distance_m,
            "geometry_index": window.start.geometry_index,
            "elevation_meters": window.start.elevation_m,
            "longitude": start_coordinate.longitude,
            "latitude": start_coordinate.latitude,
        },
        "end": {
            "time_seconds": window.end.time_s,
            "time_minutes": window.end.time_s / 60.0,
            "distance_meters": window.end.distance_m,
            "geometry_index": window.end.geometry_index,
            "elevation_meters": window.end.elevation_m,
            "longitude": end_coordinate.longitude,
            "latitude": end_coordinate.latitude,
        },
        "distance_meters": window.distance_m,
        "elevation_gain_meters": window.elevation_gain_m,
        "elevation_loss_meters": window.elevation_loss_m,
        "net_elevation_gain_meters": window.net_elevation_gain_m,
        "comparison": asdict(comparison),
        "quality": asdict(analysis.quality),
        "interruptions": asdict(analysis.interruptions),
    }


async def find_best_cycling_training_window(
    locations: Annotated[
        list[str],
        (
            "Ordered route locations. Each item may be a place name "
            "or 'latitude,longitude' coordinates."
        ),
    ],
    start_time_min_minutes: Annotated[
        float,
        "Earliest allowed training-window start, minutes from route start.",
    ] = 20.0,
    start_time_max_minutes: Annotated[
        float,
        "Latest allowed training-window start, minutes from route start.",
    ] = 30.0,
    durations_minutes: Annotated[
        list[float] | None,
        (
            "Candidate continuous training-window durations in minutes. "
            "Defaults to 20, 30 and 40."
        ),
    ] = None,
    step_minutes: Annotated[
        float,
        "Candidate start-time search step in minutes.",
    ] = 1.0,
    country: Annotated[
        str | None,
        "Optional ISO country code used to constrain named-place geocoding.",
    ] = None,
    focus_longitude: Annotated[
        float | None,
        "Optional longitude used to bias named-place geocoding.",
    ] = None,
    focus_latitude: Annotated[
        float | None,
        "Optional latitude used to bias named-place geocoding.",
    ] = None,
    snap_radius_m: Annotated[
        float,
        "Maximum network snap radius in meters for each route location.",
    ] = 350.0,
    ctx: Context | None = None,
) -> str:
    """Find the best continuous climbing-oriented cycling training window.

    Builds a road-cycling route through the ordered locations, evaluates the
    best window independently for each requested duration, then compares those
    duration winners using normalized climbing metrics.
    """

    assert ctx is not None
    config: ICUConfig = await ctx.get_state("config")

    if not config.openrouteservice_api_key.strip():
        return ResponseBuilder.build_error_response(
            "OpenRouteService API is not configured.",
            error_type="configuration_error",
        )

    if len(locations) < 2:
        return ResponseBuilder.build_error_response(
            "At least two route locations are required.",
            error_type="validation_error",
        )

    candidate_durations = (
        durations_minutes
        if durations_minutes is not None
        else [20.0, 30.0, 40.0]
    )

    if not candidate_durations:
        return ResponseBuilder.build_error_response(
            "At least one training-window duration is required.",
            error_type="validation_error",
        )

    if any(duration <= 0 for duration in candidate_durations):
        return ResponseBuilder.build_error_response(
            "Training-window durations must be greater than zero.",
            error_type="validation_error",
        )

    if step_minutes <= 0:
        return ResponseBuilder.build_error_response(
            "step_minutes must be greater than zero.",
            error_type="validation_error",
        )

    if snap_radius_m <= 0:
        return ResponseBuilder.build_error_response(
            "snap_radius_m must be greater than zero.",
            error_type="validation_error",
        )

    try:
        async with OpenRouteServiceClient(config) as client:
            resolved_locations = [
                await resolve_location(
                    client,
                    value,
                    country=country,
                    focus_lon=focus_longitude,
                    focus_lat=focus_latitude,
                    snap_radius_m=snap_radius_m,
                )
                for value in locations
            ]

            route = await build_cycling_route(
                client,
                resolved_locations,
            )

        timeline = calculate_route_timeline(route)

        analyses = find_best_training_windows_by_duration(
            route,
            timeline,
            start_time_min_s=start_time_min_minutes * 60.0,
            start_time_max_s=start_time_max_minutes * 60.0,
            durations_s=tuple(
                duration * 60.0
                for duration in candidate_durations
            ),
            step_s=step_minutes * 60.0,
        )

        if not analyses:
            return ResponseBuilder.build_error_response(
                "No valid training window fits within the generated route.",
                error_type="not_found",
            )

        ranked = rank_training_windows_across_durations(
            analyses,
        )
        best = ranked[0]

        return ResponseBuilder.build_response(
            data={
                "route": {
                    "distance_meters": route.distance_m,
                    "duration_seconds": route.duration_s,
                    "elevation_gain_meters": route.elevation_gain_m,
                    "elevation_loss_meters": route.elevation_loss_m,
                    "resolved_locations": [
                        asdict(location)
                        for location in resolved_locations
                    ],
                },
                "best_training_window": (
                    _serialize_training_window_analysis(
                        route,
                        best,
                    )
                ),
                "best_by_duration": [
                    _serialize_training_window_analysis(
                        route,
                        analysis,
                    )
                    for analysis in analyses
                ],
            },
            metadata={
                "profile": "cycling-road",
                "candidate_duration_minutes": candidate_durations,
                "start_time_range_minutes": [
                    start_time_min_minutes,
                    start_time_max_minutes,
                ],
                "step_minutes": step_minutes,
            },
            query_type="cycling_training_window",
        )

    except LocationResolutionError as exc:
        return ResponseBuilder.build_error_response(
            str(exc),
            error_type="location_resolution_error",
        )
    except (ValueError, RouteParsingError) as exc:
        return ResponseBuilder.build_error_response(
            str(exc),
            error_type="validation_error",
        )
    except OpenRouteServiceAPIError as exc:
        return ResponseBuilder.build_error_response(
            str(exc),
            error_type="api_error",
        )
    except Exception as exc:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {exc}",
            error_type="internal_error",
        )
