"""Routing helpers for cycling route generation."""

from __future__ import annotations

import math
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
