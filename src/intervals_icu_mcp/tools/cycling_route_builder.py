"""Generic ordered-point road-cycling route builder."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import asdict
from typing import Annotated, Any

from fastmcp import Context

from ..auth import ICUConfig
from ..gpx_delivery import CyclingToolResponse, attach_gpx_resource
from ..openrouteservice_client import OpenRouteServiceAPIError, OpenRouteServiceClient
from ..response_builder import ResponseBuilder
from .routing import (
    LocationResolutionError,
    ResolvedLocation,
    RouteParsingError,
    build_cycling_route,
    calculate_cycling_route_quality_metrics,
    calculate_route_timeline,
    cycling_route_to_gpx,
    resolve_location,
)


def _serialize_resolved_location(location: ResolvedLocation) -> dict[str, Any]:
    """Expose original, resolved and snapped routing coordinates explicitly."""

    return {
        "input": location.input_value,
        "source": location.source,
        "resolved_name": location.label,
        "resolved_coordinates": {
            "longitude": location.original_longitude,
            "latitude": location.original_latitude,
        },
        "routing_coordinates": {
            "longitude": location.longitude,
            "latitude": location.latitude,
        },
        "snapped_distance_meters": location.snapped_distance_m,
    }


def _route_identifier(locations: list[ResolvedLocation]) -> str:
    """Return a stable short identifier for ordered final routing points."""

    coordinate_key = ";".join(
        f"{location.longitude:.7f},{location.latitude:.7f}" for location in locations
    )
    return hashlib.sha256(coordinate_key.encode("ascii")).hexdigest()[:12]


async def build_ordered_cycling_route(
    locations: Annotated[
        list[str],
        (
            "Ordered route points as names, addresses or "
            "'latitude,longitude' coordinates. Repeat the first point last "
            "to request an explicit circular route."
        ),
    ],
    include_gpx: Annotated[
        bool,
        "Attach the complete route as a downloadable GPX 1.1 resource.",
    ] = False,
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
        "Maximum cycling-road network snap radius in meters.",
    ] = 350.0,
    ctx: Context | None = None,
) -> CyclingToolResponse:
    """Build one cycling-road route through ordered points without training analysis."""

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
    if any(not location.strip() for location in locations):
        return ResponseBuilder.build_error_response(
            "Route locations must not be empty.",
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
                    location,
                    country=country,
                    focus_lon=focus_longitude,
                    focus_lat=focus_latitude,
                    snap_radius_m=snap_radius_m,
                )
                for location in locations
            ]
            route = await build_cycling_route(client, resolved_locations)

        timeline = calculate_route_timeline(route)
        quality = calculate_cycling_route_quality_metrics(route, timeline)
        route_id = _route_identifier(resolved_locations)
        serialized_route: dict[str, Any] = {
            "id": route_id,
            "distance_meters": route.distance_m,
            "duration_seconds": route.duration_s,
            "elevation_gain_meters": route.elevation_gain_m,
            "elevation_loss_meters": route.elevation_loss_m,
            "waypoint_indices": list(route.waypoint_indices),
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [point.longitude, point.latitude, point.elevation_m] for point in route.geometry
                ],
            },
        }
        gpx_content: bytes | None = None
        if include_gpx:
            gpx_content = cycling_route_to_gpx(
                route,
                name=f"Cycling route {route_id}",
                description="Ordered-point route generated by intervals-icu-mcp.",
            )
            serialized_route["gpx"] = {
                "format": "GPX 1.1",
                "encoding": "base64",
                "size_bytes": len(gpx_content),
                "content_base64": base64.b64encode(gpx_content).decode("ascii"),
            }
        response = ResponseBuilder.build_response(
            data={
                "resolved_locations": [
                    _serialize_resolved_location(location) for location in resolved_locations
                ],
                "route": serialized_route,
                "route_quality": asdict(quality),
            },
            metadata={
                "profile": "cycling-road",
                "location_count": len(locations),
                "gpx_included": include_gpx,
            },
            query_type="build_cycling_route",
        )
        if gpx_content is None:
            return response
        return attach_gpx_resource(
            response,
            content=gpx_content,
            filename=f"cycling-route-{route_id}.gpx",
        )
    except LocationResolutionError as exc:
        return ResponseBuilder.build_error_response(
            str(exc), error_type="location_resolution_error"
        )
    except (ValueError, RouteParsingError) as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="validation_error")
    except OpenRouteServiceAPIError as exc:
        return ResponseBuilder.build_error_response(str(exc), error_type="api_error")
    except Exception as exc:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {exc}", error_type="internal_error"
        )
