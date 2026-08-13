"""Native MCP delivery helpers for generated cycling GPX files."""

from __future__ import annotations

import base64
from typing import TypeAlias

from mcp.types import BlobResourceContents, EmbeddedResource, TextContent
from pydantic import AnyUrl

CyclingToolContent: TypeAlias = TextContent | EmbeddedResource
CyclingToolResponse: TypeAlias = str | list[CyclingToolContent]


def attach_gpx_resource(
    response: str,
    *,
    content: bytes,
    filename: str,
) -> list[CyclingToolContent]:
    """Attach exact GPX bytes to the normal JSON response as an MCP resource."""

    if not filename.endswith(".gpx") or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a simple .gpx filename")
    resource = BlobResourceContents(
        uri=AnyUrl(f"file:///{filename}"),
        mimeType="application/gpx+xml",
        blob=base64.b64encode(content).decode("ascii"),
    )
    return [
        TextContent(type="text", text=response),
        EmbeddedResource(type="resource", resource=resource),
    ]


def response_text_and_resources(
    response: CyclingToolResponse,
) -> tuple[str, list[EmbeddedResource]]:
    """Separate one delegated JSON response from native attached resources."""

    if isinstance(response, str):
        return response, []
    text_blocks = [block for block in response if isinstance(block, TextContent)]
    if len(text_blocks) != 1:
        raise ValueError("Cycling tool response must contain exactly one text block")
    resources = [block for block in response if isinstance(block, EmbeddedResource)]
    return text_blocks[0].text, resources


def rebuild_response(
    text: str,
    resources: list[EmbeddedResource],
) -> CyclingToolResponse:
    """Preserve delegated native resources after enriching response JSON."""

    if not resources:
        return text
    return [TextContent(type="text", text=text), *resources]
