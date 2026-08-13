"""Tests for native MCP delivery of generated GPX bytes."""

import base64

from fastmcp import Client, FastMCP
from mcp.types import BlobResourceContents, EmbeddedResource

from intervals_icu_mcp.gpx_delivery import attach_gpx_resource


async def test_fastmcp_delivers_gpx_as_embedded_binary_resource() -> None:
    content = b"<?xml version='1.0'?><gpx><trk><trkseg/></trk></gpx>"
    server = FastMCP("gpx-delivery-contract")

    @server.tool()
    def download_route():
        return attach_gpx_resource(
            '{"data":{"size_bytes":55}}',
            content=content,
            filename="cycling-training-route-round-trip-1.gpx",
        )

    async with Client(server) as client:
        result = await client.call_tool("download_route")

    resources = [block for block in result.content if isinstance(block, EmbeddedResource)]
    assert len(resources) == 1
    resource = resources[0].resource
    assert isinstance(resource, BlobResourceContents)
    assert resource.mimeType == "application/gpx+xml"
    assert str(resource.uri) == ("file:///cycling-training-route-round-trip-1.gpx")
    delivered = base64.b64decode(resource.blob, validate=True)
    assert delivered == content
    assert len(delivered) == len(content)
