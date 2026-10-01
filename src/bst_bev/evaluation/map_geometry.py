from __future__ import annotations

from shapely.geometry import (
    LineString,
    Polygon,
)


def map_record_geometry(
    nusc_map,
    layer: str,
    record: dict,
):
    """
    Convert selected nuScenes map records into
    Shapely geometry.
    """

    nodes = {
        node["token"]: node
        for node in nusc_map.node
    }

    if layer in (
        "lane_divider",
        "road_divider",
    ):
        points = [
            (
                nodes[token]["x"],
                nodes[token]["y"],
            )
            for token
            in record["node_tokens"]
        ]

        return LineString(
            points
        )

    if layer == "ped_crossing":

        points = [
            (
                nodes[token]["x"],
                nodes[token]["y"],
            )
            for token
            in record[
                "exterior_node_tokens"
            ]
        ]

        return Polygon(
            points
        )

    raise ValueError(
        f"Unsupported layer: {layer}"
    )