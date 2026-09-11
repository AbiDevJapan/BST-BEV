from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from bst_id.encoder import BSTIDEncoder
from nuscenes.scripts.export_poses import derive_latlon


EARTH_CIRCUMFERENCE_M = 40_075_016.686

DEFAULT_MAP_LAYERS = (
    "lane_divider",
    "road_divider",
    "ped_crossing",
)

CellKey = tuple[int, int]


# ============================================================
# Data objects
# ============================================================


@dataclass(frozen=True, slots=True)
class MapPrimitive:
    """
    Reference to one nuScenes vector-map primitive.
    """

    layer: str
    token: str


@dataclass(slots=True)
class PatchQueryResult:
    """
    Result of a BST region query.

    Besides the actual retrieved primitives, this stores
    diagnostics that are useful for experiments.
    """

    primitives: set[MapPrimitive]

    cell_count: int
    sample_count: int

    requested_step_m: float
    actual_step_m: float

    @property
    def tokens(self) -> set[tuple[str, str]]:
        return {
            (primitive.layer, primitive.token)
            for primitive in self.primitives
        }

    @property
    def candidate_count(self) -> int:
        return len(self.primitives)


# ============================================================
# nuScenes XY -> BST
# ============================================================


class BSTSpatialEncoder:
    """
    Converts nuScenes global XY coordinates into BST IDs.

    The bst-id implementation expects geographic
    longitude/latitude rather than nuScenes map coordinates,
    so this class owns that conversion.
    """

    def __init__(
        self,
        location: str,
        zoom: int,
    ):
        if zoom < 1:
            raise ValueError("zoom must be >= 1")

        self.location = location
        self.zoom = zoom

    def xy_to_lonlat(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        fake_pose = {
            "timestamp": 0,
            "translation": [x, y, 0.0],
        }

        result = derive_latlon(
            location=self.location,
            poses=[fake_pose],
        )[0]

        longitude = result["longitude"]
        latitude = result["latitude"]

        return longitude, latitude

    def encode(
        self,
        x: float,
        y: float,
    ) -> CellKey:

        longitude, latitude = self.xy_to_lonlat(
            x,
            y,
        )

        bst_id, bit_len = BSTIDEncoder.encode(
            x=longitude,
            y=latitude,
            f=None,
            t_unix=None,
            zoom_x=self.zoom,
            zoom_y=self.zoom,
        )

        return bst_id, bit_len

    def approximate_cell_width_m(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Approximate the physical width of a Web-Mercator BST
        cell at the given nuScenes position.

        This is intended for sampling-resolution selection,
        not exact geometric reasoning.
        """

        _, latitude = self.xy_to_lonlat(
            x,
            y,
        )

        equator_width = (
            EARTH_CIRCUMFERENCE_M
            / (2 ** self.zoom)
        )

        return (
            equator_width
            * math.cos(math.radians(latitude))
        )

    def recommended_spacing_m(
        self,
        x: float,
        y: float,
        *,
        maximum: float = 1.0,
        fraction: float = 0.5,
    ) -> float:
        """
        Choose sampling spacing relative to BST cell size.

        fraction=0.5 means sample at approximately half
        the cell width.
        """

        if maximum <= 0:
            raise ValueError("maximum must be > 0")

        if not 0 < fraction <= 1:
            raise ValueError(
                "fraction must be in the interval (0, 1]"
            )

        cell_width = self.approximate_cell_width_m(
            x,
            y,
        )

        return min(
            maximum,
            cell_width * fraction,
        )


# ============================================================
# BST HD-map index
# ============================================================


class BSTMapIndex:
    """
    BST-based spatial index for nuScenes vector HD-map
    primitives.

    The index maps:

        BST cell
            ->
        set of map primitives

    rather than storing every sampled coordinate individually.
    """

    def __init__(
        self,
        nusc_map,
        location: str,
        zoom: int = 20,
        *,
        map_layers=DEFAULT_MAP_LAYERS,
        index_spacing_m: float = 1.0,
        adaptive_index_spacing: bool = False,
        sampling_fraction: float = 0.5,
        max_query_samples: int = 7_000_000,
        max_index_samples: int = 10_000_000,
    ):
        self.nusc_map = nusc_map
        self.location = location
        self.zoom = zoom

        self.map_layers = tuple(map_layers)

        self.index_spacing_m = index_spacing_m
        self.adaptive_index_spacing = (
            adaptive_index_spacing
        )

        self.sampling_fraction = sampling_fraction

        self.max_query_samples = max_query_samples
        self.max_index_samples = max_index_samples

        self.encoder = BSTSpatialEncoder(
            location=location,
            zoom=zoom,
        )

        # BST cell -> unique map primitives.
        self.index: dict[
            CellKey,
            set[MapPrimitive],
        ] = defaultdict(set)

        # Fast nuScenes node lookup.
        self.nodes = {
            node["token"]: node
            for node in nusc_map.node
        }

        self.index_sample_count = 0
        self._built = False

    # ========================================================
    # Geometry
    # ========================================================

    def _get_node_tokens(
        self,
        layer: str,
        record: dict,
    ) -> list[str]:

        if layer in (
            "lane_divider",
            "road_divider",
        ):
            return record["node_tokens"]

        if layer == "ped_crossing":
            return record["exterior_node_tokens"]

        raise ValueError(
            f"Unsupported map layer: {layer}"
        )

    def _get_nodes(
        self,
        layer: str,
        record: dict,
    ) -> list[dict]:

        return [
            self.nodes[token]
            for token in self._get_node_tokens(
                layer,
                record,
            )
        ]

    @staticmethod
    def _densify_segment(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        spacing: float,
    ) -> Iterator[tuple[float, float]]:
        """
        Yield regularly spaced samples along a line segment.
        """

        if spacing <= 0:
            raise ValueError(
                "spacing must be > 0"
            )

        dx = x2 - x1
        dy = y2 - y1

        distance = math.hypot(dx, dy)

        if distance == 0:
            yield x1, y1
            return

        steps = max(
            1,
            math.ceil(distance / spacing),
        )

        for i in range(steps + 1):
            t = i / steps

            yield (
                x1 + t * dx,
                y1 + t * dy,
            )

    def _segment_spacing(
        self,
        p1: dict,
        p2: dict,
    ) -> float:
        """
        Select map-geometry sampling resolution.
        """

        if not self.adaptive_index_spacing:
            return self.index_spacing_m

        midpoint_x = (
            p1["x"] + p2["x"]
        ) / 2

        midpoint_y = (
            p1["y"] + p2["y"]
        ) / 2

        return self.encoder.recommended_spacing_m(
            midpoint_x,
            midpoint_y,
            maximum=self.index_spacing_m,
            fraction=self.sampling_fraction,
        )

    def _iter_segments(
        self,
        nodes: list[dict],
        *,
        closed: bool,
    ) -> Iterator[tuple[dict, dict]]:

        for i in range(len(nodes) - 1):
            yield nodes[i], nodes[i + 1]

        # Polygon layers need their final edge.
        if (
            closed
            and len(nodes) > 2
            and nodes[0]["token"] != nodes[-1]["token"]
        ):
            yield nodes[-1], nodes[0]

    def _iter_record_points(
        self,
        layer: str,
        record: dict,
    ) -> Iterator[tuple[float, float]]:
        """
        Densify one map primitive.

        Importantly, this happens ONCE per primitive.
        """

        nodes = self._get_nodes(
            layer,
            record,
        )

        if not nodes:
            return

        if len(nodes) == 1:
            yield (
                nodes[0]["x"],
                nodes[0]["y"],
            )
            return

        closed = layer == "ped_crossing"

        first_segment = True

        for p1, p2 in self._iter_segments(
            nodes,
            closed=closed,
        ):
            spacing = self._segment_spacing(
                p1,
                p2,
            )

            for point_index, point in enumerate(
                self._densify_segment(
                    p1["x"],
                    p1["y"],
                    p2["x"],
                    p2["y"],
                    spacing=spacing,
                )
            ):
                # Avoid indexing the shared endpoint of
                # adjacent segments twice.
                if (
                    not first_segment
                    and point_index == 0
                ):
                    continue

                yield point

            first_segment = False

    # ========================================================
    # Index construction
    # ========================================================

    def build(self) -> "BSTMapIndex":
        """
        Build the BST spatial index.
        """

        self.index.clear()
        self.index_sample_count = 0

        for layer in self.map_layers:

            records = getattr(
                self.nusc_map,
                layer,
            )

            for record in records:

                primitive = MapPrimitive(
                    layer=layer,
                    token=record["token"],
                )

                for x, y in self._iter_record_points(
                    layer,
                    record,
                ):
                    self.index_sample_count += 1

                    if (
                        self.index_sample_count
                        > self.max_index_samples
                    ):
                        raise RuntimeError(
                            "BST index construction exceeded "
                            f"{self.max_index_samples:,} "
                            "geometry samples. "
                            "The requested zoom/adaptive "
                            "spacing is too fine for this "
                            "full-map sampling approach."
                        )

                    cell_key = self.encoder.encode(
                        x,
                        y,
                    )

                    self.index[cell_key].add(
                        primitive
                    )

        self._built = True

        return self

    def _ensure_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "BSTMapIndex has not been built. "
                "Call index.build() first."
            )

    # ========================================================
    # Point query
    # ========================================================

    def query_xy(
        self,
        x: float,
        y: float,
    ) -> set[MapPrimitive]:

        self._ensure_built()

        cell_key = self.encoder.encode(
            x,
            y,
        )

        return set(
            self.index.get(
                cell_key,
                set(),
            )
        )

    # ========================================================
    # Patch query
    # ========================================================

    def adaptive_query_step(
        self,
        center_x: float,
        center_y: float,
        *,
        maximum: float = 1.0,
    ) -> float:
        """
        Query spacing based on current BST resolution.

        By default:

            step <= 1 metre
            step <= approximately half BST cell width
        """

        return self.encoder.recommended_spacing_m(
            center_x,
            center_y,
            maximum=maximum,
            fraction=self.sampling_fraction,
        )

    def query_patch(
        self,
        center_x: float,
        center_y: float,
        *,
        radius: float = 30.0,
        step_m: float | None = None,
        max_step_m: float = 1.0,
    ) -> PatchQueryResult:
        """
        Query BST cells touched by a square region.

        If step_m is None, adaptive spacing is selected
        automatically from the BST resolution.

        Note:
            This is still a sampling-based implementation.
            It is intended for validating BST retrieval.
            It is not the final hierarchical region-cover
            algorithm.
        """

        self._ensure_built()

        if radius <= 0:
            raise ValueError(
                "radius must be > 0"
            )

        if step_m is None:
            requested_step = (
                self.adaptive_query_step(
                    center_x,
                    center_y,
                    maximum=max_step_m,
                )
            )
        else:
            if step_m <= 0:
                raise ValueError(
                    "step_m must be > 0"
                )

            requested_step = step_m

        x_min = center_x - radius
        x_max = center_x + radius

        y_min = center_y - radius
        y_max = center_y + radius

        width = 2 * radius

        # Dividing the complete width into N intervals
        # guarantees both patch boundaries are sampled.
        intervals = max(
            1,
            math.ceil(
                width / requested_step
            ),
        )

        actual_step = (
            width / intervals
        )

        samples_per_axis = intervals + 1

        sample_count = (
            samples_per_axis
            * samples_per_axis
        )

        if sample_count > self.max_query_samples:
            raise RuntimeError(
                "BST patch query would require "
                f"{sample_count:,} samples "
                f"(zoom={self.zoom}, "
                f"step≈{actual_step:.4f} m). "
                "This is too expensive for the "
                "sampling-based query implementation. "
                "Use a coarser zoom or move to direct "
                "BST region/cell enumeration."
            )

        cell_keys: set[CellKey] = set()

        for ix in range(samples_per_axis):

            x = (
                x_min
                + ix * actual_step
            )

            for iy in range(samples_per_axis):

                y = (
                    y_min
                    + iy * actual_step
                )

                cell_keys.add(
                    self.encoder.encode(
                        x,
                        y,
                    )
                )

        primitives: set[
            MapPrimitive
        ] = set()

        for cell_key in cell_keys:

            primitives.update(
                self.index.get(
                    cell_key,
                    set(),
                )
            )

        return PatchQueryResult(
            primitives=primitives,
            cell_count=len(cell_keys),
            sample_count=sample_count,
            requested_step_m=requested_step,
            actual_step_m=actual_step,
        )

    def query_patch_tokens(
        self,
        center_x: float,
        center_y: float,
        *,
        radius: float = 30.0,
        step_m: float | None = None,
        max_step_m: float = 1.0,
    ) -> set[tuple[str, str]]:

        result = self.query_patch(
            center_x,
            center_y,
            radius=radius,
            step_m=step_m,
            max_step_m=max_step_m,
        )

        return result.tokens

    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def primitive_count(self) -> int:
        """
        Number of unique primitives represented by the index.
        """

        primitives = set()

        for values in self.index.values():
            primitives.update(values)

        return len(primitives)

    def cell_width_at(
        self,
        x: float,
        y: float,
    ) -> float:

        return self.encoder.approximate_cell_width_m(
            x,
            y,
        )

    def __len__(self) -> int:
        """
        Number of occupied BST cells.
        """

        return len(self.index)