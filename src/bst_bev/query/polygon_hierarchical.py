from __future__ import annotations

import math
import time

from dataclasses import dataclass, field

from shapely.geometry import (
    Polygon,
    box,
)

from bst_id.constants import (
    X_RANGE,
    MAX_MERCATOR_LAT,
)

from bst_id.region_algebra import Cell

from bst_bev.query.oriented_region import (
    OrientedBEVRegion,
)


# ============================================================
# Internal descriptor
# ============================================================

@dataclass(
    frozen=True,
    slots=True,
)
class _CellDescriptor:
    x_zoom: int
    x_prefix: int

    y_zoom: int
    y_prefix: int

    def to_cell(self) -> Cell:
        return Cell.from_parts(
            x=(
                self.x_zoom,
                self.x_prefix,
            ),
            y=(
                self.y_zoom,
                self.y_prefix,
            ),
        )


# ============================================================
# Result
# ============================================================

@dataclass(
    frozen=True,
    slots=True,
)
class PolygonHierarchicalCoverResult:

    cells: tuple[Cell, ...]

    target_zoom: int
    min_zoom: int

    cover_time_ms: float

    visited_cell_count: int
    discarded_cell_count: int

    interior_cell_count: int
    boundary_cell_count: int

    leaf_equivalent_count: int

    tile_polygon: Polygon = field(
        repr=False
    )

    @property
    def hierarchical_cell_count(
        self,
    ) -> int:
        return len(
            self.cells
        )

    @property
    def compression_ratio(
        self,
    ) -> float:
        if not self.cells:
            return 0.0

        return (
            self.leaf_equivalent_count
            / self.hierarchical_cell_count
        )

    @property
    def reduction_fraction(
        self,
    ) -> float:
        if self.leaf_equivalent_count == 0:
            return 0.0

        return (
            1.0
            - self.hierarchical_cell_count
            / self.leaf_equivalent_count
        )

    def zoom_distribution(
        self,
    ) -> dict[
        tuple[int, int],
        int,
    ]:
        counts = {}

        for cell in self.cells:

            x_zoom, _ = (
                cell.descriptor("x")
            )

            y_zoom, _ = (
                cell.descriptor("y")
            )

            key = (
                x_zoom,
                y_zoom,
            )

            counts[key] = (
                counts.get(
                    key,
                    0,
                )
                + 1
            )

        return counts


# ============================================================
# Polygon cover
# ============================================================

class HierarchicalPolygonRegionCover:
    """
    Conservative hierarchical BST cover of an arbitrary
    polygon.

    Geometry is evaluated in BST tile coordinates at the
    requested target zoom.

    A cell is:

        outside  -> discarded
        inside   -> retained immediately
        partial  -> recursively subdivided

    Partial target-resolution cells are retained so that the
    resulting cell union conservatively covers the query
    region.

    The encoder object only needs to expose:

        encoder.xy_to_lonlat(x, y)
    """

    OUTSIDE = 0
    INSIDE = 1
    PARTIAL = 2

    def __init__(
        self,
        *,
        encoder,
        min_zoom: int,
        target_zoom: int,
        boundary_step_m: float = 1.0,
        max_visited_cells: int = 100_000,
    ):
        if min_zoom < 0:
            raise ValueError(
                "min_zoom must be >= 0"
            )

        if target_zoom < min_zoom:
            raise ValueError(
                "target_zoom must be >= min_zoom"
            )

        if boundary_step_m <= 0:
            raise ValueError(
                "boundary_step_m must be > 0"
            )

        self.encoder = encoder

        self.min_zoom = min_zoom
        self.target_zoom = target_zoom

        self.boundary_step_m = (
            boundary_step_m
        )

        self.max_visited_cells = (
            max_visited_cells
        )

    # ========================================================
    # Coordinate conversion
    # ========================================================

    @staticmethod
    def _longitude_to_tile_x(
        longitude: float,
        zoom: int,
    ) -> float:
        """
        Continuous equivalent of BST-ID's
        encode_x_tile_index().
        """

        min_val, max_val = X_RANGE

        span = (
            max_val
            - min_val
        )

        wrapped = (
            (
                longitude
                - min_val
            )
            % span
            + min_val
        )

        normalized = (
            wrapped
            - min_val
        ) / span

        return (
            normalized
            * (1 << zoom)
        )

    @staticmethod
    def _latitude_to_tile_y(
        latitude: float,
        zoom: int,
    ) -> float:
        """
        Continuous equivalent of BST-ID's
        encode_y_tile_index().
        """

        latitude = max(
            -MAX_MERCATOR_LAT,
            min(
                MAX_MERCATOR_LAT,
                latitude,
            ),
        )

        lat_rad = math.radians(
            latitude
        )

        mercator = math.log(
            math.tan(lat_rad)
            + 1.0
            / math.cos(lat_rad)
        )

        normalized = (
            1.0
            - mercator
            / math.pi
        ) / 2.0

        return (
            normalized
            * (1 << zoom)
        )

    def _xy_to_tile(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        longitude, latitude = (
            self.encoder.xy_to_lonlat(
                x,
                y,
            )
        )

        tile_x = (
            self._longitude_to_tile_x(
                longitude,
                self.target_zoom,
            )
        )

        tile_y = (
            self._latitude_to_tile_y(
                latitude,
                self.target_zoom,
            )
        )

        return (
            tile_x,
            tile_y,
        )

    # ========================================================
    # Polygon conversion
    # ========================================================

    def _sample_boundary(
        self,
        region: OrientedBEVRegion,
    ) -> list[
        tuple[float, float]
    ]:
        """
        Densify the global-XY rectangle boundary before
        projecting it into Web-Mercator tile space.

        This avoids assuming that a straight global XY edge
        remains perfectly straight after projection.
        """

        corners = list(
            region.global_corners
        )

        samples = []

        for i in range(
            len(corners)
        ):
            x1, y1 = corners[i]

            x2, y2 = corners[
                (i + 1)
                % len(corners)
            ]

            length = math.hypot(
                x2 - x1,
                y2 - y1,
            )

            segment_count = max(
                1,
                math.ceil(
                    length
                    / self.boundary_step_m
                ),
            )

            # Do not include the endpoint here;
            # the next edge supplies it.
            for j in range(
                segment_count
            ):
                t = (
                    j
                    / segment_count
                )

                x = (
                    x1
                    + t
                    * (x2 - x1)
                )

                y = (
                    y1
                    + t
                    * (y2 - y1)
                )

                samples.append(
                    (
                        x,
                        y,
                    )
                )

        return samples

    def _region_to_tile_polygon(
        self,
        region: OrientedBEVRegion,
    ) -> Polygon:

        boundary_xy = (
            self._sample_boundary(
                region
            )
        )

        boundary_tiles = [
            self._xy_to_tile(
                x,
                y,
            )
            for x, y
            in boundary_xy
        ]

        polygon = Polygon(
            boundary_tiles
        )

        if not polygon.is_valid:
            polygon = (
                polygon.buffer(0)
            )

        if polygon.is_empty:
            raise RuntimeError(
                "Projected BEV polygon is empty"
            )

        return polygon

    # ========================================================
    # Cell geometry
    # ========================================================

    def _descriptor_bounds(
        self,
        cell: _CellDescriptor,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Bounds of a mixed-resolution cell expressed in
        target-zoom tile coordinates.
        """

        x_scale = (
            1
            << (
                self.target_zoom
                - cell.x_zoom
            )
        )

        y_scale = (
            1
            << (
                self.target_zoom
                - cell.y_zoom
            )
        )

        x_min = (
            cell.x_prefix
            * x_scale
        )

        x_max = (
            (
                cell.x_prefix
                + 1
            )
            * x_scale
        )

        y_min = (
            cell.y_prefix
            * y_scale
        )

        y_max = (
            (
                cell.y_prefix
                + 1
            )
            * y_scale
        )

        return (
            x_min,
            y_min,
            x_max,
            y_max,
        )

    def _descriptor_polygon(
        self,
        cell: _CellDescriptor,
    ):
        return box(
            *self._descriptor_bounds(
                cell
            )
        )

    def cell_bounds(
        self,
        cell: Cell,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Public helper for plotting/debugging.
        """

        x_zoom, x_prefix = (
            cell.descriptor("x")
        )

        y_zoom, y_prefix = (
            cell.descriptor("y")
        )

        descriptor = (
            _CellDescriptor(
                x_zoom=x_zoom,
                x_prefix=x_prefix,
                y_zoom=y_zoom,
                y_prefix=y_prefix,
            )
        )

        return (
            self._descriptor_bounds(
                descriptor
            )
        )

    # ========================================================
    # Geometry classification
    # ========================================================

    def _classify(
        self,
        cell: _CellDescriptor,
        polygon: Polygon,
    ) -> int:

        rectangle = (
            self._descriptor_polygon(
                cell
            )
        )

        if rectangle.disjoint(
            polygon
        ):
            return self.OUTSIDE

        if polygon.covers(
            rectangle
        ):
            return self.INSIDE

        return self.PARTIAL

    # ========================================================
    # Splitting
    # ========================================================

    @staticmethod
    def _split_x(
        cell: _CellDescriptor,
    ) -> tuple[
        _CellDescriptor,
        _CellDescriptor,
    ]:

        new_zoom = (
            cell.x_zoom
            + 1
        )

        left_prefix = (
            cell.x_prefix
            << 1
        )

        return (
            _CellDescriptor(
                x_zoom=new_zoom,
                x_prefix=left_prefix,

                y_zoom=cell.y_zoom,
                y_prefix=cell.y_prefix,
            ),
            _CellDescriptor(
                x_zoom=new_zoom,
                x_prefix=(
                    left_prefix + 1
                ),

                y_zoom=cell.y_zoom,
                y_prefix=cell.y_prefix,
            ),
        )

    @staticmethod
    def _split_y(
        cell: _CellDescriptor,
    ) -> tuple[
        _CellDescriptor,
        _CellDescriptor,
    ]:

        new_zoom = (
            cell.y_zoom
            + 1
        )

        first_prefix = (
            cell.y_prefix
            << 1
        )

        return (
            _CellDescriptor(
                x_zoom=cell.x_zoom,
                x_prefix=cell.x_prefix,

                y_zoom=new_zoom,
                y_prefix=first_prefix,
            ),
            _CellDescriptor(
                x_zoom=cell.x_zoom,
                x_prefix=cell.x_prefix,

                y_zoom=new_zoom,
                y_prefix=(
                    first_prefix + 1
                ),
            ),
        )

    def _evaluate_split(
        self,
        children,
        polygon,
    ):
        """
        Evaluate one candidate axis split.

        Prefer the split that leaves the least unresolved
        (PARTIAL) area.
        """

        classified = []

        unresolved_area = 0.0
        partial_count = 0

        for child in children:

            classification = (
                self._classify(
                    child,
                    polygon,
                )
            )

            classified.append(
                (
                    child,
                    classification,
                )
            )

            if (
                classification
                == self.PARTIAL
            ):
                x1, y1, x2, y2 = (
                    self._descriptor_bounds(
                        child
                    )
                )

                unresolved_area += (
                    (x2 - x1)
                    * (y2 - y1)
                )

                partial_count += 1

        return (
            unresolved_area,
            partial_count,
            tuple(classified),
        )

    def _choose_split(
        self,
        cell: _CellDescriptor,
        polygon: Polygon,
    ):
        can_split_x = (
            cell.x_zoom
            < self.target_zoom
        )

        can_split_y = (
            cell.y_zoom
            < self.target_zoom
        )

        if (
            not can_split_x
            and not can_split_y
        ):
            return None

        if (
            can_split_x
            and not can_split_y
        ):
            children = (
                self._split_x(
                    cell
                )
            )

            classified = tuple(
                (
                    child,
                    self._classify(
                        child,
                        polygon,
                    ),
                )
                for child
                in children
            )

            return (
                "x",
                classified,
            )

        if (
            can_split_y
            and not can_split_x
        ):
            children = (
                self._split_y(
                    cell
                )
            )

            classified = tuple(
                (
                    child,
                    self._classify(
                        child,
                        polygon,
                    ),
                )
                for child
                in children
            )

            return (
                "y",
                classified,
            )

        # Both axes can be subdivided.
        x_children = (
            self._split_x(
                cell
            )
        )

        y_children = (
            self._split_y(
                cell
            )
        )

        (
            x_unresolved,
            x_partial_count,
            x_classified,
        ) = self._evaluate_split(
            x_children,
            polygon,
        )

        (
            y_unresolved,
            y_partial_count,
            y_classified,
        ) = self._evaluate_split(
            y_children,
            polygon,
        )

        x_score = (
            x_unresolved,
            x_partial_count,
        )

        y_score = (
            y_unresolved,
            y_partial_count,
        )

        if x_score < y_score:
            return (
                "x",
                x_classified,
            )

        if y_score < x_score:
            return (
                "y",
                y_classified,
            )

        # Tie-break using the longer current dimension.
        x_span = (
            1
            << (
                self.target_zoom
                - cell.x_zoom
            )
        )

        y_span = (
            1
            << (
                self.target_zoom
                - cell.y_zoom
            )
        )

        if x_span >= y_span:
            return (
                "x",
                x_classified,
            )

        return (
            "y",
            y_classified,
        )

    # ========================================================
    # Initial cells
    # ========================================================

    def _initial_cells(
        self,
        polygon: Polygon,
    ):
        min_x, min_y, max_x, max_y = (
            polygon.bounds
        )

        scale = (
            1
            << (
                self.target_zoom
                - self.min_zoom
            )
        )

        index_limit = (
            (1 << self.min_zoom)
            - 1
        )

        x_min_index = max(
            0,
            min(
                index_limit,
                math.floor(
                    min_x
                    / scale
                ),
            ),
        )

        x_max_index = max(
            0,
            min(
                index_limit,
                math.floor(
                    max_x
                    / scale
                ),
            ),
        )

        y_min_index = max(
            0,
            min(
                index_limit,
                math.floor(
                    min_y
                    / scale
                ),
            ),
        )

        y_max_index = max(
            0,
            min(
                index_limit,
                math.floor(
                    max_y
                    / scale
                ),
            ),
        )

        for x_prefix in range(
            x_min_index,
            x_max_index + 1,
        ):
            for y_prefix in range(
                y_min_index,
                y_max_index + 1,
            ):

                yield _CellDescriptor(
                    x_zoom=self.min_zoom,
                    x_prefix=x_prefix,

                    y_zoom=self.min_zoom,
                    y_prefix=y_prefix,
                )

    # ========================================================
    # Cover
    # ========================================================

    def cover(
        self,
        region: OrientedBEVRegion,
    ) -> PolygonHierarchicalCoverResult:

        start = time.perf_counter()

        polygon = (
            self._region_to_tile_polygon(
                region
            )
        )

        output = []

        visited = 0
        discarded = 0

        interior = 0
        boundary = 0

        def visit(
            cell: _CellDescriptor,
            classification=None,
        ):
            nonlocal visited
            nonlocal discarded
            nonlocal interior
            nonlocal boundary

            visited += 1

            if (
                visited
                > self.max_visited_cells
            ):
                raise RuntimeError(
                    "Polygon cover exceeded "
                    f"{self.max_visited_cells:,} "
                    "visited cells"
                )

            if classification is None:

                classification = (
                    self._classify(
                        cell,
                        polygon,
                    )
                )

            # ----------------------------------------------
            # Outside
            # ----------------------------------------------

            if (
                classification
                == self.OUTSIDE
            ):
                discarded += 1
                return

            # ----------------------------------------------
            # Fully inside
            #
            # Keep this coarse cell immediately.
            # ----------------------------------------------

            if (
                classification
                == self.INSIDE
            ):
                output.append(
                    cell
                )

                interior += 1
                return

            # ----------------------------------------------
            # Partial intersection at maximum resolution.
            #
            # Keep conservatively.
            # ----------------------------------------------

            at_max_x = (
                cell.x_zoom
                == self.target_zoom
            )

            at_max_y = (
                cell.y_zoom
                == self.target_zoom
            )

            if (
                at_max_x
                and at_max_y
            ):
                output.append(
                    cell
                )

                boundary += 1
                return

            # ----------------------------------------------
            # Partial cell -> choose one axis to subdivide.
            # ----------------------------------------------

            split = (
                self._choose_split(
                    cell,
                    polygon,
                )
            )

            if split is None:
                # Defensive fallback.
                output.append(
                    cell
                )

                boundary += 1
                return

            _, children = split

            for (
                child,
                child_classification,
            ) in children:

                visit(
                    child,
                    child_classification,
                )

        for initial_cell in (
            self._initial_cells(
                polygon
            )
        ):
            visit(
                initial_cell
            )

        cells = tuple(
            descriptor.to_cell()
            for descriptor
            in output
        )

        # Number of z=target leaves represented by the
        # resulting hierarchical cells.
        leaf_equivalent_count = 0

        for descriptor in output:

            x_leaf_span = (
                1
                << (
                    self.target_zoom
                    - descriptor.x_zoom
                )
            )

            y_leaf_span = (
                1
                << (
                    self.target_zoom
                    - descriptor.y_zoom
                )
            )

            leaf_equivalent_count += (
                x_leaf_span
                * y_leaf_span
            )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            PolygonHierarchicalCoverResult(
                cells=cells,

                target_zoom=(
                    self.target_zoom
                ),

                min_zoom=(
                    self.min_zoom
                ),

                cover_time_ms=(
                    elapsed_ms
                ),

                visited_cell_count=(
                    visited
                ),

                discarded_cell_count=(
                    discarded
                ),

                interior_cell_count=(
                    interior
                ),

                boundary_cell_count=(
                    boundary
                ),

                leaf_equivalent_count=(
                    leaf_equivalent_count
                ),

                tile_polygon=(
                    polygon
                ),
            )
        )