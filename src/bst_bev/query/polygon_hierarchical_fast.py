from __future__ import annotations

import math
import time

from dataclasses import dataclass, field

from shapely.geometry import Polygon

from bst_id.constants import (
    X_RANGE,
    MAX_MERCATOR_LAT,
)
from bst_id.region_algebra import Cell

from bst_bev.query.oriented_region import (
    OrientedBEVRegion,
)


@dataclass(
    frozen=True,
    slots=True,
    order=True,
)
class _CellDescriptor:
    x_zoom: int
    x_prefix: int
    y_zoom: int
    y_prefix: int

    def to_cell(self) -> Cell:
        return Cell.from_parts(
            x=(self.x_zoom, self.x_prefix),
            y=(self.y_zoom, self.y_prefix),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class FastPolygonHierarchicalCoverResult:
    """
    Result of the optimized conservative hierarchical cover.
    """

    cells: tuple[Cell, ...]

    target_zoom: int
    min_zoom: int

    cover_time_ms: float

    visited_cell_count: int
    discarded_cell_count: int
    interior_cell_count: int
    boundary_cell_count: int

    leaf_equivalent_count: int

    projected_vertex_count: int
    hull_vertex_count: int

    tile_polygon: Polygon = field(
        repr=False
    )

    @property
    def hierarchical_cell_count(
        self,
    ) -> int:
        return len(self.cells)

    @property
    def compression_ratio(
        self,
    ) -> float:
        if not self.cells:
            return 0.0

        return (
            self.leaf_equivalent_count
            / len(self.cells)
        )

    @property
    def reduction_fraction(
        self,
    ) -> float:
        if self.leaf_equivalent_count == 0:
            return 0.0

        return (
            1.0
            - len(self.cells)
            / self.leaf_equivalent_count
        )

    def zoom_distribution(
        self,
    ) -> dict[tuple[int, int], int]:
        counts: dict[
            tuple[int, int],
            int,
        ] = {}

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


class FastHierarchicalPolygonRegionCover:
    """
    Optimized conservative hierarchical cover for an
    ego-oriented BEV polygon.

    Main differences from the generic Step 4B version:

    1. Shapely is used only once to construct the projected
       polygon / convex hull.
    2. Every recursive cell test is performed with cheap
       arithmetic half-space tests rather than allocating
       a Shapely box and calling disjoint()/covers().
    3. Only one split axis is chosen. We do not speculatively
       build/test both X and Y child sets.
    4. Traversal is iterative instead of recursive.
    5. Cell objects are created only after the cover has been
       finalized.

    The returned cover is conservative with respect to the
    projected sampled polygon: partially intersecting cells
    at target_zoom are retained.
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
        classification_epsilon: float = 1e-10,
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

        if max_visited_cells <= 0:
            raise ValueError(
                "max_visited_cells must be > 0"
            )

        self.encoder = encoder

        self.min_zoom = int(
            min_zoom
        )

        self.target_zoom = int(
            target_zoom
        )

        self.boundary_step_m = float(
            boundary_step_m
        )

        self.max_visited_cells = int(
            max_visited_cells
        )

        self.classification_epsilon = float(
            classification_epsilon
        )

        # Filled once per cover() call.
        self._poly_bounds = None
        self._edges = ()

    # ========================================================
    # Geographic -> target-zoom tile coordinates
    # ========================================================

    @staticmethod
    def _longitude_to_tile_x(
        longitude: float,
        zoom: int,
    ) -> float:

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
            math.tan(
                lat_rad
            )
            + 1.0
            / math.cos(
                lat_rad
            )
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

        return (
            self._longitude_to_tile_x(
                longitude,
                self.target_zoom,
            ),
            self._latitude_to_tile_y(
                latitude,
                self.target_zoom,
            ),
        )

    # ========================================================
    # Project the oriented BEV boundary once
    # ========================================================

    def _sample_boundary(
        self,
        region: OrientedBEVRegion,
    ) -> list[tuple[float, float]]:
        """
        Densify the four global-XY BEV edges before converting
        them to tile coordinates.

        Shapely is NOT used here.
        """

        corners = list(
            region.global_corners
        )

        samples: list[
            tuple[float, float]
        ] = []

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

            for j in range(
                segment_count
            ):
                t = (
                    j
                    / segment_count
                )

                samples.append(
                    (
                        x1
                        + t
                        * (x2 - x1),

                        y1
                        + t
                        * (y2 - y1),
                    )
                )

        return samples

    def _project_region_polygon(
        self,
        region: OrientedBEVRegion,
    ) -> tuple[
        Polygon,
        int,
    ]:
        """
        Construct the projected tile-space polygon once.

        We use the convex hull of the projected sampled
        boundary.  The input region is convex, and the hull
        provides a conservative representation of the sampled
        boundary while also giving us a convex polygon for
        very cheap half-space classification.
        """

        boundary_xy = (
            self._sample_boundary(
                region
            )
        )

        tile_points = [
            self._xy_to_tile(
                x,
                y,
            )
            for x, y
            in boundary_xy
        ]

        polygon = Polygon(
            tile_points
        )

        if not polygon.is_valid:
            polygon = (
                polygon.buffer(0)
            )

        if polygon.is_empty:
            raise RuntimeError(
                "Projected BEV polygon is empty"
            )

        hull = (
            polygon.convex_hull
        )

        if (
            hull.is_empty
            or hull.geom_type
            != "Polygon"
        ):
            raise RuntimeError(
                "Projected BEV convex hull is invalid"
            )

        return (
            hull,
            len(tile_points),
        )

    # ========================================================
    # Convex polygon arithmetic representation
    # ========================================================

    @staticmethod
    def _signed_area2(
        vertices: list[
            tuple[float, float]
        ],
    ) -> float:

        total = 0.0

        count = len(
            vertices
        )

        for i in range(
            count
        ):
            x1, y1 = (
                vertices[i]
            )

            x2, y2 = (
                vertices[
                    (i + 1)
                    % count
                ]
            )

            total += (
                x1 * y2
                - x2 * y1
            )

        return total

    def _prepare_polygon(
        self,
        polygon: Polygon,
    ) -> int:
        """
        Convert the convex polygon into CCW edge half-spaces.

        For a CCW edge from p1 to p2:

            cross(p2-p1, p-p1) >= 0

        means p lies on the interior side of that edge.

        Each cell can therefore be classified using only
        additions/multiplications.
        """

        vertices = list(
            polygon.exterior.coords[
                :-1
            ]
        )

        if len(vertices) < 3:
            raise RuntimeError(
                "Projected polygon has fewer than 3 vertices"
            )

        # Force counter-clockwise orientation.
        if (
            self._signed_area2(
                vertices
            )
            < 0.0
        ):
            vertices.reverse()

        edges = []

        for i in range(
            len(vertices)
        ):
            x1, y1 = (
                vertices[i]
            )

            x2, y2 = (
                vertices[
                    (i + 1)
                    % len(vertices)
                ]
            )

            dx = (
                x2 - x1
            )

            dy = (
                y2 - y1
            )

            if (
                dx == 0.0
                and dy == 0.0
            ):
                continue

            # cross(edge, point-p1) has units of tile^2.
            # Scale epsilon by edge magnitude.
            eps = (
                self.classification_epsilon
                * max(
                    1.0,
                    abs(dx)
                    + abs(dy),
                )
            )

            edges.append(
                (
                    x1,
                    y1,
                    dx,
                    dy,
                    eps,
                )
            )

        self._edges = tuple(
            edges
        )

        self._poly_bounds = (
            polygon.bounds
        )

        return len(
            vertices
        )

    # ========================================================
    # Cell arithmetic
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
            x_min
            + x_scale
        )

        y_min = (
            cell.y_prefix
            * y_scale
        )

        y_max = (
            y_min
            + y_scale
        )

        return (
            float(x_min),
            float(y_min),
            float(x_max),
            float(y_max),
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
        Public plotting/debugging helper compatible with the
        Step 4B/4C notebook.
        """

        x_zoom, x_prefix = (
            cell.descriptor("x")
        )

        y_zoom, y_prefix = (
            cell.descriptor("y")
        )

        return (
            self._descriptor_bounds(
                _CellDescriptor(
                    x_zoom=x_zoom,
                    x_prefix=x_prefix,
                    y_zoom=y_zoom,
                    y_prefix=y_prefix,
                )
            )
        )

    # ========================================================
    # Fast cell / convex-polygon classification
    # ========================================================

    def _classify_bounds(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> int:
        """
        Classify an axis-aligned BST cell against the convex
        polygon without constructing any Shapely geometry.

        Tests:
          1. Polygon/AABB bounding-box separation.
          2. Polygon-edge half-spaces.

        For each polygon edge we evaluate the minimum and
        maximum possible half-space value over the rectangle.

        - max < 0 on any edge => entire cell is outside.
        - min >= 0 on every edge => entire cell is inside.
        - otherwise => partial intersection.
        """

        (
            poly_min_x,
            poly_min_y,
            poly_max_x,
            poly_max_y,
        ) = self._poly_bounds

        eps = (
            self.classification_epsilon
        )

        # Rectangle axes are two of the SAT axes.
        if (
            x_max
            < poly_min_x - eps
            or x_min
            > poly_max_x + eps
            or y_max
            < poly_min_y - eps
            or y_min
            > poly_max_y + eps
        ):
            return self.OUTSIDE

        cx = (
            0.5
            * (
                x_min
                + x_max
            )
        )

        cy = (
            0.5
            * (
                y_min
                + y_max
            )
        )

        hx = (
            0.5
            * (
                x_max
                - x_min
            )
        )

        hy = (
            0.5
            * (
                y_max
                - y_min
            )
        )

        completely_inside = True

        for (
            x1,
            y1,
            dx,
            dy,
            edge_eps,
        ) in self._edges:

            # f(p) = cross(edge, p-p1)
            center_value = (
                dx
                * (
                    cy - y1
                )
                - dy
                * (
                    cx - x1
                )
            )

            # Range of f over the AABB.
            radius = (
                abs(dy)
                * hx
                + abs(dx)
                * hy
            )

            maximum = (
                center_value
                + radius
            )

            # Entire AABB lies outside this polygon edge.
            if maximum < -edge_eps:
                return self.OUTSIDE

            minimum = (
                center_value
                - radius
            )

            if minimum < -edge_eps:
                completely_inside = False

        if completely_inside:
            return self.INSIDE

        return self.PARTIAL

    def _classify(
        self,
        cell: _CellDescriptor,
    ) -> int:

        return self._classify_bounds(
            *self._descriptor_bounds(
                cell
            )
        )

    # ========================================================
    # Cheap deterministic split
    # ========================================================

    def _split(
        self,
        cell: _CellDescriptor,
    ) -> tuple[
        _CellDescriptor,
        _CellDescriptor,
    ]:
        """
        Split only one axis.

        We split the currently larger target-zoom span.  This
        keeps the recursion approximately balanced without
        testing hypothetical X and Y splits.

        At equal span, X is split first. The children then
        have a larger Y span, so the next split naturally
        occurs along Y.
        """

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
            raise RuntimeError(
                "Cannot split target-resolution cell"
            )

        if can_split_x:
            x_span = (
                1
                << (
                    self.target_zoom
                    - cell.x_zoom
                )
            )
        else:
            x_span = -1

        if can_split_y:
            y_span = (
                1
                << (
                    self.target_zoom
                    - cell.y_zoom
                )
            )
        else:
            y_span = -1

        split_x = (
            can_split_x
            and (
                not can_split_y
                or x_span >= y_span
            )
        )

        if split_x:
            new_zoom = (
                cell.x_zoom
                + 1
            )

            first_prefix = (
                cell.x_prefix
                << 1
            )

            return (
                _CellDescriptor(
                    x_zoom=new_zoom,
                    x_prefix=first_prefix,
                    y_zoom=cell.y_zoom,
                    y_prefix=cell.y_prefix,
                ),
                _CellDescriptor(
                    x_zoom=new_zoom,
                    x_prefix=(
                        first_prefix + 1
                    ),
                    y_zoom=cell.y_zoom,
                    y_prefix=cell.y_prefix,
                ),
            )

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

    # ========================================================
    # Initial min-zoom cells
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
    ) -> FastPolygonHierarchicalCoverResult:

        start = (
            time.perf_counter()
        )

        (
            tile_polygon,
            projected_vertex_count,
        ) = self._project_region_polygon(
            region
        )

        hull_vertex_count = (
            self._prepare_polygon(
                tile_polygon
            )
        )

        stack = list(
            self._initial_cells(
                tile_polygon
            )
        )

        output: list[
            _CellDescriptor
        ] = []

        visited = 0
        discarded = 0
        interior = 0
        boundary = 0

        # Iterative DFS avoids Python recursion overhead.
        while stack:

            cell = stack.pop()

            visited += 1

            if (
                visited
                > self.max_visited_cells
            ):
                raise RuntimeError(
                    "Fast polygon cover exceeded "
                    f"{self.max_visited_cells:,} "
                    "visited cells"
                )

            classification = (
                self._classify(
                    cell
                )
            )

            if (
                classification
                == self.OUTSIDE
            ):
                discarded += 1
                continue

            if (
                classification
                == self.INSIDE
            ):
                output.append(
                    cell
                )

                interior += 1
                continue

            at_target = (
                cell.x_zoom
                == self.target_zoom
                and cell.y_zoom
                == self.target_zoom
            )

            if at_target:
                # Conservative boundary retention.
                output.append(
                    cell
                )

                boundary += 1
                continue

            child_a, child_b = (
                self._split(
                    cell
                )
            )

            stack.append(
                child_b
            )

            stack.append(
                child_a
            )

        # Deterministic output ordering helps debugging and CSV
        # comparisons while costing little at this scale.
        output.sort()

        leaf_equivalent_count = 0

        for descriptor in output:

            leaf_equivalent_count += (
                (
                    1
                    << (
                        self.target_zoom
                        - descriptor.x_zoom
                    )
                )
                *
                (
                    1
                    << (
                        self.target_zoom
                        - descriptor.y_zoom
                    )
                )
            )

        cells = tuple(
            descriptor.to_cell()
            for descriptor
            in output
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            FastPolygonHierarchicalCoverResult(
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

                projected_vertex_count=(
                    projected_vertex_count
                ),

                hull_vertex_count=(
                    hull_vertex_count
                ),

                tile_polygon=(
                    tile_polygon
                ),
            )
        )
