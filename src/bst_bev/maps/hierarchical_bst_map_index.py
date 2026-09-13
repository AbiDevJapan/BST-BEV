from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from bst_id.constants import X_RANGE
from bst_id.encoder_tile_calculator import (
    encode_x_tile_index,
    encode_y_tile_index,
)
from bst_id.region_algebra import Cell

from bst_bev.maps.bst_map_index import (
    BSTMapIndex,
    MapPrimitive,
)


@dataclass(
    frozen=True,
    slots=True,
    order=True,
)
class HierarchicalCellKey:
    """
    Hashable representation of a mixed-resolution
    2D BST cell.

    Example:
        x at zoom 20
        y at zoom 23
    """

    x_zoom: int
    x_prefix: int

    y_zoom: int
    y_prefix: int

    @classmethod
    def from_cell(
        cls,
        cell: Cell,
    ) -> "HierarchicalCellKey":

        if not cell.active("x"):
            raise ValueError(
                "Cell must have active x dimension"
            )

        if not cell.active("y"):
            raise ValueError(
                "Cell must have active y dimension"
            )

        x_zoom, x_prefix = (
            cell.descriptor("x")
        )

        y_zoom, y_prefix = (
            cell.descriptor("y")
        )

        return cls(
            x_zoom=x_zoom,
            x_prefix=x_prefix,
            y_zoom=y_zoom,
            y_prefix=y_prefix,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class HierarchicalQueryResult:

    primitives: frozenset[MapPrimitive]

    cell_count: int
    lookup_time_ms: float

    @property
    def candidate_count(self) -> int:
        return len(self.primitives)

    @property
    def tokens(
        self,
    ) -> set[tuple[str, str]]:

        return {
            (
                primitive.layer,
                primitive.token,
            )
            for primitive in self.primitives
        }


class HierarchicalBSTMapIndex(BSTMapIndex):
    """
    Multi-resolution BST spatial index.

    A map primitive is indexed at all x/y prefix
    combinations between min_zoom and max_zoom.

    For example, with min_zoom=20 and max_zoom=24:

        (20,20)
        (20,21)
        ...
        (20,24)

        (21,20)
        ...
        (24,24)

    are all addressable.

    This allows a mixed-resolution hierarchical query
    to directly look up map primitives without expanding
    its cells back to z24 leaves.
    """

    def __init__(
        self,
        nusc_map,
        location: str,
        *,
        min_zoom: int = 20,
        max_zoom: int = 24,
        map_layers=None,
        index_spacing_m: float = 1.0,
        max_index_samples: int = 5_000_000,
    ):
        if min_zoom < 1:
            raise ValueError(
                "min_zoom must be >= 1"
            )

        if max_zoom > 32:
            raise ValueError(
                "max_zoom must be <= 32"
            )

        if min_zoom > max_zoom:
            raise ValueError(
                "min_zoom cannot exceed max_zoom"
            )

        kwargs = {}

        if map_layers is not None:
            kwargs["map_layers"] = map_layers

        super().__init__(
            nusc_map=nusc_map,
            location=location,
            zoom=max_zoom,
            index_spacing_m=index_spacing_m,
            adaptive_index_spacing=False,
            max_index_samples=max_index_samples,
            **kwargs,
        )

        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

        self.spatial_index: dict[
            HierarchicalCellKey,
            set[MapPrimitive],
        ] = defaultdict(set)

        # Store each actual primitive once.
        # The spatial index only stores references.
        self.primitive_store: dict[
            MapPrimitive,
            dict,
        ] = {}

        self.build_time_s = 0.0

        self.unique_leaf_memberships = 0
        self.membership_count = 0

    # ========================================================
    # Leaf tile conversion
    # ========================================================

    def _xy_to_leaf_tile(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """
        Convert nuScenes XY to the leaf tile indices at
        max_zoom.
        """

        longitude, latitude = (
            self.encoder.xy_to_lonlat(
                x,
                y,
            )
        )

        tile_x = encode_x_tile_index(
            longitude,
            *X_RANGE,
            self.max_zoom,
        )

        tile_y = encode_y_tile_index(
            latitude,
            self.max_zoom,
        )

        return tile_x, tile_y

    # ========================================================
    # Hierarchy
    # ========================================================

    def _prefix(
        self,
        leaf_value: int,
        target_zoom: int,
    ) -> int:
        """
        Convert a max_zoom leaf tile coordinate into
        its prefix at target_zoom.
        """

        if not (
            self.min_zoom
            <= target_zoom
            <= self.max_zoom
        ):
            raise ValueError(
                f"target_zoom must be between "
                f"{self.min_zoom} and "
                f"{self.max_zoom}"
            )

        shift = (
            self.max_zoom
            - target_zoom
        )

        return leaf_value >> shift

    def _hierarchical_keys_for_leaf(
        self,
        leaf_x: int,
        leaf_y: int,
    ):
        """
        Generate every supported mixed-resolution key
        containing this leaf.
        """

        x_prefixes = {
            zoom: self._prefix(
                leaf_x,
                zoom,
            )
            for zoom in range(
                self.min_zoom,
                self.max_zoom + 1,
            )
        }

        y_prefixes = {
            zoom: self._prefix(
                leaf_y,
                zoom,
            )
            for zoom in range(
                self.min_zoom,
                self.max_zoom + 1,
            )
        }

        for x_zoom, x_prefix in (
            x_prefixes.items()
        ):
            for y_zoom, y_prefix in (
                y_prefixes.items()
            ):
                yield HierarchicalCellKey(
                    x_zoom=x_zoom,
                    x_prefix=x_prefix,
                    y_zoom=y_zoom,
                    y_prefix=y_prefix,
                )

    # ========================================================
    # Build
    # ========================================================

    def build(
        self,
    ) -> "HierarchicalBSTMapIndex":

        start = time.perf_counter()

        self.spatial_index.clear()
        self.primitive_store.clear()

        self.index_sample_count = 0
        self.unique_leaf_memberships = 0

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

                # Store actual map record once.
                self.primitive_store[
                    primitive
                ] = record

                # ------------------------------------------
                # Determine all z=max_zoom leaf cells
                # occupied by this primitive.
                #
                # Use a set so many densified points in the
                # same leaf don't cause repeated work.
                # ------------------------------------------

                leaf_tiles = set()

                for x, y in (
                    self._iter_record_points(
                        layer,
                        record,
                    )
                ):
                    self.index_sample_count += 1

                    if (
                        self.index_sample_count
                        > self.max_index_samples
                    ):
                        raise RuntimeError(
                            "Hierarchical index exceeded "
                            f"{self.max_index_samples:,} "
                            "geometry samples."
                        )

                    leaf_tiles.add(
                        self._xy_to_leaf_tile(
                            x,
                            y,
                        )
                    )

                self.unique_leaf_memberships += (
                    len(leaf_tiles)
                )

                # ------------------------------------------
                # Insert primitive into every relevant
                # ancestor combination.
                # ------------------------------------------

                for leaf_x, leaf_y in leaf_tiles:

                    for key in (
                        self._hierarchical_keys_for_leaf(
                            leaf_x,
                            leaf_y,
                        )
                    ):
                        self.spatial_index[
                            key
                        ].add(
                            primitive
                        )

        self.membership_count = sum(
            len(primitives)
            for primitives
            in self.spatial_index.values()
        )

        self.build_time_s = (
            time.perf_counter()
            - start
        )

        self._built = True

        return self

    # ========================================================
    # Query
    # ========================================================

    def _validate_cell(
        self,
        cell: Cell,
    ) -> None:

        key = (
            HierarchicalCellKey.from_cell(
                cell
            )
        )

        if not (
            self.min_zoom
            <= key.x_zoom
            <= self.max_zoom
        ):
            raise ValueError(
                f"x zoom {key.x_zoom} is outside "
                f"index range "
                f"[{self.min_zoom}, "
                f"{self.max_zoom}]"
            )

        if not (
            self.min_zoom
            <= key.y_zoom
            <= self.max_zoom
        ):
            raise ValueError(
                f"y zoom {key.y_zoom} is outside "
                f"index range "
                f"[{self.min_zoom}, "
                f"{self.max_zoom}]"
            )

    def query_cells(
        self,
        cells: Iterable[Cell],
    ) -> HierarchicalQueryResult:
        """
        Query an explicit mixed-resolution BST cover.
        """

        self._ensure_built()

        cells = tuple(cells)

        start = time.perf_counter()

        primitives = set()

        for cell in cells:

            self._validate_cell(
                cell
            )

            key = (
                HierarchicalCellKey.from_cell(
                    cell
                )
            )

            primitives.update(
                self.spatial_index.get(
                    key,
                    set(),
                )
            )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return HierarchicalQueryResult(
            primitives=frozenset(
                primitives
            ),

            cell_count=len(cells),

            lookup_time_ms=elapsed_ms,
        )

    def query_tokens(
        self,
        cells: Iterable[Cell],
    ) -> set[tuple[str, str]]:

        return self.query_cells(
            cells
        ).tokens

    # ========================================================
    # Statistics
    # ========================================================

    @property
    def key_count(self) -> int:
        return len(
            self.spatial_index
        )

    @property
    def primitive_count(self) -> int:
        return len(
            self.primitive_store
        )

    def __len__(self) -> int:
        return self.key_count