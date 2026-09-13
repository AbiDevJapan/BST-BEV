from __future__ import annotations

import math
import time
from dataclasses import dataclass

from bst_id.region_algebra import Cell

from bst_bev.query.fixed_zoom import (
    FixedZoomRegionCover,
    TileBounds,
)


@dataclass(frozen=True, slots=True)
class DyadicBlock:
    """
    One aligned power-of-two interval at the target
    fixed zoom.

    Example at target zoom 24:

        leaf tiles: 100 .. 103
        size:       4
        shift:      2

        prefix zoom  = 24 - 2 = 22
        prefix value = 100 >> 2
    """

    start: int
    end: int

    prefix_zoom: int
    prefix_value: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class DirectHierarchicalCoverResult:
    """
    Compact mixed-resolution BST representation of a
    fixed-zoom rectangular region.
    """

    cells: tuple[Cell, ...]

    x_blocks: tuple[DyadicBlock, ...]
    y_blocks: tuple[DyadicBlock, ...]

    fixed_cell_count: int
    cover_time_ms: float

    @property
    def hierarchical_cell_count(self) -> int:
        return len(self.cells)

    @property
    def compression_ratio(self) -> float:
        if not self.cells:
            return 0.0

        return (
            self.fixed_cell_count
            / self.hierarchical_cell_count
        )

    @property
    def reduction_fraction(self) -> float:
        if self.fixed_cell_count == 0:
            return 0.0

        return 1.0 - (
            self.hierarchical_cell_count
            / self.fixed_cell_count
        )


class DirectHierarchicalRegionCover:
    """
    Generate a hierarchical BST cover directly from the
    rectangular fixed-zoom tile bounds.

    No leaf-cell enumeration.
    No generic normalize().
    """

    def __init__(
        self,
        location: str,
        zoom: int,
    ):
        if not 1 <= zoom <= 32:
            raise ValueError(
                "zoom must be in [1, 32]"
            )

        self.location = location
        self.zoom = zoom

        self.fixed_cover = (
            FixedZoomRegionCover(
                location=location,
                zoom=zoom,
            )
        )

    @staticmethod
    def _largest_aligned_block(
        start: int,
        end: int,
    ) -> int:
        """
        Return the largest power-of-two block that:

        1. begins at `start`,
        2. is correctly binary aligned,
        3. does not extend past `end`.
        """

        remaining = end - start + 1

        # Largest power of two that fits in the remaining
        # interval.
        max_by_length = (
            1 << (remaining.bit_length() - 1)
        )

        if start == 0:
            # Zero is aligned to every power-of-two boundary.
            return max_by_length

        # Largest power of two dividing start.
        max_by_alignment = (
            start & -start
        )

        return min(
            max_by_alignment,
            max_by_length,
        )

    @classmethod
    def decompose_interval(
        cls,
        start: int,
        end: int,
        *,
        target_zoom: int,
    ) -> tuple[DyadicBlock, ...]:
        """
        Decompose an inclusive fixed-zoom integer interval
        into the minimum sequence of aligned power-of-two
        BST prefix blocks.
        """

        if start < 0:
            raise ValueError(
                "start must be >= 0"
            )

        if end < start:
            raise ValueError(
                "end must be >= start"
            )

        maximum_index = (
            (1 << target_zoom) - 1
        )

        if end > maximum_index:
            raise ValueError(
                "interval exceeds target zoom"
            )

        blocks = []

        current = start

        while current <= end:

            size = cls._largest_aligned_block(
                current,
                end,
            )

            shift = size.bit_length() - 1

            prefix_zoom = (
                target_zoom - shift
            )

            prefix_value = (
                current >> shift
            )

            # Cell.from_parts currently requires active
            # axis zooms in [1, 32].
            if prefix_zoom < 1:
                raise ValueError(
                    "Interval collapsed to zoom 0. "
                    "This implementation assumes a local "
                    "region smaller than the entire axis."
                )

            blocks.append(
                DyadicBlock(
                    start=current,
                    end=current + size - 1,
                    prefix_zoom=prefix_zoom,
                    prefix_value=prefix_value,
                )
            )

            current += size

        return tuple(blocks)

    def cover(
        self,
        center_x: float,
        center_y: float,
        *,
        radius: float = 30.0,
    ) -> DirectHierarchicalCoverResult:

        start_time = time.perf_counter()

        bounds = self.fixed_cover.tile_bounds(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )

        x_blocks = self.decompose_interval(
            bounds.x_min,
            bounds.x_max,
            target_zoom=self.zoom,
        )

        y_blocks = self.decompose_interval(
            bounds.y_min,
            bounds.y_max,
            target_zoom=self.zoom,
        )

        # The rectangle is exactly the Cartesian product
        # of the hierarchical X and Y interval covers.
        cells = tuple(
            Cell.from_parts(
                x=(
                    x_block.prefix_zoom,
                    x_block.prefix_value,
                ),
                y=(
                    y_block.prefix_zoom,
                    y_block.prefix_value,
                ),
            )
            for x_block in x_blocks
            for y_block in y_blocks
        )

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return DirectHierarchicalCoverResult(
            cells=cells,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            fixed_cell_count=(
                bounds.cell_count
            ),
            cover_time_ms=elapsed_ms,
        )