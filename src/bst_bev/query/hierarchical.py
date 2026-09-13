from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from bst_id.region_algebra import (
    Cell,
    normalize,
    semantic_equal,
)


@dataclass(frozen=True, slots=True)
class HierarchicalCoverResult:
    """
    A normalized hierarchical representation of a
    fixed-resolution BST region.
    """

    cells: tuple[Cell, ...]

    leaf_cell_count: int
    hierarchical_cell_count: int

    normalization_time_ms: float

    @property
    def compression_ratio(self) -> float:
        """
        How many fixed-resolution cells are represented
        by each normalized hierarchical cell.
        """

        if self.hierarchical_cell_count == 0:
            return 0.0

        return (
            self.leaf_cell_count
            / self.hierarchical_cell_count
        )

    @property
    def reduction_fraction(self) -> float:
        """
        Fraction of region descriptors eliminated.

        Example:
            0.90 = 90% fewer cells.
        """

        if self.leaf_cell_count == 0:
            return 0.0

        return 1.0 - (
            self.hierarchical_cell_count
            / self.leaf_cell_count
        )

    def zoom_distribution(
        self,
    ) -> dict[tuple[int, int], int]:
        """
        Count hierarchical cells by (x_zoom, y_zoom).

        Note:
        BST normalization may merge axes independently,
        so cells do not necessarily have equal x/y zoom.
        """

        counts = Counter()

        for cell in self.cells:
            x_zoom, _ = cell.descriptor("x")
            y_zoom, _ = cell.descriptor("y")

            counts[
                (x_zoom, y_zoom)
            ] += 1

        return dict(
            sorted(counts.items())
        )


class HierarchicalRegionNormalizer:
    """
    Compress an exact fixed-zoom BST region using the
    BST-ID region algebra.
    """

    def normalize(
        self,
        leaf_cells,
    ) -> HierarchicalCoverResult:

        leaf_cells = tuple(leaf_cells)

        start = time.perf_counter()

        hierarchical_cells = normalize(
            leaf_cells
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return HierarchicalCoverResult(
            cells=hierarchical_cells,

            leaf_cell_count=len(
                leaf_cells
            ),

            hierarchical_cell_count=len(
                hierarchical_cells
            ),

            normalization_time_ms=(
                elapsed_ms
            ),
        )

    @staticmethod
    def verify_equivalent(
        leaf_cells,
        hierarchical_cells,
        *,
        working_zoom: int,
    ) -> bool:
        """
        Verify that the hierarchical representation covers
        exactly the same working-grid cells as the original
        fixed-resolution representation.
        """

        return semantic_equal(
            leaf_cells,
            hierarchical_cells,
            working_zoom={
                "x": working_zoom,
                "y": working_zoom,
            },
        )