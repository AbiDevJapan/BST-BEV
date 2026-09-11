from __future__ import annotations

from dataclasses import dataclass

from bst_id.constants import X_RANGE
from bst_id.encoder_tile_calculator import (
    encode_x_tile_index,
    encode_y_tile_index,
)
from bst_id.region_algebra import Cell

from bst_bev.maps.bst_map_index import BSTSpatialEncoder


CellKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class FixedZoomCoverResult:
    cells: frozenset[Cell]

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def keys(self) -> set[CellKey]:
        """
        Convert region-algebra Cells into the same key format
        used by BSTMapIndex:

            (bst_id, bit_length)
        """

        return {
            (
                cell.to_id(),
                self._bit_length(cell),
            )
            for cell in self.cells
        }

    @staticmethod
    def _bit_length(cell: Cell) -> int:
        """
        Serialized BST-ID length.

        4 presence bits
        + 5 bits per active-axis zoom header
        + payload bits for each active axis.
        """

        active_axes = [
            axis
            for axis in ("x", "y", "f", "t")
            if cell.active(axis)
        ]

        return (
            4
            + 5 * len(active_axes)
            + sum(
                cell.descriptor(axis)[0]
                for axis in active_axes
            )
        )


class FixedZoomRegionCover:
    """
    Directly enumerate all BST XY cells intersecting a
    square nuScenes patch at one fixed zoom.

    No spatial point sampling is performed.
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

        self.encoder = BSTSpatialEncoder(
            location=location,
            zoom=zoom,
        )

    def _xy_to_tile(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """
        Convert nuScenes global XY directly to BST tile
        indices at this zoom.
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
            self.zoom,
        )

        tile_y = encode_y_tile_index(
            latitude,
            self.zoom,
        )

        return tile_x, tile_y

    def cover(
        self,
        center_x: float,
        center_y: float,
        *,
        radius: float = 30.0,
    ) -> FixedZoomCoverResult:
        """
        Return all fixed-zoom BST cells intersecting the
        square patch centered at (center_x, center_y).

        The four corners are converted into BST tile space,
        then the entire integer tile rectangle is enumerated.
        """

        if radius <= 0:
            raise ValueError(
                "radius must be > 0"
            )

        corners = (
            (
                center_x - radius,
                center_y - radius,
            ),
            (
                center_x - radius,
                center_y + radius,
            ),
            (
                center_x + radius,
                center_y - radius,
            ),
            (
                center_x + radius,
                center_y + radius,
            ),
        )

        tile_corners = [
            self._xy_to_tile(x, y)
            for x, y in corners
        ]

        x_indices = [
            tile_x
            for tile_x, _ in tile_corners
        ]

        y_indices = [
            tile_y
            for _, tile_y in tile_corners
        ]

        x_min = min(x_indices)
        x_max = max(x_indices)

        y_min = min(y_indices)
        y_max = max(y_indices)

        cells = set()

        for tile_x in range(
            x_min,
            x_max + 1,
        ):
            for tile_y in range(
                y_min,
                y_max + 1,
            ):
                cells.add(
                    Cell.from_parts(
                        x=(
                            self.zoom,
                            tile_x,
                        ),
                        y=(
                            self.zoom,
                            tile_y,
                        ),
                    )
                )

        return FixedZoomCoverResult(
            cells=frozenset(cells)
        )