from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Polygon


@dataclass(frozen=True, slots=True)
class OrientedBEVRegion:
    """
    Ego-oriented rectangular BEV region.

    nuScenes ego-frame convention:

        +x : forward
        +y : left

    Distances are in metres.
    """

    center_x: float
    center_y: float

    yaw: float

    front: float
    back: float

    left: float
    right: float

    def __post_init__(self):
        for name in (
            "front",
            "back",
            "left",
            "right",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be >= 0"
                )

    # ========================================================
    # Local geometry
    # ========================================================

    @property
    def local_corners(
        self,
    ) -> tuple[
        tuple[float, float],
        ...
    ]:
        """
        Rectangle corners in ego coordinates.

        Returned counter-clockwise:

            front-left
            rear-left
            rear-right
            front-right
        """

        return (
            (
                self.front,
                self.left,
            ),
            (
                -self.back,
                self.left,
            ),
            (
                -self.back,
                -self.right,
            ),
            (
                self.front,
                -self.right,
            ),
        )

    # ========================================================
    # Ego -> global
    # ========================================================

    def local_to_global(
        self,
        local_x: float,
        local_y: float,
    ) -> tuple[float, float]:
        """
        Transform one ego-frame point into nuScenes global XY.
        """

        cos_yaw = math.cos(
            self.yaw
        )

        sin_yaw = math.sin(
            self.yaw
        )

        global_x = (
            self.center_x
            + cos_yaw * local_x
            - sin_yaw * local_y
        )

        global_y = (
            self.center_y
            + sin_yaw * local_x
            + cos_yaw * local_y
        )

        return (
            global_x,
            global_y,
        )

    @property
    def global_corners(
        self,
    ) -> tuple[
        tuple[float, float],
        ...
    ]:
        """
        Rectangle corners transformed into global map XY.
        """

        return tuple(
            self.local_to_global(
                x,
                y,
            )
            for x, y
            in self.local_corners
        )

    @property
    def polygon(
        self,
    ) -> Polygon:
        """
        Shapely polygon in nuScenes global coordinates.
        """

        return Polygon(
            self.global_corners
        )

    @property
    def area(self) -> float:
        return (
            self.front + self.back
        ) * (
            self.left + self.right
        )