"""Reusable 2-D R-tree spatial index.

The index stores axis-aligned bounding boxes only. Geometry ownership stays with
the caller, which keeps the index implementation independent from Shapely and
avoids duplicating map geometry inside the index.

Bounding-box order follows Rtree's default interleaved convention:

    (min_x, min_y, max_x, max_y)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Hashable

from rtree import index as rtree_index


Bounds2D = tuple[float, float, float, float]


class RTreeIndex:
    """Small wrapper around :mod:`rtree` for string/hashable object IDs."""

    def __init__(self) -> None:
        properties = rtree_index.Property()
        properties.dimension = 2

        self._index = rtree_index.Index(properties=properties)
        self._rtree_id_to_key: dict[int, Hashable] = {}
        self._key_to_rtree_id: dict[Hashable, int] = {}
        self._next_rtree_id = 0

    @staticmethod
    def _validate_bounds(bounds: Bounds2D) -> Bounds2D:
        if len(bounds) != 4:
            raise ValueError(
                "2-D bounds must be (min_x, min_y, max_x, max_y)."
            )

        min_x, min_y, max_x, max_y = map(float, bounds)

        if min_x > max_x or min_y > max_y:
            raise ValueError(
                f"Invalid bounds {bounds}: minimum coordinate exceeds maximum."
            )

        return min_x, min_y, max_x, max_y

    def insert(self, key: Hashable, bounds: Bounds2D) -> int:
        """Insert one object bounding box.

        Parameters
        ----------
        key:
            Caller-facing object identifier, e.g. a nuScenes map token.
        bounds:
            ``(min_x, min_y, max_x, max_y)``.

        Returns
        -------
        int
            Internal integer ID used by libspatialindex.
        """
        if key in self._key_to_rtree_id:
            raise KeyError(f"Duplicate R-tree key: {key!r}")

        normalized_bounds = self._validate_bounds(bounds)

        rtree_id = self._next_rtree_id
        self._next_rtree_id += 1

        self._index.insert(rtree_id, normalized_bounds)
        self._rtree_id_to_key[rtree_id] = key
        self._key_to_rtree_id[key] = rtree_id

        return rtree_id

    def insert_many(
        self,
        items: Iterable[tuple[Hashable, Bounds2D]],
    ) -> None:
        """Insert ``(key, bounds)`` pairs."""
        for key, bounds in items:
            self.insert(key, bounds)

    def query(self, bounds: Bounds2D) -> list[Hashable]:
        """Return keys whose indexed bounding boxes intersect ``bounds``."""
        normalized_bounds = self._validate_bounds(bounds)
        ids = self._index.intersection(normalized_bounds)
        return [self._rtree_id_to_key[rtree_id] for rtree_id in ids]

    def query_ids(self, bounds: Bounds2D) -> Iterator[int]:
        """Return internal R-tree IDs intersecting ``bounds``."""
        normalized_bounds = self._validate_bounds(bounds)
        yield from self._index.intersection(normalized_bounds)

    def key_for_id(self, rtree_id: int) -> Hashable:
        """Resolve an internal R-tree integer ID to the caller-facing key."""
        return self._rtree_id_to_key[rtree_id]

    def rtree_id_for_key(self, key: Hashable) -> int:
        """Resolve a caller-facing key to the internal R-tree integer ID."""
        return self._key_to_rtree_id[key]

    def __contains__(self, key: object) -> bool:
        return key in self._key_to_rtree_id

    def __len__(self) -> int:
        return len(self._key_to_rtree_id)

    def close(self) -> None:
        """Close the underlying R-tree index."""
        self._index.close()
