"""nuScenes evaluation for the R-tree baseline.

This is intentionally an evaluation-style pytest test:

1. Load nuScenes and the global HD maps.
2. Extract configured vector-map layers.
3. Build one R-tree per nuScenes map.
4. Generate oriented ego-centered BEV queries from nuScenes ego poses.
5. Run:
      a) R-tree bounding-box candidate retrieval.
      b) exact Shapely refinement of those candidates.
      c) brute-force exact intersection as the correctness reference.
6. Assert the R-tree refined result has perfect recall against brute force.
7. Write query-level and summary CSVs.

Environment variables
---------------------
NUSCENES_DATAROOT
    Dataset root. Default: data/nuscenes

NUSCENES_VERSION
    Default: v1.0-mini

RTREE_RESULTS_DIR
    Default: results/rtree

RTREE_LAYERS
    Comma-separated nuScenes map layers.
    Default: lane_divider,road_divider,ped_crossing

RTREE_BEV_WIDTH_M
    Local BEV width in metres. Default: 60

RTREE_BEV_HEIGHT_M
    Local BEV height in metres. Default: 30

RTREE_MAX_QUERIES_PER_MAP
    Maximum ego-pose queries per map. 0 means no limit.
    Default: 100

RTREE_QUERY_STRIDE
    Keep every N-th eligible sample before applying the per-map cap.
    Default: 1

Example
-------
NUSCENES_DATAROOT=/path/to/nuscenes \
NUSCENES_VERSION=v1.0-mini \
pytest -s tests/test_rtree_nuscenes.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
from pyquaternion import Quaternion
from rtree import index as rtree_index
from shapely.affinity import rotate, translate
from shapely.geometry import box
from shapely.ops import unary_union

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from rtree_index import RTreeIndex


DATAROOT = Path(os.getenv("NUSCENES_DATAROOT", "data/nuscenes"))
NUSCENES_VERSION = os.getenv("NUSCENES_VERSION", "v1.0-mini")
RESULTS_DIR = Path(os.getenv("RTREE_RESULTS_DIR", "outputs/rtree"))

LAYERS = tuple(
    layer.strip()
    for layer in os.getenv(
        "RTREE_LAYERS",
        "lane_divider,road_divider,ped_crossing",
    ).split(",")
    if layer.strip()
)

BEV_WIDTH_M = float(os.getenv("RTREE_BEV_WIDTH_M", "60"))
BEV_HEIGHT_M = float(os.getenv("RTREE_BEV_HEIGHT_M", "30"))
MAX_QUERIES_PER_MAP = int(os.getenv("RTREE_MAX_QUERIES_PER_MAP", "100"))
QUERY_STRIDE = max(1, int(os.getenv("RTREE_QUERY_STRIDE", "1")))


@dataclass(frozen=True)
class QuerySpec:
    map_name: str
    scene_token: str
    sample_token: str
    ego_x: float
    ego_y: float
    ego_yaw_rad: float


def _dataset_is_available() -> bool:
    """Check for the minimum nuScenes metadata needed by the configured version."""
    return (
        DATAROOT.exists()
        and (DATAROOT / NUSCENES_VERSION).exists()
        and (DATAROOT / "maps").exists()
    )


def _geometry_from_record(nusc_map: NuScenesMap, record: dict):
    """Extract a Shapely geometry from a nuScenes map record.

    Handles standard line layers, standard polygon layers, and records such as
    drivable_area that contain multiple polygon tokens.
    """
    line_token = record.get("line_token")
    if line_token:
        return nusc_map.extract_line(line_token)

    polygon_token = record.get("polygon_token")
    if polygon_token:
        return nusc_map.extract_polygon(polygon_token)

    polygon_tokens = record.get("polygon_tokens")
    if polygon_tokens:
        polygons = [
            nusc_map.extract_polygon(token)
            for token in polygon_tokens
            if token
        ]
        polygons = [geom for geom in polygons if not geom.is_empty]
        if not polygons:
            return None
        return unary_union(polygons)

    return None


def _extract_map_geometries(
    nusc_map: NuScenesMap,
    map_name: str,
    layers: Iterable[str],
) -> dict[str, object]:
    """Extract configured nuScenes map objects as Shapely geometries."""
    geometries: dict[str, object] = {}

    for layer_name in layers:
        if not hasattr(nusc_map, layer_name):
            raise ValueError(
                f"nuScenes map {map_name!r} has no layer {layer_name!r}. "
                f"Check RTREE_LAYERS."
            )

        records = getattr(nusc_map, layer_name)

        for record in records:
            geometry = _geometry_from_record(nusc_map, record)

            if geometry is None or geometry.is_empty:
                continue

            token = record.get("token")
            if token is None:
                raise KeyError(
                    f"Map record in layer {layer_name!r} has no token."
                )

            object_key = f"{map_name}:{layer_name}:{token}"
            geometries[object_key] = geometry

    return geometries


def _build_index(
    geometries: dict[str, object],
) -> tuple[RTreeIndex, float, int]:
    """Build the in-memory R-tree and return time + Python heap peak.

    ``tracemalloc`` does not include all native libspatialindex allocations.
    For that reason, disk serialization size is measured separately.
    """
    tracemalloc.start()
    start_ns = time.perf_counter_ns()

    spatial_index = RTreeIndex()
    spatial_index.insert_many(
        (object_key, tuple(map(float, geometry.bounds)))
        for object_key, geometry in geometries.items()
    )

    build_time_ms = (time.perf_counter_ns() - start_ns) / 1e6
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return spatial_index, build_time_ms, int(peak_bytes)


def _persistent_index_size_bytes(
    geometries: dict[str, object],
) -> int:
    """Serialize an equivalent R-tree and return .idx + .dat + key-map bytes.

    Rtree itself requires integer IDs. The JSON key map is included because the
    benchmark uses nuScenes string object IDs and needs that mapping to recover
    them from a persisted index.
    """
    with tempfile.TemporaryDirectory(prefix="rtree_nuscenes_") as tmp:
        prefix = Path(tmp) / "index"

        properties = rtree_index.Property()
        properties.dimension = 2

        disk_index = rtree_index.Index(str(prefix), properties=properties)

        id_to_key: dict[int, str] = {}

        for rtree_id, (object_key, geometry) in enumerate(geometries.items()):
            bounds = tuple(map(float, geometry.bounds))
            disk_index.insert(rtree_id, bounds)
            id_to_key[rtree_id] = object_key

        disk_index.close()

        key_map_path = Path(tmp) / "id_to_key.json"
        key_map_path.write_text(
            json.dumps(id_to_key, separators=(",", ":")),
            encoding="utf-8",
        )

        total = 0
        for path in Path(tmp).iterdir():
            if path.is_file():
                total += path.stat().st_size

        return int(total)


def _scene_map_name(nusc: NuScenes, scene: dict) -> str:
    log_record = nusc.get("log", scene["log_token"])
    return log_record["location"]


def _iter_scene_samples(nusc: NuScenes, scene: dict):
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


def _collect_queries(nusc: NuScenes) -> list[QuerySpec]:
    """Collect ego-pose queries, capped independently for each map."""
    counts_by_map: dict[str, int] = {}
    seen_by_map: dict[str, int] = {}
    queries: list[QuerySpec] = []

    for scene in nusc.scene:
        map_name = _scene_map_name(nusc, scene)
        counts_by_map.setdefault(map_name, 0)
        seen_by_map.setdefault(map_name, 0)

        for sample in _iter_scene_samples(nusc, scene):
            seen_idx = seen_by_map[map_name]
            seen_by_map[map_name] += 1

            if seen_idx % QUERY_STRIDE != 0:
                continue

            if (
                MAX_QUERIES_PER_MAP > 0
                and counts_by_map[map_name] >= MAX_QUERIES_PER_MAP
            ):
                continue

            lidar_token = sample["data"].get("LIDAR_TOP")
            if lidar_token is None:
                continue

            sample_data = nusc.get("sample_data", lidar_token)
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

            ego_x = float(ego_pose["translation"][0])
            ego_y = float(ego_pose["translation"][1])

            yaw_rad = float(
                Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
            )

            queries.append(
                QuerySpec(
                    map_name=map_name,
                    scene_token=scene["token"],
                    sample_token=sample["token"],
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_yaw_rad=yaw_rad,
                )
            )
            counts_by_map[map_name] += 1

    return queries


def _make_oriented_bev_query(query: QuerySpec):
    """Create the exact ego-centered oriented rectangular BEV footprint."""
    local_patch = box(
        -BEV_WIDTH_M / 2.0,
        -BEV_HEIGHT_M / 2.0,
        BEV_WIDTH_M / 2.0,
        BEV_HEIGHT_M / 2.0,
    )

    rotated = rotate(
        local_patch,
        query.ego_yaw_rad,
        origin=(0.0, 0.0),
        use_radians=True,
    )

    return translate(
        rotated,
        xoff=query.ego_x,
        yoff=query.ego_y,
    )


def _recall(predicted: set[str], reference: set[str]) -> float:
    if not reference:
        return 1.0 if not predicted else 1.0
    return len(predicted & reference) / len(reference)


def _precision(predicted: set[str], reference: set[str]) -> float:
    if not predicted:
        return 1.0 if not reference else 0.0
    return len(predicted & reference) / len(predicted)


def _percentile(values: pd.Series, q: float) -> float:
    if values.empty:
        return float("nan")
    return float(np.percentile(values.to_numpy(dtype=float), q))


def _summarize_map_queries(
    map_name: str,
    map_df: pd.DataFrame,
    *,
    num_objects: int,
    build_time_ms: float,
    python_peak_bytes: int,
    index_disk_bytes: int,
) -> dict:
    return {
        "map": map_name,
        "layers": ",".join(LAYERS),
        "num_objects": num_objects,
        "num_queries": len(map_df),
        "bev_width_m": BEV_WIDTH_M,
        "bev_height_m": BEV_HEIGHT_M,
        "build_time_ms": build_time_ms,
        "python_tracemalloc_peak_bytes": python_peak_bytes,
        "index_disk_bytes": index_disk_bytes,
        "candidate_query_mean_ms": float(map_df["candidate_query_ms"].mean()),
        "candidate_query_p50_ms": _percentile(map_df["candidate_query_ms"], 50),
        "candidate_query_p95_ms": _percentile(map_df["candidate_query_ms"], 95),
        "candidate_query_p99_ms": _percentile(map_df["candidate_query_ms"], 99),
        "exact_total_mean_ms": float(map_df["rtree_exact_total_ms"].mean()),
        "exact_total_p50_ms": _percentile(map_df["rtree_exact_total_ms"], 50),
        "exact_total_p95_ms": _percentile(map_df["rtree_exact_total_ms"], 95),
        "exact_total_p99_ms": _percentile(map_df["rtree_exact_total_ms"], 99),
        "bruteforce_mean_ms": float(map_df["bruteforce_ms"].mean()),
        "mean_candidates": float(map_df["num_candidates"].mean()),
        "median_candidates": float(map_df["num_candidates"].median()),
        "mean_exact_hits": float(map_df["num_exact_hits"].mean()),
        "mean_candidate_ratio": float(map_df["candidate_ratio"].mean()),
        "mean_candidate_precision": float(map_df["candidate_precision"].mean()),
        "min_exact_recall": float(map_df["exact_recall"].min()),
        "mean_exact_recall": float(map_df["exact_recall"].mean()),
    }


@pytest.mark.integration
@pytest.mark.nuscenes
def test_rtree_nuscenes_evaluation():
    """Benchmark R-tree retrieval on nuScenes and write evaluation CSVs."""
    if not _dataset_is_available():
        pytest.skip(
            "nuScenes dataset not found. Set NUSCENES_DATAROOT to a root "
            f"containing {NUSCENES_VERSION}/ and maps/."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(
        version=NUSCENES_VERSION,
        dataroot=str(DATAROOT),
        verbose=False,
    )

    queries = _collect_queries(nusc)
    assert queries, "No eligible nuScenes ego-pose queries were found."

    required_map_names = sorted({query.map_name for query in queries})

    query_rows: list[dict] = []
    map_build_stats: dict[str, dict] = {}

    for map_name in required_map_names:
        nusc_map = NuScenesMap(
            dataroot=str(DATAROOT),
            map_name=map_name,
        )

        geometries = _extract_map_geometries(
            nusc_map=nusc_map,
            map_name=map_name,
            layers=LAYERS,
        )
        assert geometries, (
            f"No map geometries extracted for {map_name}. "
            "Check RTREE_LAYERS and map files."
        )

        spatial_index, build_time_ms, python_peak_bytes = _build_index(
            geometries
        )
        assert len(spatial_index) == len(geometries)

        index_disk_bytes = _persistent_index_size_bytes(geometries)

        map_build_stats[map_name] = {
            "num_objects": len(geometries),
            "build_time_ms": build_time_ms,
            "python_peak_bytes": python_peak_bytes,
            "index_disk_bytes": index_disk_bytes,
        }

        map_queries = [q for q in queries if q.map_name == map_name]

        try:
            for query_index, query in enumerate(map_queries):
                query_geometry = _make_oriented_bev_query(query)
                query_bounds = tuple(map(float, query_geometry.bounds))

                # Stage 1: R-tree bounding-box candidate retrieval.
                start_ns = time.perf_counter_ns()
                candidate_keys = spatial_index.query(query_bounds)
                candidate_query_ms = (
                    time.perf_counter_ns() - start_ns
                ) / 1e6

                candidate_set = set(candidate_keys)

                # Stage 2: exact geometry refinement.
                start_refine_ns = time.perf_counter_ns()
                refined_set = {
                    key
                    for key in candidate_keys
                    if geometries[key].intersects(query_geometry)
                }
                refine_only_ms = (
                    time.perf_counter_ns() - start_refine_ns
                ) / 1e6

                rtree_exact_total_ms = (
                    candidate_query_ms + refine_only_ms
                )

                # Reference: exact brute-force geometry intersection.
                start_bruteforce_ns = time.perf_counter_ns()
                brute_force_set = {
                    key
                    for key, geometry in geometries.items()
                    if geometry.intersects(query_geometry)
                }
                bruteforce_ms = (
                    time.perf_counter_ns() - start_bruteforce_ns
                ) / 1e6

                exact_recall = _recall(refined_set, brute_force_set)
                exact_precision = _precision(
                    refined_set,
                    brute_force_set,
                )
                candidate_precision = _precision(
                    candidate_set,
                    brute_force_set,
                )

                # Correctness is part of the test, not just a reported metric.
                assert refined_set == brute_force_set, (
                    f"R-tree mismatch for map={map_name}, "
                    f"sample={query.sample_token}: "
                    f"missing={len(brute_force_set - refined_set)}, "
                    f"extra={len(refined_set - brute_force_set)}"
                )

                num_objects = len(geometries)
                num_candidates = len(candidate_set)

                query_rows.append(
                    {
                        "map": map_name,
                        "query_index": query_index,
                        "scene_token": query.scene_token,
                        "sample_token": query.sample_token,
                        "ego_x": query.ego_x,
                        "ego_y": query.ego_y,
                        "ego_yaw_rad": query.ego_yaw_rad,
                        "bev_width_m": BEV_WIDTH_M,
                        "bev_height_m": BEV_HEIGHT_M,
                        "num_objects": num_objects,
                        "num_candidates": num_candidates,
                        "num_exact_hits": len(refined_set),
                        "candidate_ratio": (
                            num_candidates / num_objects
                            if num_objects
                            else 0.0
                        ),
                        "candidate_precision": candidate_precision,
                        "exact_precision": exact_precision,
                        "exact_recall": exact_recall,
                        "candidate_query_ms": candidate_query_ms,
                        "refine_only_ms": refine_only_ms,
                        "rtree_exact_total_ms": rtree_exact_total_ms,
                        "bruteforce_ms": bruteforce_ms,
                        "speedup_vs_bruteforce": (
                            bruteforce_ms / rtree_exact_total_ms
                            if rtree_exact_total_ms > 0
                            else math.inf
                        ),
                    }
                )
        finally:
            spatial_index.close()

    query_df = pd.DataFrame(query_rows)
    assert not query_df.empty

    # Every exact R-tree query must agree with brute-force geometry evaluation.
    assert np.allclose(query_df["exact_recall"].to_numpy(), 1.0)
    assert np.allclose(query_df["exact_precision"].to_numpy(), 1.0)

    map_summary_rows: list[dict] = []
    for map_name in required_map_names:
        map_df = query_df[query_df["map"] == map_name]
        stats = map_build_stats[map_name]

        map_summary_rows.append(
            _summarize_map_queries(
                map_name,
                map_df,
                num_objects=stats["num_objects"],
                build_time_ms=stats["build_time_ms"],
                python_peak_bytes=stats["python_peak_bytes"],
                index_disk_bytes=stats["index_disk_bytes"],
            )
        )

    map_summary_df = pd.DataFrame(map_summary_rows)

    overall_summary_df = pd.DataFrame(
        [
            {
                "version": NUSCENES_VERSION,
                "layers": ",".join(LAYERS),
                "num_maps": int(query_df["map"].nunique()),
                "num_queries": len(query_df),
                "bev_width_m": BEV_WIDTH_M,
                "bev_height_m": BEV_HEIGHT_M,
                "total_indexed_objects_across_maps": int(
                    map_summary_df["num_objects"].sum()
                ),
                "total_build_time_ms": float(
                    map_summary_df["build_time_ms"].sum()
                ),
                "total_index_disk_bytes": int(
                    map_summary_df["index_disk_bytes"].sum()
                ),
                "candidate_query_mean_ms": float(
                    query_df["candidate_query_ms"].mean()
                ),
                "candidate_query_p50_ms": _percentile(
                    query_df["candidate_query_ms"], 50
                ),
                "candidate_query_p95_ms": _percentile(
                    query_df["candidate_query_ms"], 95
                ),
                "candidate_query_p99_ms": _percentile(
                    query_df["candidate_query_ms"], 99
                ),
                "exact_total_mean_ms": float(
                    query_df["rtree_exact_total_ms"].mean()
                ),
                "exact_total_p50_ms": _percentile(
                    query_df["rtree_exact_total_ms"], 50
                ),
                "exact_total_p95_ms": _percentile(
                    query_df["rtree_exact_total_ms"], 95
                ),
                "exact_total_p99_ms": _percentile(
                    query_df["rtree_exact_total_ms"], 99
                ),
                "bruteforce_mean_ms": float(
                    query_df["bruteforce_ms"].mean()
                ),
                "mean_candidates": float(
                    query_df["num_candidates"].mean()
                ),
                "mean_candidate_ratio": float(
                    query_df["candidate_ratio"].mean()
                ),
                "mean_candidate_precision": float(
                    query_df["candidate_precision"].mean()
                ),
                "min_exact_recall": float(
                    query_df["exact_recall"].min()
                ),
                "mean_exact_recall": float(
                    query_df["exact_recall"].mean()
                ),
                "mean_speedup_vs_bruteforce": float(
                    query_df["speedup_vs_bruteforce"].mean()
                ),
            }
        ]
    )

    query_csv = RESULTS_DIR / "rtree_query_results.csv"
    map_summary_csv = RESULTS_DIR / "rtree_map_summary.csv"
    overall_summary_csv = RESULTS_DIR / "rtree_overall_summary.csv"

    query_df.to_csv(query_csv, index=False)
    map_summary_df.to_csv(map_summary_csv, index=False)
    overall_summary_df.to_csv(overall_summary_csv, index=False)

    assert query_csv.exists()
    assert map_summary_csv.exists()
    assert overall_summary_csv.exists()

    print("\nR-tree nuScenes evaluation complete")
    print(f"  query results:   {query_csv}")
    print(f"  map summary:     {map_summary_csv}")
    print(f"  overall summary: {overall_summary_csv}")
    print("\nOverall summary:")
    print(overall_summary_df.to_string(index=False))
