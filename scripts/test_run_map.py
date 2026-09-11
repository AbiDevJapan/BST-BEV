from pathlib import Path

from bst_bev.nuscenes.context import NuScenesSampleContext
from bst_bev.evaluation.retrieval import BSTRetrievalEvaluator


PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATAROOT = PROJECT_ROOT / "data" / "nuscenes"

context = NuScenesSampleContext.from_scene_index(
    dataroot=DATAROOT,
    scene_index=0,
)

print("Scene:", context.scene["name"])
print("Location:", context.location)
print(
    "Ego:",
    context.ego_x,
    context.ego_y,
)

evaluator = BSTRetrievalEvaluator(
    context=context,
    radius=30.0,
    query_step=2.0,
)

results = evaluator.sweep(
    range(18, 40)
)

for result in results:
    print(
        f"z={result.zoom:2d} | "
        f"recall={result.recall:.3f} | "
        f"precision={result.precision:.3f} | "
        f"TP={result.true_positive:2d} | "
        f"FP={result.false_positive:2d} | "
        f"FN={result.false_negative:2d}"
    )

import pandas as pd


df = pd.DataFrame(
    result.to_dict()
    for result in results
)

print(df)