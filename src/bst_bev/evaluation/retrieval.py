from dataclasses import dataclass, asdict
from operator import index

from bst_bev.maps.bst_map_index import BSTMapIndex, PatchQueryResult
from bst_bev.nuscenes.context import NuScenesSampleContext


DEFAULT_MAP_LAYERS = (
    "lane_divider",
    "road_divider",
    "ped_crossing",
)


@dataclass(frozen=True)
class RetrievalMetrics:
    zoom: int

    bst_retrieved: int
    ground_truth: int

    true_positive: int
    false_positive: int
    false_negative: int

    precision: float
    recall: float

    cell_width_m: float
    query_step_m: float

    cells_queried: int
    query_samples: int

class BSTRetrievalEvaluator:
	"""
	Evaluate BST-based HD-map retrieval against
	nuScenes geometric patch queries.
	"""

	def __init__(
		self,
		context: NuScenesSampleContext,
		radius: float = 30.0,
		query_step: float = 2.0,
		map_layers=DEFAULT_MAP_LAYERS,
	):
		self.context = context

		self.radius = radius
		self.query_step = query_step
		self.map_layers = tuple(map_layers)

		# Ground truth does not depend on BST zoom,
		# so calculate it only once.
		self._ground_truth_tokens = (
			self._build_ground_truth_tokens()
		)

	@property
	def ground_truth_tokens(self) -> set:
		return self._ground_truth_tokens

	def _build_ground_truth_tokens(self) -> set:
		"""
		Retrieve all map primitives intersecting the
		local ground-truth patch.
		"""

		x = self.context.ego_x
		y = self.context.ego_y

		patch = (
			x - self.radius,
			y - self.radius,
			x + self.radius,
			y + self.radius,
		)

		records = (
			self.context.nusc_map.get_records_in_patch(
				patch,
				list(self.map_layers),
				mode="intersect",
			)
		)

		return {
			(layer, token)
			for layer, tokens in records.items()
			for token in tokens
		}

	def _build_index(
		self,
		zoom: int,
	) -> BSTMapIndex:

		index = BSTMapIndex(
			nusc_map=self.context.nusc_map,
			location=self.context.location,
			zoom=zoom,
		)

		index.build()

		return index

	def _query_index(
		self,
		index: BSTMapIndex,
	) -> tuple[
		set[tuple[str, str]],
		PatchQueryResult,
	]:

		query = index.query_patch(
			center_x=self.context.ego_x,
			center_y=self.context.ego_y,
			radius=self.radius,
		)

		return (
			query.tokens,
			query,
		)

	def evaluate(
		self,
		zoom: int,
	) -> RetrievalMetrics:

		query = index.query_patch(
			center_x=self.context.ego_x,
			center_y=self.context.ego_y,
			radius=self.radius,
		)

		bst_tokens = query.tokens
		gt_tokens = self.ground_truth_tokens

		true_positive = len(
			bst_tokens & gt_tokens
		)

		false_positive = len(
			bst_tokens - gt_tokens
		)

		false_negative = len(
			gt_tokens - bst_tokens
		)

		precision = (
			true_positive
			/ (true_positive + false_positive)
			if true_positive + false_positive
			else 0.0
		)

		recall = (
			true_positive
			/ (true_positive + false_negative)
			if true_positive + false_negative
			else 0.0
		)

		return RetrievalMetrics(
			zoom=index.zoom,

			bst_retrieved=len(bst_tokens),
			ground_truth=len(gt_tokens),

			true_positive=true_positive,
			false_positive=false_positive,
			false_negative=false_negative,

			precision=precision,
			recall=recall,

			cell_width_m=index.cell_width_at(
				self.context.ego_x,
				self.context.ego_y,
			),

			query_step_m=query.actual_step_m,

			cells_queried=query.cell_count,
			query_samples=query.sample_count,
		)

	def sweep(
		self,
		zooms,
	) -> list[RetrievalMetrics]:

		return [
			self.evaluate(zoom)
			for zoom in zooms
		]
	