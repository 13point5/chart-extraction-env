"""
Point-only OKS reward for chart series extraction.

This keeps the scale-aware OKS idea from the LineEX keypoint metric, but removes
the relaxed "near the line segment" fallback:

1. Match series by name.
2. Normalize chart coordinates using the full gold chart x/y span.
3. For each predicted point in a matched series, find the nearest gold point.
4. Count that prediction as a match only when its OKS score against the nearest
   gold point exceeds the threshold.
5. Return matched-gold recall within each series, then take a weighted average
   across series using the number of gold points.

This rewards slight point-location error without giving credit for points that
only land somewhere along the curve between labeled gold points.

Config notes:

- `oks_k` controls how quickly OKS falls off as predicted points move away from
  gold points in normalized chart coordinates. Larger `oks_k` means distance is
  penalized more gently, so the reward is more forgiving. Smaller `oks_k` makes
  the reward stricter.
- `oks_threshold` is the minimum OKS score a predicted point must exceed to
  count as matching a gold point. Lower `oks_threshold` is more forgiving.
  Higher `oks_threshold` is stricter.

Practical intuition:

- Increasing `oks_k` widens the tolerance window.
- Decreasing `oks_threshold` lowers the bar for a point to count once its OKS is
  computed.
- The default config (`oks_k=0.025`, `oks_threshold=0.5`) is fairly strict.
  A config like (`oks_k=0.05`, `oks_threshold=0.35`) is noticeably more
  forgiving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from schemas import CanonicalPoint, parse_chart_extraction
from ..state import RubricState


DEFAULT_OKS_K = 0.025
DEFAULT_OKS_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class SeriesPointValueConfig:
    oks_k: float = DEFAULT_OKS_K
    oks_threshold: float = DEFAULT_OKS_THRESHOLD

    def __post_init__(self) -> None:
        if self.oks_k <= 0:
            raise ValueError("series_point_value_oks_k must be greater than 0")
        if not 0.0 <= self.oks_threshold <= 1.0:
            raise ValueError("series_point_value_oks_threshold must be between 0 and 1")


DEFAULT_SERIES_POINT_VALUE_CONFIG = SeriesPointValueConfig()


def _point_pairs(points: list[CanonicalPoint]) -> list[tuple[float, float]]:
    return [(float(point.x), float(point.y)) for point in points]


def _normalize_point(
    point: tuple[float, float],
    x_min: float,
    x_scale: float,
    y_min: float,
    y_scale: float,
) -> tuple[float, float]:
    return (
        (point[0] - x_min) / x_scale,
        (point[1] - y_min) / y_scale,
    )


def _oks(distance: float, s: float = 1.0, k: float = DEFAULT_OKS_K) -> float:
    return math.exp(-(distance**2) / (2.0 * (s**2) * (k**2)))


def _nearest_gold_index(
    predicted_point: tuple[float, float],
    gold_points: list[tuple[float, float]],
) -> tuple[float, int]:
    best_distance = math.inf
    best_index = -1

    for index, gold_point in enumerate(gold_points):
        distance = math.dist(predicted_point, gold_point)
        if distance < best_distance:
            best_distance = distance
            best_index = index

    return best_distance, best_index


def series_point_value_score(
    predicted_points: list[CanonicalPoint],
    gold_points: list[CanonicalPoint],
    x_min: float,
    x_scale: float,
    y_min: float,
    y_scale: float,
    config: SeriesPointValueConfig = DEFAULT_SERIES_POINT_VALUE_CONFIG,
) -> float:
    if not predicted_points and not gold_points:
        return 1.0
    if not gold_points:
        return 1.0 if not predicted_points else 0.0
    if not predicted_points:
        return 0.0

    gold_pairs = _point_pairs(gold_points)
    predicted_pairs = _point_pairs(predicted_points)

    if not gold_pairs:
        return 1.0 if not predicted_pairs else 0.0
    if not predicted_pairs:
        return 0.0

    normalized_gold = [
        _normalize_point(point, x_min=x_min, x_scale=x_scale, y_min=y_min, y_scale=y_scale)
        for point in gold_pairs
    ]
    normalized_predicted = [
        _normalize_point(point, x_min=x_min, x_scale=x_scale, y_min=y_min, y_scale=y_scale)
        for point in predicted_pairs
    ]

    found_gold_indices: set[int] = set()

    for predicted_point in normalized_predicted:
        min_distance, gold_index = _nearest_gold_index(predicted_point, normalized_gold)
        if gold_index < 0:
            continue

        if _oks(min_distance, k=config.oks_k) > config.oks_threshold:
            found_gold_indices.add(gold_index)

    return len(found_gold_indices) / len(gold_pairs)


def series_point_value_chart_score(
    predicted_series: dict[str, list[CanonicalPoint]],
    gold_series: dict[str, list[CanonicalPoint]],
    config: SeriesPointValueConfig = DEFAULT_SERIES_POINT_VALUE_CONFIG,
) -> float:
    if not gold_series:
        return 1.0 if not predicted_series else 0.0

    all_gold_pairs = [
        pair for gold_points in gold_series.values() for pair in _point_pairs(gold_points)
    ]
    if not all_gold_pairs:
        return 1.0 if not predicted_series else 0.0

    gold_xs = [x for x, _ in all_gold_pairs]
    gold_ys = [y for _, y in all_gold_pairs]
    x_min = min(gold_xs)
    y_min = min(gold_ys)
    x_scale = max(max(gold_xs) - x_min, 1.0)
    y_scale = max(max(gold_ys) - y_min, 1.0)

    weighted_score_sum = 0.0
    total_weight = 0

    for name, gold_points in gold_series.items():
        weight = max(len(gold_points), 1)
        predicted_points = predicted_series.get(name, [])
        weighted_score_sum += (
            series_point_value_score(
                predicted_points,
                gold_points,
                x_min=x_min,
                x_scale=x_scale,
                y_min=y_min,
                y_scale=y_scale,
                config=config,
            )
            * weight
        )
        total_weight += weight

    return weighted_score_sum / total_weight if total_weight else 0.0


def series_point_value(
    state: RubricState, info, series_point_value_config: SeriesPointValueConfig
) -> float:
    return series_point_value_raw(
        state,
        info,
        series_point_value_config,
    )


def series_point_value_raw(
    state: RubricState,
    info,
    series_point_value_config: SeriesPointValueConfig,
) -> float:
    if "series_point_value_raw" in state:
        return state["series_point_value_raw"]

    parsed_answer = state["parsed_answer"] if "parsed_answer" in state else None
    if parsed_answer is None:
        state["series_point_value_raw"] = 0.0
        return 0.0

    gold_answer = parse_chart_extraction(
        info.get(
            "expected_answer",
            {
                "title": info.get("title", ""),
                "x_axis_label": info.get("x_axis_label", ""),
                "y_axis_label": info.get("y_axis_label", ""),
                "series": info.get("series", []),
            },
        ),
        schema_version=info.get("schema_version", "v1"),
    ).to_canonical()

    predicted_series: dict[str, list[CanonicalPoint]] = {
        item.name: item.points for item in parsed_answer.series if item.name
    }
    gold_series: dict[str, list[CanonicalPoint]] = {
        item.name: item.points for item in gold_answer.series if item.name
    }

    raw_score = series_point_value_chart_score(
        predicted_series,
        gold_series,
        config=series_point_value_config,
    )
    state["series_point_value_raw"] = raw_score
    return raw_score
