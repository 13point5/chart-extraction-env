"""
Point-count reward for chart series extraction.

Algorithm:
1. Match predicted series to gold series by normalized series name.
2. For each gold series, compare only the number of predicted points versus the
   number of gold points.
3. Compute a per-series count ratio as:

       min(predicted_count, gold_count) / max(predicted_count, gold_count)

   This gives:
   - 1.0 for an exact count match
   - a partial score for over- or under-predicting
   - 0.0 when one side has points and the other has none
4. Weight each per-series ratio by the number of gold points in that series, so
   larger series contribute more to the final reward.
5. Return the weighted average of those per-series contributions.

This reward measures count agreement only. It does not check whether the actual
point coordinates are correct.
"""

import logging

import numpy as np
from pydantic import BaseModel

from schema import ChartSeries
from ..state import RubricState


class GoldSeries(BaseModel):
    name: str
    points: list[list[float]]


class SeriesPointCountContribution(BaseModel):
    ratio: float
    weight: int


def point_count_ratio(
    predicted_points: list[list[float]],
    gold_points: list[list[float]],
) -> float:
    counts = np.asarray([len(predicted_points), len(gold_points)], dtype=float)

    if np.all(counts == 0):
        return 1.0
    if np.any(counts == 0):
        return 0.0

    return float(counts.min() / counts.max())


def weighted_series_average(
    contributions: list[SeriesPointCountContribution],
) -> float:
    if not contributions:
        return 0.0

    ratios = np.asarray(
        [contribution.ratio for contribution in contributions],
        dtype=float,
    )
    weights = np.asarray(
        [contribution.weight for contribution in contributions],
        dtype=float,
    )

    return float(np.average(ratios, weights=weights))


async def series_point_count_ratio_reward(
    state: RubricState,
    info,
    logger: logging.Logger,
) -> float:
    parsed_answer = state["parsed_answer"] if "parsed_answer" in state else None
    if parsed_answer is None:
        return 0.0

    predicted_series: dict[str, ChartSeries] = {
        item.name: item for item in parsed_answer.series if item.name
    }

    gold_series: dict[str, GoldSeries] = {
        series.name: series
        for series in (
            GoldSeries.model_validate(item) for item in info.get("series", []) if item.get("name")
        )
    }

    if not gold_series:
        return 1.0 if not predicted_series else 0.0

    contributions: list[SeriesPointCountContribution] = []
    for name, gold_series_item in gold_series.items():
        predicted_points = predicted_series[name].points if name in predicted_series else []

        ratio = point_count_ratio(predicted_points, gold_series_item.points)
        weight = max(len(gold_series_item.points), 1)

        contribution = SeriesPointCountContribution(
            ratio=ratio,
            weight=weight,
        )
        contributions.append(contribution)

    return weighted_series_average(contributions)
