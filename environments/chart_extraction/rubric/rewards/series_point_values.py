"""
Naive point-value reward for chart series extraction.

Algorithm:
1. Match series by name.
2. For each gold point, look for a predicted point with the exact same x value.
3. If that x value is missing, the gold point gets score 0.
4. If that x value exists, score it as:

       max(0, 1 - abs(predicted_y - gold_y) / y_scale)

   where y_scale is the y-range of the gold series, with a minimum of 1.
5. Average point scores within each series.
6. Take a weighted average across series using the number of gold points.
"""

from ..state import RubricState


def series_point_value_score(
    predicted_points: list[list[float]],
    gold_points: list[list[float]],
) -> float:
    if not predicted_points and not gold_points:
        return 1.0
    if not gold_points:
        return 1.0 if not predicted_points else 0.0
    if not predicted_points:
        return 0.0

    predicted_y_by_x = {
        float(point[0]): float(point[1])
        for point in predicted_points
        if len(point) == 2
    }
    gold_pairs = [
        (float(point[0]), float(point[1]))
        for point in gold_points
        if len(point) == 2
    ]

    if not gold_pairs:
        return 1.0 if not predicted_y_by_x else 0.0

    gold_ys = [gold_y for _, gold_y in gold_pairs]
    y_scale = max(max(gold_ys) - min(gold_ys), 1.0)

    total_score = 0.0
    for gold_x, gold_y in gold_pairs:
        predicted_y = predicted_y_by_x.get(gold_x)
        if predicted_y is None:
            continue

        normalized_y_error = abs(predicted_y - gold_y) / y_scale
        total_score += max(0.0, 1.0 - normalized_y_error)

    return total_score / len(gold_pairs)


async def series_point_value(
    state: RubricState,
    info,
) -> float:
    parsed_answer = state["parsed_answer"] if "parsed_answer" in state else None
    if parsed_answer is None:
        return 0.0

    predicted_series = {
        item.name: item.points
        for item in parsed_answer.series
        if item.name
    }
    gold_series = {
        item["name"]: item.get("points", [])
        for item in info.get("series", [])
        if item.get("name")
    }

    if not gold_series:
        return 1.0 if not predicted_series else 0.0

    weighted_score_sum = 0.0
    total_weight = 0

    for name, gold_points in gold_series.items():
        weight = max(len(gold_points), 1)
        predicted_points = predicted_series.get(name, [])
        weighted_score_sum += (
            series_point_value_score(predicted_points, gold_points) * weight
        )
        total_weight += weight

    return weighted_score_sum / total_weight if total_weight else 0.0
