from __future__ import annotations

import unittest

from chart_extraction_env.dataset.generator import build_recipe
from chart_extraction_env.dataset.models import (
    CanonicalAnswer,
    ChartType,
    SeriesData,
    SeriesPoint,
    XType,
)
from chart_extraction_env.environment import point_component_scores


class RewardingTests(unittest.TestCase):
    def test_build_recipe_uses_series_1_for_single_series_target(self) -> None:
        recipe = build_recipe(seed=123, example_index=0)

        self.assertEqual(recipe.series_name, "series_1")
        self.assertEqual(recipe.answer.series[0].name, "series_1")

    def test_point_component_scores_are_perfect_for_exact_match(self) -> None:
        answer = CanonicalAnswer(
            chart_type=ChartType.LINE,
            x_type=XType.NUMERIC,
            series=[
                SeriesData(
                    name="series_1",
                    points=[
                        SeriesPoint(x=0.0, y=10.0),
                        SeriesPoint(x=1.0, y=15.0),
                    ],
                )
            ],
        )

        scores = point_component_scores(answer, answer)

        self.assertEqual(scores["point_value_score"], 1.0)
        self.assertEqual(scores["point_x_score"], 1.0)
        self.assertEqual(scores["point_y_score"], 1.0)
        self.assertEqual(scores["normalized_y_mae"], 0.0)

    def test_point_component_scores_penalize_missing_points_and_y_error(self) -> None:
        target = CanonicalAnswer(
            chart_type=ChartType.BAR,
            x_type=XType.CATEGORICAL,
            series=[
                SeriesData(
                    name="series_1",
                    points=[
                        SeriesPoint(x="A", y=2.0),
                        SeriesPoint(x="B", y=4.0),
                        SeriesPoint(x="C", y=6.0),
                    ],
                )
            ],
        )
        parsed = CanonicalAnswer(
            chart_type=ChartType.BAR,
            x_type=XType.CATEGORICAL,
            series=[
                SeriesData(
                    name="series_1",
                    points=[
                        SeriesPoint(x="A", y=2.5),
                        SeriesPoint(x="B", y=4.5),
                    ],
                )
            ],
        )

        scores = point_component_scores(parsed, target)

        self.assertEqual(scores["point_x_score"], 2.0 / 3.0)
        self.assertGreater(scores["point_y_score"], 0.0)
        self.assertLess(scores["point_y_score"], 2.0 / 3.0)
        self.assertGreater(scores["point_value_score"], 0.0)
        self.assertLess(scores["point_value_score"], 2.0 / 3.0)
        self.assertEqual(scores["normalized_y_mae"], 0.125)


if __name__ == "__main__":
    unittest.main()
