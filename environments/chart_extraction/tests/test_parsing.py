from __future__ import annotations

import unittest

from chart_extraction_env.dataset.models import ChartType, OutputMode, XType
from chart_extraction_env.parsing import parse_response


class ParsingTests(unittest.TestCase):
    def test_parse_json_response_accepts_code_fence(self) -> None:
        response = """```json
        {
          "chart_type": "line",
          "x_type": "numeric",
          "series": [
            {
              "name": "series_1",
              "points": [
                {"x": 0, "y": 1.25},
                {"x": 1, "y": 2.5}
              ]
            }
          ]
        }
        ```"""

        parsed = parse_response(response, OutputMode.JSON)

        self.assertEqual(parsed.chart_type, ChartType.LINE)
        self.assertEqual(parsed.x_type, XType.NUMERIC)
        self.assertEqual(parsed.series[0].name, "series_1")
        self.assertEqual(parsed.series[0].points[1].x, 1.0)
        self.assertEqual(parsed.series[0].points[1].y, 2.5)

    def test_parse_markdown_response_extracts_categorical_points(self) -> None:
        response = """
        chart_type: bar
        x_type: categorical
        ## Series: series_1
        | x | y |
        | --- | --- |
        | North America | 98.95 |
        | Self-Serve Customers | 61.0 |
        """

        parsed = parse_response(response, OutputMode.MARKDOWN)

        self.assertEqual(parsed.chart_type, ChartType.BAR)
        self.assertEqual(parsed.x_type, XType.CATEGORICAL)
        self.assertEqual(parsed.series[0].name, "series_1")
        self.assertEqual(parsed.series[0].points[0].x, "North America")
        self.assertEqual(parsed.series[0].points[0].y, 98.95)


if __name__ == "__main__":
    unittest.main()
