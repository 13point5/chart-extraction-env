from pydantic import Field

from .base import NormalizedBaseModel
from .canonical import CanonicalChart, CanonicalPoint, CanonicalSeries


class Series_V1(NormalizedBaseModel):
    """Original compact series representation with raw [x, y] pairs."""

    name: str = Field(description="Series name")
    points: list[tuple[float, float]] = Field(description="Data points: [[x0, y0], [x1, y1], ...]")


class Chart_V1(NormalizedBaseModel):
    """Original compact chart extraction schema."""

    title: str = Field(description="Title of the chart")
    x_axis_label: str = Field(description="Label of the x-axis")
    y_axis_label: str = Field(description="Label of the y-axis")
    series: list[Series_V1] = Field(description="List of series in the chart")

    def to_canonical(self) -> CanonicalChart:
        return CanonicalChart(
            title=self.title,
            x_axis_label=self.x_axis_label,
            y_axis_label=self.y_axis_label,
            series=[
                CanonicalSeries(
                    name=series.name,
                    points=[
                        CanonicalPoint(index=index, x=float(point[0]), y=float(point[1]))
                        for index, point in enumerate(series.points)
                    ],
                )
                for series in self.series
            ],
        )
