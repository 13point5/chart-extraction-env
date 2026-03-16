from pydantic import Field

from .base import NormalizedBaseModel
from .canonical import CanonicalChart, CanonicalPoint, CanonicalSeries


class Point_V2(NormalizedBaseModel):
    """V2 indexed point representation used for the object-based schema."""

    index: int = Field(description="Zero-based point index within the series", ge=0)
    x: float = Field(description="X-axis value")
    y: float = Field(description="Y-axis value")


class Series_V2(NormalizedBaseModel):
    """V2 series representation used for the object-based schema."""

    name: str = Field(description="Series name")
    points: list[Point_V2] = Field(
        description='Data points: [{"index": 0, "x": x0, "y": y0}, {"index": 1, "x": x1, "y": y1}, ...]'
    )


class Chart_V2(NormalizedBaseModel):
    """V2 chart extraction schema with explicit indexed point objects."""

    title: str = Field(description="Title of the chart")
    x_axis_label: str = Field(description="Label of the x-axis")
    y_axis_label: str = Field(description="Label of the y-axis")
    series: list[Series_V2] = Field(description="List of series in the chart")

    def to_canonical(self) -> CanonicalChart:
        return CanonicalChart(
            title=self.title,
            x_axis_label=self.x_axis_label,
            y_axis_label=self.y_axis_label,
            series=[
                CanonicalSeries(
                    name=series.name,
                    points=[
                        CanonicalPoint(index=point.index, x=float(point.x), y=float(point.y))
                        for point in series.points
                    ],
                )
                for series in self.series
            ],
        )
