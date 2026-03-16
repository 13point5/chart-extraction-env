from pydantic import Field

from .base import NormalizedBaseModel


class CanonicalPoint(NormalizedBaseModel):
    """Schema-agnostic point representation used internally for scoring."""

    index: int = Field(description="Zero-based point index within the series", ge=0)
    x: float = Field(description="X-axis value")
    y: float = Field(description="Y-axis value")


class CanonicalSeries(NormalizedBaseModel):
    """Schema-agnostic series representation used internally for scoring."""

    name: str = Field(description="Series name")
    points: list[CanonicalPoint] = Field(
        description='Data points: [{"index": 0, "x": x0, "y": y0}, {"index": 1, "x": x1, "y": y1}, ...]'
    )


class CanonicalChart(NormalizedBaseModel):
    """Schema-agnostic extraction representation used internally for scoring."""

    title: str = Field(description="Title of the chart")
    x_axis_label: str = Field(description="Label of the x-axis")
    y_axis_label: str = Field(description="Label of the y-axis")
    series: list[CanonicalSeries] = Field(description="List of series in the chart")
