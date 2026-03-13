"""Pydantic models for chart extraction output. JSON schema is derived from these."""

import json

from pydantic import BaseModel, ConfigDict, Field


class NormalizedBaseModel(BaseModel):
    """Base model that strips surrounding whitespace from all strings once."""

    model_config = ConfigDict(str_strip_whitespace=True)


class ChartSeries(NormalizedBaseModel):
    """A single series in the chart."""

    name: str = Field(description="Series name")
    points: list[list[float]] = Field(description="Data points: [[x0, y0], [x1, y1], ...]")


class ChartExtraction_V1(NormalizedBaseModel):
    """Extracted data from a line chart image."""

    title: str = Field(description="Title of the chart")
    x_axis_label: str = Field(description="Label of the x-axis")
    y_axis_label: str = Field(description="Label of the y-axis")
    series: list[ChartSeries] = Field(description="List of series in the chart")


def get_json_schema_string(pydantic_model: type[BaseModel], indent: int = 2) -> str:
    """Return the JSON schema for a Pydantic model as a formatted string."""

    schema = pydantic_model.model_json_schema()
    return json.dumps(schema, indent=indent)
