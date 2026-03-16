import json
from typing import Literal

from pydantic import BaseModel

from .canonical import CanonicalChart, CanonicalPoint, CanonicalSeries
from .v1 import Chart_V1, Series_V1
from .v2 import Chart_V2, Point_V2, Series_V2


SchemaVersion = Literal["v1", "v2"]


SCHEMA_MODELS: dict[SchemaVersion, type[BaseModel]] = {
    "v1": Chart_V1,
    "v2": Chart_V2,
}


def get_schema_model(schema_version: SchemaVersion) -> type[BaseModel]:
    """Return the Pydantic model for a schema version."""

    try:
        return SCHEMA_MODELS[schema_version]
    except KeyError as exc:
        raise ValueError(f"Unsupported schema_version: {schema_version}") from exc


def parse_chart_extraction(
    answer: dict,
    schema_version: SchemaVersion,
) -> Chart_V1 | Chart_V2:
    """Parse a raw dict with an explicit schema version."""

    return get_schema_model(schema_version).model_validate(answer)

def get_json_schema_string(pydantic_model: type[BaseModel], indent: int = 2) -> str:
    """Return the JSON schema for a Pydantic model as a formatted string."""

    schema = pydantic_model.model_json_schema()
    return json.dumps(schema, indent=indent)


__all__ = [
    "CanonicalChart",
    "CanonicalPoint",
    "CanonicalSeries",
    "Chart_V1",
    "Chart_V2",
    "Point_V2",
    "Series_V1",
    "Series_V2",
    "SchemaVersion",
    "get_json_schema_string",
    "get_schema_model",
    "parse_chart_extraction",
]
