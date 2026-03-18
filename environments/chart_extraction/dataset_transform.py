import base64
import io
import json
from typing import Any

from datasets import load_dataset

from prompts import SystemPromptVersion, get_system_prompt
from schemas import (
    Chart_V1,
    Chart_V2,
    Point_V2,
    Series_V1,
    Series_V2,
    SchemaVersion,
)

USER_PROMPT_TEXT = "Extract the data from this line chart image."


def strip_text(text: str | None) -> str:
    return text.strip() if text else ""


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def get_first_text(chart_elements: list[dict[str, Any]], category_name: str) -> str:
    for element in chart_elements:
        if element["category_name"] == category_name and element.get("text"):
            return strip_text(element["text"])
    return ""


def series_points_v1(raw_series_xy: list[float]) -> list[tuple[float, float]]:
    return [
        (float(raw_series_xy[i]), float(raw_series_xy[i + 1]))
        for i in range(0, len(raw_series_xy), 2)
    ]


def series_points_v2(raw_series_xy: list[float]) -> list[Point_V2]:
    return [
        Point_V2(
            index=i // 2,
            x=float(raw_series_xy[i]),
            y=float(raw_series_xy[i + 1]),
        )
        for i in range(0, len(raw_series_xy), 2)
    ]


def build_expected_answer(
    row: dict[str, Any],
    schema_version: SchemaVersion,
) -> Chart_V1 | Chart_V2:
    chart_elements = row["chart_elements"]

    if schema_version == "v1":
        series = [
            Series_V1(
                name=strip_text(line["line_name"]),
                points=series_points_v1(line["raw_series_xy"]),
            )
            for line in row["lines"]
        ]

        return Chart_V1(
            title=get_first_text(chart_elements, "ChartTitle"),
            x_axis_label=get_first_text(chart_elements, "CategoryAxisTitle"),
            y_axis_label=get_first_text(chart_elements, "ValueAxisTitle"),
            series=series,
        )

    series = [
        Series_V2(
            name=strip_text(line["line_name"]),
            points=series_points_v2(line["raw_series_xy"]),
        )
        for line in row["lines"]
    ]

    return Chart_V2(
        title=get_first_text(chart_elements, "ChartTitle"),
        x_axis_label=get_first_text(chart_elements, "CategoryAxisTitle"),
        y_axis_label=get_first_text(chart_elements, "ValueAxisTitle"),
        series=series,
    )


def build_info(
    row: dict[str, Any],
    schema_version: SchemaVersion,
    system_prompt: SystemPromptVersion,
) -> str:
    expected_answer = build_expected_answer(row, schema_version=schema_version)
    expected_answer_dict = expected_answer.model_dump(mode="json")

    info = {
        "schema_version": schema_version,
        "system_prompt": system_prompt,
        "image_id": row["image_id"],
        "file_name": row["file_name"],
        "width": row["width"],
        "height": row["height"],
        "data_type": row["data_type"],
        "chart_elements": row["chart_elements"],
        "lines": row["lines"],
        "title": expected_answer.title,
        "legend_names": [item.name for item in expected_answer.series],
        "x_axis_label": expected_answer.x_axis_label,
        "y_axis_label": expected_answer.y_axis_label,
        "series": expected_answer_dict["series"],
        "expected_answer": expected_answer_dict,
    }

    return json.dumps(info, separators=(",", ":"))


def build_prompt(
    image,
    schema_version: SchemaVersion,
    system_prompt: SystemPromptVersion,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": get_system_prompt(
                        schema_version=schema_version,
                        system_prompt=system_prompt,
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT_TEXT},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ],
        },
    ]


def transform_row(
    row: dict[str, Any],
    schema_version: SchemaVersion,
    system_prompt: SystemPromptVersion,
) -> dict[str, Any]:
    return {
        "prompt": build_prompt(
            row["image"],
            schema_version=schema_version,
            system_prompt=system_prompt,
        ),
        "info": build_info(
            row,
            schema_version=schema_version,
            system_prompt=system_prompt,
        ),
    }


def load_chart_extraction_dataset(
    split: str = "test",
    schema_version: SchemaVersion = "v1",
    system_prompt: SystemPromptVersion = "v1",
    max_examples: int | None = None,
):
    def build():
        split_spec = split
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be a positive integer when provided")
            split_spec = f"{split}[:{max_examples}]"

        dataset = load_dataset("13point5/line-ex", split=split_spec)

        return dataset.map(
            lambda row: transform_row(
                row,
                schema_version=schema_version,
                system_prompt=system_prompt,
            ),
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )

    return build
