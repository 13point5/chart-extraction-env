import base64
import io
import json

from datasets import load_dataset
from prompts import DEFAULT_SYSTEM_PROMPT_V1

USER_PROMPT_TEXT = "Extract the data from this line chart image."


def strip_text(text: str | None) -> str:
    return text.strip() if text else ""


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def get_first_text(chart_elements: list[dict], category_name: str) -> str:
    for element in chart_elements:
        if element["category_name"] == category_name and element.get("text"):
            return strip_text(element["text"])
    return ""


def series_points(raw_series_xy: list[float]) -> list[list[float]]:
    return [
        [float(raw_series_xy[i]), float(raw_series_xy[i + 1])]
        for i in range(0, len(raw_series_xy), 2)
    ]


def build_info(row: dict) -> str:
    chart_elements = row["chart_elements"]
    series = [
        {
            "name": strip_text(line["line_name"]),
            "points": series_points(line["raw_series_xy"]),
        }
        for line in row["lines"]
    ]

    info = {
        "image_id": row["image_id"],
        "file_name": row["file_name"],
        "width": row["width"],
        "height": row["height"],
        "data_type": row["data_type"],
        "title": get_first_text(chart_elements, "ChartTitle"),
        "legend_names": [item["name"] for item in series],
        "x_axis_label": get_first_text(chart_elements, "CategoryAxisTitle"),
        "y_axis_label": get_first_text(chart_elements, "ValueAxisTitle"),
        "series": series,
    }

    return json.dumps(info, separators=(",", ":"))


def build_prompt(image) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT_V1}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT_TEXT},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ],
        },
    ]


def transform_row(row: dict) -> dict:
    return {
        "prompt": build_prompt(row["image"]),
        "info": build_info(row),
    }


def load_chart_extraction_dataset(
    split: str = "test",
):
    dataset = load_dataset("13point5/line-ex", split=split)

    return dataset.map(
        lambda row: transform_row(row),
        remove_columns=dataset.column_names,
    )
