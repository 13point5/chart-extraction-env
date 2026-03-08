from __future__ import annotations

import base64
import io

from PIL import Image as PILImage

from chart_extraction_env.dataset.models import OutputMode


def build_user_prompt(*, image: PILImage.Image, output_mode: OutputMode) -> list[dict[str, object]]:
    prompt_text = (
        json_prompt_text()
        if output_mode == OutputMode.JSON
        else markdown_prompt_text()
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ],
        }
    ]


def json_prompt_text() -> str:
    return (
        "Extract the chart data from the image. Return only valid JSON with this schema:\n"
        "{\n"
        '  "chart_type": "line" | "bar",\n'
        '  "x_type": "numeric" | "categorical",\n'
        '  "series": [\n'
        "    {\n"
        '      "name": "series_1",\n'
        '      "points": [{"x": <number or string>, "y": <number>}, ...]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        'Return one series only and always use "series_1" as the series name. '
        "Do not add explanation or markdown fences."
    )


def markdown_prompt_text() -> str:
    return (
        "Extract the chart data from the image and return Markdown only in this format:\n"
        "chart_type: <line|bar>\n"
        "x_type: <numeric|categorical>\n"
        "## Series: series_1\n"
        "| x | y |\n"
        "| --- | --- |\n"
        "| ... | ... |\n"
        'Return one series only and always use "series_1" as the series name. '
        "Do not add explanation outside the format."
    )


def image_to_data_url(image: PILImage.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
