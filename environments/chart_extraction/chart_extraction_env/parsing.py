from __future__ import annotations

import json
import re

from chart_extraction_env.dataset.models import (
    CanonicalAnswer,
    ChartType,
    OutputMode,
    SeriesData,
    SeriesPoint,
    XType,
)


def extract_assistant_text(completion: list[dict[str, object]]) -> str:
    if not completion:
        return ""
    content = completion[-1].get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def parse_canonical_answer_json(payload: str) -> CanonicalAnswer:
    return CanonicalAnswer.model_validate(json.loads(payload))


def parse_response(text: str, output_mode: OutputMode) -> CanonicalAnswer:
    if output_mode == OutputMode.JSON:
        return parse_json_response(text)
    if output_mode == OutputMode.MARKDOWN:
        return parse_markdown_response(text)
    raise ValueError(f"Unsupported output mode: {output_mode}")


def parse_json_response(text: str) -> CanonicalAnswer:
    payload = _strip_code_fence(text)
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response does not contain a JSON object.")
    return CanonicalAnswer.model_validate(json.loads(payload[start : end + 1]))


def parse_markdown_response(text: str) -> CanonicalAnswer:
    chart_type = _match_scalar(text, "chart_type")
    x_type = _match_scalar(text, "x_type")
    series_name = _match_series_name(text)
    points = _parse_markdown_points(text, x_type)
    return CanonicalAnswer(
        chart_type=ChartType(chart_type),
        x_type=XType(x_type),
        series=[SeriesData(name=series_name, points=points)],
    )


def normalized_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _match_scalar(text: str, field_name: str) -> str:
    pattern = rf"^\s*{field_name}\s*:\s*([A-Za-z0-9_ -]+)\s*$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError(f"Missing scalar field: {field_name}")
    return match.group(1).strip().lower().replace(" ", "_")


def _match_series_name(text: str) -> str:
    patterns = [
        r"^\s*##\s*Series\s*:\s*(.+?)\s*$",
        r"^\s*series\s*:\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    raise ValueError("Missing series name.")


def _parse_markdown_points(text: str, x_type: str) -> list[SeriesPoint]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    table_lines = [line for line in lines if not re.match(r"^\|\s*-", line)]
    if len(table_lines) < 2:
        raise ValueError("Markdown table is missing point rows.")

    header_cells = _split_markdown_row(table_lines[0])
    if [cell.lower() for cell in header_cells] != ["x", "y"]:
        raise ValueError("Markdown table must start with | x | y |.")

    points: list[SeriesPoint] = []
    for row in table_lines[1:]:
        cells = _split_markdown_row(row)
        if len(cells) != 2:
            continue
        x_value: float | str
        if x_type == XType.NUMERIC.value:
            x_value = float(cells[0])
        else:
            x_value = cells[0]
        points.append(SeriesPoint(x=x_value, y=float(cells[1])))
    if not points:
        raise ValueError("Markdown table did not contain any points.")
    return points


def _split_markdown_row(row: str) -> list[str]:
    stripped = row.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]
