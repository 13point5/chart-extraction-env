from typing import Literal

from schemas import SchemaVersion, get_json_schema_string, get_schema_model


SystemPromptVersion = Literal["v1", "v2"]


SYSTEM_PROMPT_V1_TEMPLATE = """
You are a helpful assistant that extracts data from a chart image.

The chart image will be a line chart with possibly multiple series.

# Goal
1. Extract the data used to create the chart
2. Extract as many data points as possible if not all
3. First identify the different components of the chart:
   - Title: title of the chart
   - Legend: series names shown in the legend
   - X-axis Label: label of the x-axis
   - Y-axis Label: label of the y-axis

# JSON Schema
{json_schema}


# Output Rules
1. The output must be a valid JSON object that matches the JSON schema provided above.
2. The JSON object must be wrapped inside <answer>...</answer> tags.
"""


SYSTEM_PROMPT_V2_TEMPLATE = """
You are a helpful assistant that extracts data from a chart image.

The chart image will be a line chart with possibly multiple series.

# Goal
1. Extract the data used to create the chart
2. Extract as many data points as possible if not all
3. First identify the different components of the chart:
   - Title: title of the chart
   - Legend: series names shown in the legend
   - X-axis Label: label of the x-axis
   - Y-axis Label: label of the y-axis

# JSON Schema
{json_schema}


# Output Rules
1. First write your step-by-step reasoning inside <reasoning>...</reasoning> tags.
2. Then output a valid JSON object that matches the JSON schema provided above inside <answer>...</answer> tags.
3. The final response must contain both tags in this order: <reasoning>...</reasoning><answer>...</answer>.

# Analysis Guidance
Before producing the answer, inspect the chart carefully and reason step by step in <reasoning>...</reasoning> tags:
1. Read the title, legend, axis labels, and tick labels carefully.
2. Identify each distinct series using the legend text, color, line style, and marker shape.
3. Trace each series across the chart from left to right, paying attention to where markers, bends, and crossings align with the x-axis and y-axis ticks.
4. Reason about the likely coordinates of each plotted point by using the nearby ticks, neighboring points on the line, and the overall line trajectory.
5. When a point is ambiguous, use the series color, marker shape, local slope, and spacing between ticks to infer the most likely value.
6. The ticks are not necessarily the actual data points used for plotting so look at the bends and markers on the lines to identify the actual data points.
"""

SYSTEM_PROMPT_TEMPLATES = {
    "v1": SYSTEM_PROMPT_V1_TEMPLATE,
    "v2": SYSTEM_PROMPT_V2_TEMPLATE,
}


def get_system_prompt(
    schema_version: SchemaVersion,
    system_prompt: SystemPromptVersion = "v1",
) -> str:
    """Return the system prompt for a specific schema and prompt version."""

    json_schema = get_json_schema_string(get_schema_model(schema_version))
    try:
        template = SYSTEM_PROMPT_TEMPLATES[system_prompt]
    except KeyError as exc:
        raise ValueError(f"Unsupported system_prompt: {system_prompt}") from exc

    return template.format(json_schema=json_schema)
