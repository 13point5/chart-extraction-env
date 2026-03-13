from pydantic import BaseModel
from schema import ChartExtraction_V1, get_json_schema_string


SYSTEM_PROMPT_TEMPLATE_V1 = """
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
   - Data points for each series: [[x0, y0], [x1, y1], ...] depending on the JSON schema provided below

# JSON Schema
{json_schema}

# Output Rules
1. The output must be a valid JSON object that matches the JSON schema provided above.
2. The JSON object must be wrapped inside <answer>...</answer> tags.
"""


def get_system_prompt(
    pydantic_model: type[BaseModel],
    template: str = SYSTEM_PROMPT_TEMPLATE_V1,
) -> str:
    """Return the system prompt with the JSON schema. Uses ChartExtraction schema by default."""

    json_schema = get_json_schema_string(pydantic_model)
    prompt = template.format(json_schema=json_schema)

    return prompt


DEFAULT_SYSTEM_PROMPT_V1 = get_system_prompt(
    pydantic_model=ChartExtraction_V1,
    template=SYSTEM_PROMPT_TEMPLATE_V1,
)
