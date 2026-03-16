from schemas import SchemaVersion, get_json_schema_string, get_schema_model


SYSTEM_PROMPT_TEMPLATE = """
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

def get_system_prompt(schema_version: SchemaVersion) -> str:
    """Return the system prompt for a specific schema version."""

    return SYSTEM_PROMPT_TEMPLATE.format(
        json_schema=get_json_schema_string(get_schema_model(schema_version))
    )
