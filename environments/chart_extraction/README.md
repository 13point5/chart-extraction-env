# chart-extraction

### Overview

- **Environment ID**: `chart-extraction`
- **Short description**: Extract structured line-chart data from chart images.
- **Tags**: `single-turn`, `multimodal`, `vision`

### Datasets

- **Primary dataset(s)**: `13point5/line-ex`, a line-chart image dataset with chart text annotations and ground-truth series points.
- **Source links**:
  - Dataset: [13point5/line-ex on Hugging Face](https://huggingface.co/datasets/13point5/line-ex "Hugging Face dataset for the chart-extraction environment")
  - Paper: [LineEX: Data Extraction From Scientific Line Charts](https://openaccess.thecvf.com/content/WACV2023/papers/P._LineEX_Data_Extraction_From_Scientific_Line_Charts_WACV_2023_paper.pdf "Original LineEX paper from WACV 2023")
  - Upload and analysis repo: [13point5/line-ex-paper-analysis](https://github.com/13point5/line-ex-paper-analysis "Scripts and analysis for uploading the original LineEX dataset to Hugging Face") (includes the scripts used to upload the original LineEX paper dataset to Hugging Face)
- **Split sizes**: `train` has 30,000 examples and `test` has 20,000 examples.

### Task

- **Type**: `single-turn`
- **Parser**: `XMLParser(["answer"])`
- **Output format expectations**: Return a JSON object matching the selected chart extraction schema, wrapped in `<answer>...</answer>` tags.
- **Schema versions**:
  - `v1`: original compact point format `[[x0, y0], [x1, y1], ...]`
  - `v2`: explicit point objects with `index`, `x`, and `y`
- **Schema implementation**:
  - versioned model schemas live in [`schemas/v1.py`](./schemas/v1.py) and [`schemas/v2.py`](./schemas/v2.py)
  - rewards use a schema-agnostic internal shape from [`schemas/canonical.py`](./schemas/canonical.py)
  - both `v1` and `v2` parse into typed Pydantic models and then convert via `.to_canonical()` before scoring
- **Rubric overview**: The main `reward` is a weighted sum of four rewards:
  - `format_reward_func` (`weight = 1.0`): checks that the response follows the expected `<answer>...</answer>` format.
  - `series_name_f1` (`weight = 1.0`): computes F1 between predicted series names and gold legend names.
  - `series_point_count_ratio` (`weight = 2.0`): scores agreement on how many points each gold series contains, weighted by series length.
  - `series_point_value` (`weight = 2.0`): scores matched series points with a point-only OKS criterion, giving credit only when predicted points land close to labeled gold points after chart-scale normalization. It does not give credit for landing somewhere along the line segment between gold points.
- **Info payload**: The dataset `info` JSON includes `schema_version`, `expected_answer` in the same schema shown to the model, and the original dataset columns such as `chart_elements` and `lines` so you can compare raw annotations directly in Prime's UI.

### Quickstart

Run an evaluation with a vision model:

```bash
prime eval run chart-extraction -m 'qwen/qwen3-vl-8b-instruct' -n 1 -r 1
```

Run the `v2` schema explicitly:

```bash
prime eval run chart-extraction -m 'qwen/qwen3-vl-8b-instruct' -n 1 -r 1 --env-kwargs '{"schema_version":"v2"}'
```

Notes:

- Use `-n` / `--num-examples` to limit how many examples are evaluated.

### Environment Arguments

- `schema_version`: chooses the output schema and matching gold `expected_answer` shape.
  - default: `"v1"`
  - `"v1"`: original compact point pairs
  - `"v2"`: explicit indexed point objects

The environment always uses the dataset `train` split for rollouts and the `test` split for eval.

### Metrics

| Metric                     | Meaning                                                                        |
| -------------------------- | ------------------------------------------------------------------------------ |
| `reward`                   | Main scalar reward: weighted sum of the four rubric rewards                    |
| `format_reward_func`       | Output-format adherence score from the XML parser reward                       |
| `series_name_f1`           | F1 score for predicted series names versus gold legend names                   |
| `series_point_count_ratio` | Weighted agreement on the number of points in each gold series                 |
| `series_point_value`       | Weighted point-only OKS score for labeled gold points, without nearby line-segment credit |
| `num_turns`                | Number of turns taken in the rollout                                           |

### Parsing And Scoring Flow

1. The environment chooses a schema version via `load_environment(schema_version=...)`.
2. The model output is validated against the corresponding typed schema:
   - `Chart_V1`
   - `Chart_V2`
3. The parsed schema object converts itself into `CanonicalChart` via `.to_canonical()`.
4. All reward functions operate on that canonical internal representation.

This means the env does not infer schema version from payload shape. The configured `schema_version` is used directly for both model outputs and gold `expected_answer` parsing.

### `series_point_value` reward

This reward is a strict point-matching metric inspired by the OKS portion of the LineEX keypoint metric, but without the relaxed line-segment fallback.

Algorithm:

1. Match predicted and gold series by exact series name.
2. Collect all gold points across the chart and normalize `x` and `y` coordinates by the full gold chart span so the tolerance is scale-aware.
3. For each predicted point in a matched series, find the nearest labeled gold point in that same series.
4. Convert that normalized point distance `d` into an OKS score:

```text
OKS(d) = exp(-(d^2) / (2 * k^2)), where k = 0.025
```

5. Count the predicted point as a match only if `OKS(d) > 0.5`.
6. Score each series as:

```text
matched_unique_gold_points / total_gold_points
```

7. Return the weighted average of those series scores, using the number of gold points in each series as the weight.

Implications:

- Small `x` and `y` errors around a labeled gold point can still earn credit.
- A prediction does not earn credit just for lying near the curve between labeled points.
- Extra predicted points do not help unless they land close enough to distinct labeled gold points.
