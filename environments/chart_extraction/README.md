# chart-extraction

### Overview

- **Environment ID**: `chart-extraction`
- **Short description**: Single-turn chart-to-structure extraction with synthetic line and bar charts
- **Tags**: `vision`, `chart-extraction`, `synthetic`, `single-turn`

### Datasets

- **Primary dataset(s)**: synthetic charts generated from `matplotlib` by the local generator package
- **Upload target**: Hugging Face dataset with `id`, `image`, `answer`, and `info` columns
- **Default local fallback**: in-memory synthetic dataset generation for smoke tests

### Task

- **Type**: `single-turn`
- **Output modes**: `json` or `markdown`
- **Rubric overview**: format validity, chart type, series name, point count, and point-value accuracy

### Quickstart

Generate a local smoke dataset:

```bash
cd environments/chart_extraction
python -m chart_extraction_env.dataset_cli --train-size 128 --validation-size 32 --test-size 32
```

Install the environment:

```bash
prime env install chart-extraction
```

Run a local smoke eval using the in-memory fallback dataset:

```bash
prime eval run chart-extraction -m gpt-4.1-mini -n 5 -a '{"output_mode":"json","default_examples":32}'
```

Run against a local saved dataset:

```bash
prime eval run chart-extraction -m gpt-4.1-mini -n 5 -a '{"dataset_local_path":"environments/chart_extraction/outputs/chart_extraction/smoke_dataset_v2","output_mode":"json"}'
```

Run the fast local test suite:

```bash
PYTHONPATH=environments/chart_extraction python -m unittest discover -s environments/chart_extraction/tests -p 'test_*.py' -v
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `split` | `str` | `"train"` | Dataset split to load |
| `eval_split` | `str \| None` | `None` | Eval split; defaults to `validation` when `split="train"` |
| `output_mode` | `str` | `"json"` | Prompt and parser mode: `json` or `markdown` |
| `dataset_local_path` | `str \| None` | `None` | Local Hugging Face dataset saved with `save_to_disk` |
| `dataset_repo_id` | `str \| None` | `None` | Hugging Face dataset repo id |
| `max_examples` | `int` | `-1` | Optional example cap |
| `seed` | `int` | `7` | Seed for in-memory fallback generation |
| `default_examples` | `int` | `64` | Number of fallback examples when no dataset source is passed |

### Metrics

| Metric | Meaning |
| --- | --- |
| `reward` | Weighted sum of the rubric functions |
| `format_valid` | Whether the model response parses in the selected mode |
| `chart_type_score` | Exact chart type match |
| `series_name_score` | Exact normalized series-name match |
| `point_count_score` | Predicted vs target point-count overlap |
| `point_value_score` | Continuous score on x/y point accuracy |
| `x_type_score_metric` | Zero-weight metric for `x_type` accuracy |
| `point_x_score_metric` | Zero-weight metric for x-value accuracy |
| `point_y_score_metric` | Zero-weight metric for y-value accuracy |
| `exact_answer_score_metric` | Zero-weight exact canonical-answer match |
| `normalized_y_mae_metric` | Zero-weight normalized y-error, where lower is better |
