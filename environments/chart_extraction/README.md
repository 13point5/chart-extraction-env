# chart-extraction

### Overview
- **Environment ID**: `chart-extraction`
- **Short description**: Extract structured line-chart data from chart images.
- **Tags**: `single-turn`, `multimodal`, `vision`, `eval`

### Datasets
- **Primary dataset(s)**: `13point5/lineex-test`, a line-chart image dataset with chart text annotations and ground-truth series points.
- **Source links**: `https://huggingface.co/datasets/13point5/lineex-test`
- **Split sizes**: eval uses the `test` split with 20,000 examples.

### Task
- **Type**: `single-turn`
- **Parser**: `XMLParser(["answer"])`
- **Output format expectations**: Return a JSON object matching the chart extraction schema, wrapped in `<answer>...</answer>` tags.
- **Rubric overview**: One parser-based format reward checks that the model follows the expected `<answer>` output format.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run chart-extraction
```

Configure a vision model and a small smoke test:

```bash
prime eval run chart-extraction -m 'qwen/qwen3-vl-8b-instruct' -n 1 -r 1 -a '{"max_examples":1}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward |
| `format_reward_func` | Output-format adherence score from the XML parser reward |
| `num_turns` | Number of turns taken in the rollout |
