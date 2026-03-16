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
- **Output format expectations**: Return a JSON object matching the chart extraction schema, wrapped in `<answer>...</answer>` tags.
- **Rubric overview**: The main `reward` is an equally weighted average of four rewards:
  - `format_reward_func`: checks that the response follows the expected `<answer>...</answer>` format.
  - `series_name_f1`: computes F1 between predicted series names and gold legend names.
  - `series_point_count_ratio`: scores agreement on how many points each gold series contains, weighted by series length.
  - `series_point_value`: scores matched series points with a point-only OKS criterion, giving credit only when predicted points land close to labeled gold points after chart-scale normalization.

### Quickstart

Run an evaluation with a vision model:

```bash
prime eval run chart-extraction -m 'qwen/qwen3-vl-8b-instruct' -n 1 -r 1
```

Notes:

- Use `-n` / `--num-examples` to limit how many examples are evaluated.

### Environment Arguments

This environment does not currently expose custom `load_environment(...)` arguments.
It always uses the dataset `train` split for rollouts and the `test` split for eval.

### Metrics

| Metric                     | Meaning                                                                        |
| -------------------------- | ------------------------------------------------------------------------------ |
| `reward`                   | Main scalar reward: the equally weighted average of the four rubric rewards    |
| `format_reward_func`       | Output-format adherence score from the XML parser reward                       |
| `series_name_f1`           | F1 score for predicted series names versus gold legend names                   |
| `series_point_count_ratio` | Weighted agreement on the number of points in each gold series                 |
| `series_point_value`       | Weighted point-only OKS score for labeled gold points, without nearby line-segment credit |
| `num_turns`                | Number of turns taken in the rollout                                           |
