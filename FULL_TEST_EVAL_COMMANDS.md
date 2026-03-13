# Full Test Eval Commands

These commands run hosted evals on the original base model `qwen/qwen3-vl-8b-instruct` against the full held-out `test` split of `13point5/chart-extraction-dense-noisy-v1`, restricted to line charts only.

After `chart_type_filter="line"`, the test split contains `700` examples.

## Markdown

```bash
prime eval run 13point5/chart-extraction \
  --endpoints-path configs/endpoints.toml \
  --model qwen/qwen3-vl-8b-instruct \
  --env-args '{"split":"test","eval_split":"test","output_mode":"markdown","dataset_repo_id":"13point5/chart-extraction-dense-noisy-v1","chart_type_filter":"line"}' \
  --num-examples 700 \
  --rollouts-per-example 3 \
  --max-concurrent 12 \
  --max-tokens 15000 \
  --temperature 0 \
  --max-retries 2 \
  --save-results
```

## JSON

```bash
prime eval run 13point5/chart-extraction \
  --endpoints-path configs/endpoints.toml \
  --model qwen/qwen3-vl-8b-instruct \
  --env-args '{"split":"test","eval_split":"test","output_mode":"json","dataset_repo_id":"13point5/chart-extraction-dense-noisy-v1","chart_type_filter":"line"}' \
  --num-examples 700 \
  --rollouts-per-example 3 \
  --max-concurrent 12 \
  --max-tokens 15000 \
  --temperature 0 \
  --max-retries 2 \
  --save-results
```

## Notes

- This evaluates the base model only, not an RL checkpoint.
- `split="test"` and `eval_split="test"` force the held-out test split.
- `chart_type_filter="line"` removes the bar charts from eval.
- `max_tokens=15000` is intentionally high to reduce truncation on dense line charts.
