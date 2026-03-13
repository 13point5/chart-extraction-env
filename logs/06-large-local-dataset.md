# Large Local Dataset And Baseline

- 2026-03-11 21:12:01 EDT: started a new local-only dataset pass for a harder baseline. Goal: keep `line` and `bar`, bias heavily toward noisy dense line charts, allow up to 500 points, generate `train=3000`, `validation=1000`, `test=1000`, and confirm baseline headroom before any RL run.
- 2026-03-11 21:12:01 EDT: decided to preserve the original `v1` synthetic generator and add a second typed variant instead of mutating the existing dataset in place. This keeps previous results reproducible while letting the new dataset become a harder branch of the same package.

## Planned Changes

- add a `dense_noisy_v1` dataset variant to the typed generator models
- extend the sampler with dense noisy line profiles and business-style bar profiles
- update the renderer for dense tick spacing and large local charts
- add tests for dense point counts, label uniqueness, metadata, and rendering
- generate the dataset locally only under `outputs/`
- run baseline evals with a large output token budget and inspect whether reward leaves RL headroom

## Progress

- 2026-03-11 21:18:49 EDT: ran the fast local test suite after adding the new variant plumbing. Result: `9` tests passed, covering parsing, reward math, dense point-count coverage, label uniqueness, metadata recording, and large-chart rendering.
- 2026-03-11 21:18:49 EDT: generated a smoke dataset at `environments/chart_extraction/outputs/chart_extraction/dense_noisy_v1_smoke` with `train=48`, `validation=16`, `test=16` to inspect the variant before spending time on the full build.
- 2026-03-11 21:19:11 EDT: inspected the smoke dataset with `scripts/chart_extraction/inspect_dataset.py`. Observed split mix was roughly `73%` line / `27%` bar. The smoke train split had `num_points min=10 p50=96 p90=322 max=400 mean=139.7`, which confirms the dense regime is reachable while still mixing in shorter examples.
- 2026-03-11 21:19:11 EDT: visually spot-checked line and bar previews. The dense line charts looked like plausible ML or business trend plots, and the bar charts remained legible with rotated ticks and sparse labeling.
- 2026-03-11 21:19:57 EDT: committed the generator and test changes as `4bdc113` with message `Add dense noisy local dataset variant`.
- 2026-03-11 21:26:18 EDT: generated the full local dataset at `environments/chart_extraction/outputs/chart_extraction/dense_noisy_local_v1` with `train=3000`, `validation=1000`, `test=1000`.
- 2026-03-11 21:26:18 EDT: inspected the full dataset. Validation split stats were:
  - `chart_type`: `700` line / `300` bar
  - `num_points`: `min=6 p50=96 p90=297 max=499 mean=134.0`
  - `line_num_points`: `min=32 p50=179 p90=327 max=499 mean=185.0`
  - `bar_num_points`: `min=6 p50=14 p90=18 max=58 mean=14.9`
  - `answer_chars`: `min=230 p50=2305 p90=6650 max=10995 mean=3096.2`
- 2026-03-11 21:29:13 EDT: ran a `3` example JSON smoke eval against the local dataset with `max_examples=100` to make sure the environment could serve the harder dataset without truncation. Result: avg reward `0.5151`; the model kept the format correct but heavily underpredicted point counts and sparse-sampled dense lines.
- 2026-03-11 21:30:17 EDT: launched the main `100` example JSON baseline on the first `100` validation rows. The wrapper process stalled after saving results, but the eval artifacts were written successfully under run `14cab8c5`.
- 2026-03-11 21:42:55 EDT: launched the main `100` example Markdown baseline on the same local subset. Eval artifacts were written under run `c7fe28c4`.

## Baseline Report

### Baseline Setup

- model: `qwen/qwen3-vl-8b-instruct`
- dataset: local only, `environments/chart_extraction/outputs/chart_extraction/dense_noisy_local_v1`
- eval subset: first `100` rows from the validation split via `max_examples=100`
- completion budget: `max_tokens=8192`
- concurrency: `3`
- temperature: `0`

### Aggregate Results

| Run ID | Mode | Completed Rows | Avg Reward | Point Count | Point Value | X Type | Point X | Point Y | Output Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `14cab8c5` | JSON | 99 | `0.5858` | `0.4382` | `0.3082` | `0.9697` | `0.2226` | `0.3653` | `806.8` |
| `c7fe28c4` | Markdown | 100 | `0.6786` | `0.5980` | `0.4377` | `1.0000` | `0.3975` | `0.4645` | `1011.9` |

### Fair Overlap Comparison

- The JSON wrapper stalled after saving `99` completed rows, so the clean comparison is the intersection of those `99` example IDs.
- On the shared `99` rows:
  - JSON avg reward: `0.5858`
  - Markdown avg reward: `0.6736`
  - JSON point value score: `0.3082`
  - Markdown point value score: `0.4288`

### Main Findings

- This dataset is meaningfully harder than the original v1. The Qwen3-VL 8B baseline is far from saturated, so there is clear room for RL.
- Dense line reconstruction is the main bottleneck. The weakest cases are long noisy line charts with roughly `260` to `474` points.
- JSON underperforms Markdown on this harder dataset. The biggest gap is not format compliance alone; Markdown also improves point count recovery and x-position recovery.
- JSON had three fully invalid outputs in the saved run and several near-failure cases with almost zero point overlap. Markdown did not show the same format-failure pattern on the finished run.
- The best examples in both modes still reached nearly perfect reward, so the task is not impossible for the base model; performance is highly example-dependent.

### Failure Themes

- Dense line undercounting: the model often outputs a coarse subset of points instead of the full series, especially on `300+` point charts.
- Sparse x reconstruction: on weak dense-line examples, predicted x values jump in steps instead of matching the per-point index sequence.
- JSON formatting brittleness: a few dense cases failed to parse at all, which directly zeroed the reward.
- Hardest saved examples were dense line charts such as:
  - `seasonal`, `474` points, `large_wide`, `grid`
  - `noisy_trend_down`, `350` points, `medium_wide`, `ml_dense`
  - `loss_decay`, `344` points, `medium_wide`, `ml_dense`
  - `random_walk`, `334` points, `xl_wide`, `business_grid`

### Recommendation

- Stop before RL for now. Baseline headroom is confirmed.
- If RL starts on this dataset, use the Markdown target first, not JSON.
- Keep the validation subset at `100` rows for training-time eval, but retain the full `1000` row validation/test splits for offline analysis.
