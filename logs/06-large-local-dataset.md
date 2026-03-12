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
