# Dataset Generator Log

## 2026-03-08 18:31:30 EDT

### Goal

- build a typed synthetic dataset generator for single-series line and bar charts
- save Hugging Face-compatible splits with `id`, `image`, `answer`, and `info`
- make small preview bundles easy to inspect before moving to baselines

### Implementation Notes

- 2026-03-08 18:26:00 EDT: added a module-based generator CLI: `python -m chart_extraction_env.dataset_cli`
- 2026-03-08 18:28:00 EDT: added strong typed models for latent specs, render plans, answers, and generated examples
- 2026-03-08 18:29:00 EDT: added Hugging Face-compatible save path via `datasets.DatasetDict.save_to_disk`
- 2026-03-08 18:30:00 EDT: added a preview bundle writer that emits `preview.md` plus preview PNGs

### Smoke Checks

- 2026-03-08 18:30:30 EDT: local smoke dataset written to `environments/chart_extraction/outputs/chart_extraction/smoke_dataset_v2`
- 2026-03-08 18:30:45 EDT: row shape confirmed as `id`, `image`, `answer`, `info`
- 2026-03-08 18:31:00 EDT: small smoke split showed balanced line/bar coverage and even style coverage

### Notes

- 2026-03-08 18:31:15 EDT: current generator is intentionally limited to single-series charts
- 2026-03-08 18:37:00 EDT: series names were fixed to `series_1` for v1 so the target never depends on invisible legend text
