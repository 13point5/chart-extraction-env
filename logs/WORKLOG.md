# Work Log

## 2026-03-08

### Branch

- `wt/chart-extraction-baselines`

### Goal

- build a typed synthetic dataset generator for line and bar charts
- build a single-turn Verifiers environment around that dataset
- establish at least one baseline evaluation and compare prompt/output strategies if time permits
- keep the work split into a small number of reviewable commits

### Current Constraints

- Prime CLI is installed and authenticated via local CLI state
- Hugging Face CLI is installed and authenticated via local CLI state
- common API keys are not present in the current shell environment
- initial work should therefore focus on local generation, installability, and smoke tests, then attempt Hub upload and Prime eval through the authenticated CLIs

### Planned Commit Structure

1. Typed dataset generator package and preview workflow
2. Single-turn environment with JSON and Markdown extraction modes
3. Baseline evaluation configs, smoke runs, and experiment logs

### Progress

- created and pushed the dataset strategy document
- moved the strategy document to the repo root
- created working branch for implementation
- inspected local auth state and confirmed CLI-level auth for Prime and Hugging Face
- started implementing the typed dataset generator package
- added a module-based dataset generator CLI at `python -m chart_extraction_env.dataset_cli`
- added strong typed models for latent chart specs, canonical answers, render plans, and generated examples
- smoke-generated a local Hugging Face dataset to `environments/chart_extraction/outputs/chart_extraction/smoke_dataset_v2`
- confirmed row shape is `id`, `image`, `answer`, `info`
- tightened sampling to guarantee balanced chart-type, style, and size coverage even on tiny smoke runs

### Smoke Notes

- train smoke split produced a `6 / 6` line-to-bar mix on 12 examples
- style profiles were evenly represented on the 12-example smoke split
- annotation frequency stayed sparse, which matches the near-term plan
- preview bundle is generated as `preview.md` plus `preview_images/` for manual inspection
