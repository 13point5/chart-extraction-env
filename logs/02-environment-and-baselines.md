# Environment And Baselines Log

## 2026-03-08 18:34:00 EDT

### Goal

- build a single-turn Verifiers environment around the synthetic chart dataset
- support at least two extraction modes for comparison: `json` and `markdown`
- establish an initial baseline with a vision model available through Prime

### Environment Notes

- 2026-03-08 18:34:00 EDT: implemented a single-turn environment with local, Hugging Face, and in-memory dataset loading paths
- 2026-03-08 18:34:30 EDT: added prompt builders for `json` and `markdown` output modes
- 2026-03-08 18:35:00 EDT: added a parser for both modes
- 2026-03-08 18:38:00 EDT: added zero-weight rubric metrics for diagnosis, not just the weighted reward

### Baseline Observations

- 2026-03-08 18:36:04 EDT: first `json` smoke run on `qwen/qwen3-vl-8b-instruct` started end to end through Prime eval
- 2026-03-08 18:36:12 EDT: eval reached real rollouts with multimodal prompts; warnings were noisy but not fatal
- 2026-03-08 18:36:20 EDT: observed reward around `0.846` before the schema fix
- 2026-03-08 18:36:20 EDT: before fixing the target schema, `series_name_score` collapsed to zero because the single-series name was often not visible in the chart itself
- 2026-03-08 18:37:00 EDT: treated that as a reward-design bug, not a model failure
- 2026-03-08 18:37:00 EDT: fixed the near-term single-series target to always use `series_1`
- 2026-03-08 18:39:56 EDT: reran the `json` baseline after the schema fix and reached `avg_reward=0.9949` on 4 validation examples
- 2026-03-08 18:42:15 EDT: ran the matching `markdown` baseline on the same model and reached `avg_reward=0.9949`
- 2026-03-08 18:42:15 EDT: markdown used materially fewer output tokens than json on the same slice (`81.5` vs `134.75` average output tokens)
- 2026-03-08 18:44:50 EDT: reran a 2-example `json` smoke eval after the output-mode typing cleanup and the environment remained stable at `avg_reward=0.994`

### Reward Hacking Watch

- 2026-03-08 18:36:20 EDT: failure mode observed: the model can produce very strong point values while still missing latent fields that are not observable from the image
- 2026-03-08 18:37:00 EDT: action taken: remove invisible supervision targets from the near-term task contract
- 2026-03-08 18:38:20 EDT: next thing to watch: whether the model learns to always emit `line` or a generic numeric sequence when uncertainty is high
- 2026-03-08 18:42:15 EDT: markdown currently looks viable as a cheaper output format, but it may still be easier to reward-hack with malformed tables once charts get harder

### Test Coverage

- 2026-03-08 18:48:20 EDT: added fast unit tests for parser behavior, reward components, and the single-series `series_1` target contract
- 2026-03-08 18:48:40 EDT: test command `PYTHONPATH=environments/chart_extraction python -m unittest discover -s environments/chart_extraction/tests -p 'test_*.py' -v` passed with 5/5 tests
- 2026-03-08 18:48:40 EDT: avoided the host `pytest` install after repeated segmentation faults in the workstation Python stack; `unittest` is stable here

### Pending

- 2026-03-08 18:49:00 EDT: commit the environment, baseline comparison, and tests as one logical unit
- 2026-03-08 18:49:00 EDT: push the environment to Prime and inspect action status/logs before starting RL
