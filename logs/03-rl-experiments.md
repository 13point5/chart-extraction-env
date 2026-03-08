# RL Experiments Log

## 2026-03-08 18:38:40 EDT

### Status

- 2026-03-08 18:38:40 EDT: not started yet
- 2026-03-08 19:06:30 EDT: first hosted RL launch attempt reached config validation and failed before run creation because `wandb.entity` was missing
- 2026-03-08 19:06:30 EDT: action taken: add a required W&B entity field to both hosted RL configs
- 2026-03-08 19:08:40 EDT: second hosted RL launch attempt failed because `Qwen/Qwen3-VL-8B-Instruct` is not actually available in the live hosted RL backend
- 2026-03-08 19:08:40 EDT: observed discrepancy: `prime rl models` now lists `Qwen/Qwen3-VL-4B-Instruct`, not the 8B VL model
- 2026-03-08 19:08:40 EDT: action taken: keep the 8B configs for intent tracking, switch W&B entity to `13point5-labs`, and add 4B VL fallback configs so hosted RL experiments can proceed
- 2026-03-08 19:11:45 EDT: launched hosted JSON RL run `gqxe480i8tte8n0d7c01mfm9` on `Qwen/Qwen3-VL-4B-Instruct`
- 2026-03-08 19:11:55 EDT: launched hosted Markdown RL run `wy876rtey6v27zbnf4eqmj0x` on `Qwen/Qwen3-VL-4B-Instruct`
- 2026-03-08 19:13:15 EDT: both runs passed environment install, W&B initialization, and training-environment startup on the cluster
- 2026-03-08 19:13:55 EDT: JSON step 0 reached `reward=0.9893`, `val_reward=0.9722`, `point_value_score=0.9811`, `point_count_score=0.9872`
- 2026-03-08 19:14:30 EDT: JSON step 1 reached `reward=0.9797`, `point_value_score=0.9624`, `point_x_score_metric=0.9636`
- 2026-03-08 19:14:10 EDT: Markdown step 0 reached `reward=0.9731`, `val_reward=0.9768`, `point_value_score=0.9520`, `point_count_score=0.9711`
- 2026-03-08 19:14:40 EDT: Markdown step 1 reached `reward=0.9577`, `point_value_score=0.9231`, `point_x_score_metric=0.9036`

### Planned Scope

- 2026-03-08 18:38:40 EDT: start only after the environment and baseline evals are stable
- 2026-03-08 18:38:40 EDT: keep at most two active experiments at a time
- 2026-03-08 18:38:40 EDT: track reward hacking observations per run
- 2026-03-08 18:38:40 EDT: use repo-local configs and scripts for launch, inspection, and early stopping

### Reward Hacking Watch

- 2026-03-08 19:14:50 EDT: JSON rollout samples mostly preserve the point schema, but one low-reward sample duplicated a category and effectively dropped a point
- 2026-03-08 19:14:50 EDT: Markdown rollout samples already show a clearer failure mode: the model sometimes invents intermediate x values such as `0.5`, `1.5`, and `2.5` while still producing valid Markdown
- 2026-03-08 19:14:50 EDT: current interpretation: JSON is the stronger near-term RL target because its structure leaves less room for silently plausible but wrong point grids
