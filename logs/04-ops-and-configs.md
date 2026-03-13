# Operations And Configs Log

## 2026-03-08 18:55:00 EDT

### Goal

- make baseline evaluation and inspection repeatable from repo-local scripts
- add the missing local endpoint alias for `qwen3-vl-8b`
- prepare initial hosted RL configs for `json` and `markdown` output modes

### Notes

- 2026-03-08 18:50:30 EDT: pushed the environment to Prime as `13point5/chart-extraction`
- 2026-03-08 18:52:10 EDT: Prime integration action `g5l43t8gw6lyyq2yga3q1dzn` completed with `SUCCESS`
- 2026-03-08 18:55:00 EDT: started adding repo-local scripts for baselines, eval summaries, and dataset inspection
- 2026-03-08 18:55:00 EDT: initial hosted RL configs target `Qwen/Qwen3-VL-8B-Instruct` with large-batch settings (`batch_size=512`, `rollouts_per_example=16`)
- 2026-03-08 19:01:30 EDT: generated and pushed the corrected synthetic dataset to Hugging Face as `13point5/chart-extraction-synth-v1`
- 2026-03-08 19:03:10 EDT: found that `prime eval run` did not resolve the new endpoint alias in this workflow and instead treated it as a raw model id
- 2026-03-08 19:03:10 EDT: changed the baseline runner default to the full model id `qwen/qwen3-vl-8b-instruct` to keep the script reliable
- 2026-03-08 19:04:40 EDT: ran upstream baselines against `13point5/chart-extraction` with the HF dataset `13point5/chart-extraction-synth-v1`
- 2026-03-08 19:04:40 EDT: upstream `json` baseline reached `avg_reward=0.996` with `output_tokens=142.75`
- 2026-03-08 19:04:40 EDT: upstream `markdown` baseline reached `avg_reward=0.996` with `output_tokens=84.75`
- 2026-03-08 19:04:40 EDT: remote eval startup still showed repeated environment health-check timeouts before the env server became healthy, but both runs completed successfully
