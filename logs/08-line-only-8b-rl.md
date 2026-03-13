# Line-Only 8B RL Relaunch

- 2026-03-11 22:10:00 EDT: started a new RL prep pass for the dense dataset after the dataset was pushed to Hugging Face.
- 2026-03-11 22:10:00 EDT: goal is to run two hosted RL jobs on `Qwen/Qwen3-VL-8B-Instruct`, one with JSON outputs and one with Markdown outputs, both restricted to line charts from `13point5/chart-extraction-dense-noisy-v1`.
- 2026-03-11 22:10:00 EDT: user-requested training settings are `rollouts_per_example=8`, validation every `25` steps, and `300` total steps.
- 2026-03-11 22:16:00 EDT: added `chart_type_filter` support to the environment loader and dataset loader so hosted evals and RL can be restricted to line charts without generating a separate dataset repo.
- 2026-03-11 22:17:00 EDT: updated the chart extraction RL configs to use `13point5/chart-extraction-dense-noisy-v1`, `chart_type_filter="line"`, `rollouts_per_example=8`, `batch_size=128`, validation every `25` steps, and `300` training steps.
- 2026-03-11 22:18:00 EDT: added `test_dataset_loading.py` to verify that `chart_type_filter="line"` returns only line-chart rows from a saved local dataset.
- 2026-03-11 22:33:00 EDT: local markdown smoke eval on 5 examples succeeded against the Hugging Face dataset with `output_mode="markdown"`. Average reward was `0.740`; `format_valid`, `chart_type_score`, and `x_type_score_metric` were perfect on the sample; `is_truncated` averaged `0.400`, so I kept `max_tokens=4096` and did not lower it.
- 2026-03-11 22:41:00 EDT: local JSON smoke eval on the same line-only setup succeeded with `output_mode="json"`. Average reward was `0.542`; `is_truncated` averaged `0.000`; outputs were materially coarser on dense lines than the markdown baseline.
- 2026-03-11 22:41:00 EDT: takeaway before RL is unchanged: markdown is the stronger starting target on dense line charts, but JSON remains a useful comparator because it is faster and less truncation-prone.
- 2026-03-11 22:41:48 EDT: pushed updated environment version `13point5/chart-extraction@0.1.1` with line-only dataset filtering support.
- 2026-03-11 22:42:34 EDT: checked the Prime environment action status for `0.1.1`; integration job `dklpdqup1i8mxwg8gyk93488` completed with `SUCCESS`.
- 2026-03-11 22:43:29 EDT: launched hosted RL run `jp2xfslsy7io92dz6idgozst` for the JSON config (`qwen3-vl-8b-json-dense-line-v1`).
- 2026-03-11 22:43:29 EDT: launched hosted RL run `nmnb0uqwu2gl1trjhj2r9nb6` for the Markdown config (`qwen3-vl-8b-markdown-dense-line-v1`).
- 2026-03-11 22:43:29 EDT: both runs were accepted by hosted RL against `13point5/chart-extraction@0.1.1` and entered the `QUEUED` state with the expected line-only environment args and validation schedule.
- 2026-03-11 22:43:35 EDT: both hosted RL runs transitioned from `QUEUED` to `RUNNING`, confirming that the new environment version and config payloads were accepted by the trainer backend.
