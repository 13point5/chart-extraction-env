# Markdown RL Rerun

- 2026-03-12 00:29:00 EDT: checked the previous dense line-only Markdown RL run `nmnb0uqwu2gl1trjhj2r9nb6`.
- 2026-03-12 00:29:00 EDT: the run failed with `BackoffLimitExceeded: Job has reached the specified backoff limit`.
- 2026-03-12 00:29:00 EDT: the matching JSON run `jp2xfslsy7io92dz6idgozst` on the same environment version and dataset slice was still running, so the failure signature points to trainer infrastructure rather than an environment or dataset issue.
- 2026-03-12 00:30:00 EDT: relaunching the same Markdown config against `13point5/chart-extraction@0.1.1` and `13point5/chart-extraction-dense-noisy-v1` with `chart_type_filter="line"`.
