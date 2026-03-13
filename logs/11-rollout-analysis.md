# Rollout Analysis

## 2026-03-12 10:26:30 EDT

- Goal: inspect Prime RL rollout samples for the completed dense line-only JSON and Markdown runs and quantify where the current point-count and point-value rewards under-measure mistakes.
- Runs analyzed:
  - `jp2xfslsy7io92dz6idgozst` (`json`)
  - `admofyilspcdijc5s6ve0bu2` (`markdown`)
- Observation from Prime CLI: each saved checkpoint only exposed `8` rollout samples, which corresponds to one prompt with `8` rollouts. Switched from single-step analysis to aggregating all `29` shared saved steps to get useful coverage.
- Implementation:
  - added `/Users/13point5/projects/chart-extraction-env/scripts/chart_extraction/analyze_rl_rollouts.py`
  - reused environment parsing/reward code where possible:
    - `extract_assistant_text`
    - `parse_response`
    - `point_component_scores`
  - added a lenient wrapper for Prime rollout payloads because `prime rl rollouts` inserts raw line wraps inside embedded JSON strings and numbers
  - normalized rollout answers against the dataset target JSON before matching back to local images
- Output:
  - report: `/Users/13point5/projects/chart-extraction-env/reports/chart_extraction_rollout_analysis_all_steps/REPORT.md`
  - assets: `/Users/13point5/projects/chart-extraction-env/reports/chart_extraction_rollout_analysis_all_steps/assets`
- Key findings:
  - JSON run aggregate over saved rollouts: mean reward `0.7784`, mean missing points `67.62`, parse fail rate `3.45%`, materially wrong rate `91.38%`
  - Markdown run aggregate over saved rollouts: mean reward `0.9471`, mean missing points `26.86`, materially wrong rate `95.26%`, high-reward-wrong rate `88.79%`
  - Markdown is much more likely to receive high reward despite omitting `20-40+` points or having large y-magnitude distortions on `100+` matched points
  - JSON fails harder and more often, but its lower rewards track those failures more honestly than Markdown
- Reward weakness called out in the report:
  - `point_count_score = min(pred_count, target_count) / max(pred_count, target_count)`
  - `point_value_score = mean(0.4 * x_score + 0.6 * y_score) * coverage`
  - those formulas compress missing tails and can still reward sparse subsampling or coarse reconstruction when the prefix alignment looks plausible
