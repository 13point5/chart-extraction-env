# Chart Extraction Summary

As of March 11, 2026, this project has a working synthetic dataset generator, a single-turn Verifiers environment, baseline eval results, and completed hosted RL runs.

## What Was Built


| Area                                      | Status | Main artifacts                                                      |
| ----------------------------------------- | ------ | ------------------------------------------------------------------- |
| Dataset strategy                          | Done   | `DATASET_GENERATION_STRATEGY.md`                                    |
| Typed synthetic dataset generator         | Done   | `environments/chart_extraction/chart_extraction_env/dataset/`       |
| Hugging Face dataset packaging and upload | Done   | HF dataset `13point5/chart-extraction-synth-v1`                     |
| Single-turn chart extraction environment  | Done   | `environments/chart_extraction/chart_extraction_env/environment.py` |
| JSON and Markdown output modes            | Done   | `prompts.py`, `parsing.py`, rubric in `environment.py`              |
| Baseline eval utilities                   | Done   | `scripts/chart_extraction/run_baselines.py`, `summarize_eval.py`    |
| Dataset inspection utility                | Done   | `scripts/chart_extraction/inspect_dataset.py`                       |
| Unit tests for parsing and reward logic   | Done   | `environments/chart_extraction/tests/`                              |
| Hosted RL configs                         | Done   | `configs/rl/chart-extraction-*.toml`                                |
| Hosted RL runs                            | Done   | JSON and Markdown fallback runs on `Qwen/Qwen3-VL-4B-Instruct`      |


## Dataset

### Published Dataset


| Item                                     | Value                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| Hugging Face dataset                     | `13point5/chart-extraction-synth-v1`                                        |
| Local output path used during generation | `environments/chart_extraction/outputs/chart_extraction/dataset_v1_series1` |
| Dataset format                           | Hugging Face `DatasetDict`                                                  |
| Splits                                   | `train`, `validation`, `test`                                               |
| Row columns                              | `id`, `image`, `answer`, `info`                                             |
| Series naming contract                   | Always `series_1` for v1                                                    |


### Row Schema


| Column   | Type      | Meaning                              |
| -------- | --------- | ------------------------------------ |
| `id`     | `string`  | Stable example id like `train-00000` |
| `image`  | `Image()` | Rendered chart PNG                   |
| `answer` | `string`  | Canonical JSON extraction target     |
| `info`   | `string`  | Canonical JSON generator metadata    |


### Implemented Dataset Scope


| Axis                 | Implemented values                                                                      | Notes                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `chart_type`         | `line`, `bar`                                                                           | Single-series only                                                                       |
| `x_mode`             | `numeric_even`, `categorical_ordered`                                                   | Numeric for line, categorical for bar                                                    |
| `data_profile`       | `mono_up`, `mono_down`, `single_peak`, `single_valley`, `zigzag`, `flat`, `random_walk` | Controls the underlying shape                                                            |
| `label_profile`      | `short`, `medium`, `long`                                                               | Affects category labels                                                                  |
| `style_profile`      | `clean`, `grid`, `markers`, `grid_markers`                                              | Markers only affect line charts                                                          |
| `annotation_profile` | `none`, `endpoint_labels`                                                               | Sparse annotation variation only                                                         |
| `layout_profile`     | `square`, `wide`, `tall`                                                                | `tall` is defined in the model but not materially exercised in the published dataset mix |
| `image_size_profile` | `small_square`, `medium_square`, `medium_wide`, `large_wide`                            | First-class diversity axis                                                               |
| `y_scale_profile`    | `near_flat`, `small_range`, `medium_range`, `large_range`                               | Controls dynamic range                                                                   |
| `num_points`         | 4-12 for line, 4-10 for bar, then capped by labels/annotations/size                     | Long labels and small images reduce the max                                              |


### Split Sizes


| Split        | Rows  |
| ------------ | ----- |
| `train`      | 2,048 |
| `validation` | 256   |
| `test`       | 256   |
| Total        | 2,560 |


### Observed Split Coverage

These are the observed counts in the published dataset.


| Feature              | Train                                                                                                             | Validation                                                               | Test                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------- |
| `chart_type`         | `line` 1024, `bar` 1024                                                                                           | `line` 128, `bar` 128                                                    | `line` 128, `bar` 128 |
| `data_profile`       | 293 each for `mono_up`, `mono_down`, `single_peak`, `single_valley`; 292 each for `zigzag`, `flat`, `random_walk` | 37 each for first four; 36 each for last three                           | same as validation    |
| `style_profile`      | 512 each for `clean`, `grid`, `markers`, `grid_markers`                                                           | 64 each                                                                  | 64 each               |
| `label_profile`      | `short` 512, `medium` 1024, `long` 512                                                                            | `short` 64, `medium` 128, `long` 64                                      | same as validation    |
| `image_size_profile` | `small_square` 342, `medium_square` 341, `medium_wide` 682, `large_wide` 683                                      | `small_square` 43, `medium_square` 43, `medium_wide` 85, `large_wide` 85 | same as validation    |


### Important Dataset Design Decisions


| Decision                                       | Why                                                        |
| ---------------------------------------------- | ---------------------------------------------------------- |
| Single-series only in v1                       | Keeps reward failures interpretable                        |
| `series_1` fixed as the only series name       | Avoids rewarding text that may not be visible in the image |
| Programmatic chart generation only             | Exact ground truth without LLM noise                       |
| Canonical JSON strings for `answer` and `info` | Stable reward parsing and HF-friendly upload               |


## Environment And Tests


| Item                             | Result                                              |
| -------------------------------- | --------------------------------------------------- |
| Prime environment slug           | `13point5/chart-extraction`                         |
| Prime integration action on push | Success                                             |
| Output modes                     | `json`, `markdown`                                  |
| Rubric                           | Weighted reward plus zero-weight diagnostic metrics |
| Unit tests                       | 5/5 passing with `unittest`                         |


### Reward Components


| Metric                      | Type        | Purpose                                |
| --------------------------- | ----------- | -------------------------------------- |
| `format_valid`              | weighted    | response parses in the selected format |
| `chart_type_score`          | weighted    | exact chart type match                 |
| `series_name_score`         | weighted    | exact normalized series-name match     |
| `point_count_score`         | weighted    | predicted vs target count overlap      |
| `point_value_score`         | weighted    | combined x/y point accuracy            |
| `x_type_score_metric`       | zero-weight | diagnostic metric                      |
| `point_x_score_metric`      | zero-weight | diagnostic metric                      |
| `point_y_score_metric`      | zero-weight | diagnostic metric                      |
| `exact_answer_score_metric` | zero-weight | exact structured match                 |
| `normalized_y_mae_metric`   | zero-weight | normalized y error                     |


## Baseline Evaluations

### Key Baseline Story

The first local JSON smoke run looked much worse than it should have because the task rewarded a single-series name that was often not visible in the chart. After fixing the target contract to always use `series_1`, both JSON and Markdown baselines became very strong.

### Main Baseline Results


| Run ID     | Model                       | Env source | Dataset source     | Mode     | Examples | Avg reward | Point value | X type | Avg output tokens | Notes                              |
| ---------- | --------------------------- | ---------- | ------------------ | -------- | -------- | ---------- | ----------- | ------ | ----------------- | ---------------------------------- |
| `70c7e525` | `qwen/qwen3-vl-8b-instruct` | local      | in-memory fallback | JSON     | 3        | 0.8456     | 0.9911      | 0.0000 | 149.3             | pre-fix, invisible series-name bug |
| `5484796c` | `qwen/qwen3-vl-8b-instruct` | local      | in-memory fallback | JSON     | 4        | 0.9949     | 0.9899      | 1.0000 | 134.8             | post-fix                           |
| `756c8503` | `qwen/qwen3-vl-8b-instruct` | local      | in-memory fallback | Markdown | 4        | 0.9949     | 0.9898      | 1.0000 | 81.5              | post-fix                           |
| `dd909236` | `qwen/qwen3-vl-8b-instruct` | local      | in-memory fallback | JSON     | 2        | 0.9944     | 0.9889      | 1.0000 | 143.5             | post-typing cleanup                |
| `539bc21e` | `qwen/qwen3-vl-8b-instruct` | pushed env | HF dataset         | JSON     | 8        | 0.9963     | 0.9926      | 1.0000 | 142.8             | upstream path                      |
| `2503bf4b` | `qwen/qwen3-vl-8b-instruct` | pushed env | HF dataset         | Markdown | 8        | 0.9964     | 0.9928      | 1.0000 | 84.8              | upstream path                      |


### Baseline Takeaways


| Finding                                                                   | Interpretation                                          |
| ------------------------------------------------------------------------- | ------------------------------------------------------- |
| JSON and Markdown both solve the current narrow task very well            | The task is learnable with the current synthetic setup  |
| Markdown uses far fewer output tokens                                     | It is cheaper to emit                                   |
| Exact structured match stays at 0 while point-value accuracy is very high | The model is approximately right, not exactly canonical |
| The major early failure was reward design, not model capability           | Invisible supervision targets were the main issue       |


## RL Experiments

### Model Availability Constraint

The original goal was to use `Qwen3-VL 8B`, but hosted Prime RL did not actually expose `Qwen/Qwen3-VL-8B-Instruct` at run creation time. The fallback hosted runs used `Qwen/Qwen3-VL-4B-Instruct`.

### Hosted RL Runs


| Mode     | Run ID                     | Model                       | Status    | Started             | Completed           | Duration   |
| -------- | -------------------------- | --------------------------- | --------- | ------------------- | ------------------- | ---------- |
| JSON     | `gqxe480i8tte8n0d7c01mfm9` | `Qwen/Qwen3-VL-4B-Instruct` | Completed | 2026-03-08 23:01:46 | 2026-03-09 01:24:44 | 2h 22m 58s |
| Markdown | `wy876rtey6v27zbnf4eqmj0x` | `Qwen/Qwen3-VL-4B-Instruct` | Completed | 2026-03-08 23:01:47 | 2026-03-09 00:31:39 | 1h 29m 51s |


### RL Configuration


| Setting                | Value          |
| ---------------------- | -------------- |
| `max_steps`            | 60             |
| `batch_size`           | 512            |
| `rollouts_per_example` | 16             |
| `max_tokens`           | 512            |
| Validation interval    | every 10 steps |
| Validation examples    | 32             |


### RL Outcome Summary


| Mode     | Metric rows | First step reward | First step val reward | Best reward step | Best reward | Best val step | Best val reward | Last step | Last reward |
| -------- | ----------- | ----------------- | --------------------- | ---------------- | ----------- | ------------- | --------------- | --------- | ----------- |
| JSON     | 60          | 0.9893            | 0.9722                | 25               | 0.9952      | 30            | 0.9918          | 59        | 0.9849      |
| Markdown | 59          | 0.9731            | 0.9768                | 39               | 0.9953      | 40            | 0.9888          | 59        | 0.9882      |


### Final Observed Point Metrics


| Mode     | Last `point_value_score` | Last `point_count_score` | Last `point_x_score_metric` | Last `x_type_score_metric` |
| -------- | ------------------------ | ------------------------ | --------------------------- | -------------------------- |
| JSON     | 0.9727                   | 0.9855                   | 0.9837                      | 1.0000                     |
| Markdown | 0.9785                   | 0.9894                   | 0.9832                      | 1.0000                     |


### RL Interpretation


| Finding                                         | Interpretation                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------- |
| JSON started stronger than Markdown             | Better early training signal on this task                                    |
| JSON achieved the better best validation reward | Better near-term generalization signal: `0.9918` vs `0.9888`                 |
| Markdown later caught up on training reward     | It can optimize the reward, but that did not clearly beat JSON on validation |
| Both runs completed cleanly                     | The environment, dataset, and hosted RL path all worked end to end           |


## Reward-Hacking And Failure Modes

### Observed Failure Modes


| Mode                  | Failure mode                                                             | Effect on reward                                                     |
| --------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| JSON                  | Occasionally duplicates a category and effectively drops a point         | Hurts point count and x alignment, but the structure stays parseable |
| Markdown              | Sometimes invents intermediate numeric x values like `0.5`, `1.5`, `2.5` | Keeps format valid while damaging point count and x-value accuracy   |
| Pre-fix task contract | Required a latent series name that was not visible in the image          | Artificially crushed reward without reflecting model quality         |


### Current Judgment

For this v1 task, JSON is the better default training target. Markdown is viable, but it leaves more room for silently plausible table outputs that are structurally valid and semantically wrong.

## Open Issues


| Issue                                                          | Status                                           |
| -------------------------------------------------------------- | ------------------------------------------------ |
| Hosted RL availability for `Qwen3-VL 8B`                       | Not available at run creation time               |
| Dataset scope is still narrow                                  | Only single-series line and bar are implemented  |
| Exact canonical answer matching is still weak                  | Models are close but not byte-for-byte canonical |
| Multimodal prompt serialization warnings appear in local evals | Non-fatal, but still noisy                       |


## Recommended Next Steps

1. Keep JSON as the default RL target for the next round.
2. Add held-out evaluation slices that are harder than the current train distribution.
3. Expand the dataset to scatter and then multi-series line/grouped bar.
4. Add more dataset-level validation and analysis beyond simple coverage counts.
5. Revisit hosted RL on 8B VL only if the model becomes actually available in the live RL backend.

## Pointers


| Topic              | File                                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Dataset strategy   | `DATASET_GENERATION_STRATEGY.md`                                                                                                  |
| Dataset models     | `environments/chart_extraction/chart_extraction_env/dataset/models.py`                                                            |
| Sampling logic     | `environments/chart_extraction/chart_extraction_env/dataset/sampling.py`                                                          |
| Generator          | `environments/chart_extraction/chart_extraction_env/dataset/generator.py`                                                         |
| Renderer           | `environments/chart_extraction/chart_extraction_env/dataset/render.py`                                                            |
| HF packaging       | `environments/chart_extraction/chart_extraction_env/dataset/hf.py`                                                                |
| Dataset CLI        | `environments/chart_extraction/chart_extraction_env/dataset_cli.py`                                                               |
| Dataset inspection | `scripts/chart_extraction/inspect_dataset.py`                                                                                     |
| Work logs          | `logs/01-dataset-generator.md`, `logs/02-environment-and-baselines.md`, `logs/03-rl-experiments.md`, `logs/04-ops-and-configs.md` |

