#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "environments/chart_extraction/outputs/evals/chart-extraction--qwen--qwen3-vl-8b-instruct/dc09fd55"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/series-point-value-low-score-analysis.md"
DEFAULT_ASSET_DIR = REPO_ROOT / "docs/assets/series-point-value-low-score"
DEFAULT_EXAMPLE_IDS = [1228, 255, 101, 1024, 1018]
ANSWER_RE = re.compile(r"<answer>\s*(\{.*\})\s*</answer>", re.DOTALL)


@dataclass
class ExampleDiagnostics:
    example_id: int
    file_name: str
    reward: float
    series_name_f1: float
    series_point_count_ratio: float
    series_point_value: float
    order_aligned_y_score: float
    mean_x_step_ratio: float | None
    exact_x_match_fraction: float
    title: str
    x_axis_label: str
    y_axis_label: str
    image_path: Path
    gold_series: list[dict]
    predicted_series: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown report for very low series_point_value cases.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Eval run directory containing metadata.json and results.jsonl.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory where selected chart images will be written.",
    )
    parser.add_argument(
        "--example-id",
        action="append",
        type=int,
        dest="example_ids",
        help="Specific example id to include. Can be supplied multiple times.",
    )
    return parser.parse_args()


def load_metadata(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def load_results(run_dir: Path) -> list[dict]:
    results_path = run_dir / "results.jsonl"
    with results_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def parse_answer(raw_completion: str) -> dict:
    match = ANSWER_RE.search(raw_completion)
    if match is None:
        raise ValueError("Completion is missing an <answer> JSON payload")
    return json.loads(match.group(1))


def sort_points(points: list[list[float]]) -> list[tuple[float, float]]:
    return sorted(
        [(float(point[0]), float(point[1])) for point in points if len(point) == 2],
        key=lambda point: point[0],
    )


def order_aligned_y_score(
    predicted_points: list[list[float]],
    gold_points: list[list[float]],
) -> float:
    predicted = sort_points(predicted_points)
    gold = sort_points(gold_points)

    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0

    gold_ys = [gold_y for _, gold_y in gold]
    y_scale = max(max(gold_ys) - min(gold_ys), 1.0)
    matched_count = min(len(predicted), len(gold))

    total_score = 0.0
    for index in range(matched_count):
        _, predicted_y = predicted[index]
        _, gold_y = gold[index]
        total_score += max(0.0, 1.0 - abs(predicted_y - gold_y) / y_scale)

    return total_score / len(gold)


def mean_x_step_ratio(
    predicted_points: list[list[float]],
    gold_points: list[list[float]],
) -> float | None:
    predicted = sort_points(predicted_points)
    gold = sort_points(gold_points)

    if len(predicted) != len(gold) or len(gold) < 2:
        return None

    gold_steps = [
        abs(gold[index + 1][0] - gold[index][0])
        for index in range(len(gold) - 1)
        if gold[index + 1][0] != gold[index][0]
    ]
    if not gold_steps:
        return None

    reference_step = statistics.median(gold_steps)
    mean_abs_x_offset = statistics.mean(
        abs(predicted[index][0] - gold[index][0])
        for index in range(len(gold))
    )
    return mean_abs_x_offset / reference_step


def exact_x_match_fraction(
    predicted_points: list[list[float]],
    gold_points: list[list[float]],
) -> float:
    predicted_xs = {float(point[0]) for point in predicted_points if len(point) == 2}
    gold_xs = [float(point[0]) for point in gold_points if len(point) == 2]

    if not gold_xs:
        return 1.0 if not predicted_xs else 0.0

    exact_matches = sum(1 for gold_x in gold_xs if gold_x in predicted_xs)
    return exact_matches / len(gold_xs)


def weighted_example_diagnostics(
    result_row: dict,
    predicted_answer: dict,
) -> tuple[float, float | None, float]:
    predicted_by_name = {
        series["name"]: series.get("points", [])
        for series in predicted_answer.get("series", [])
        if series.get("name")
    }
    gold_by_name = {
        series["name"]: series.get("points", [])
        for series in result_row["info"].get("series", [])
        if series.get("name")
    }

    total_weight = 0
    weighted_y_score = 0.0
    weighted_exact_fraction = 0.0
    x_step_ratios: list[float] = []

    for name, gold_points in gold_by_name.items():
        predicted_points = predicted_by_name.get(name, [])
        weight = max(len(gold_points), 1)
        weighted_y_score += order_aligned_y_score(predicted_points, gold_points) * weight
        weighted_exact_fraction += exact_x_match_fraction(predicted_points, gold_points) * weight
        total_weight += weight

        ratio = mean_x_step_ratio(predicted_points, gold_points)
        if ratio is not None:
            x_step_ratios.append(ratio)

    return (
        weighted_y_score / total_weight if total_weight else 0.0,
        statistics.mean(x_step_ratios) if x_step_ratios else None,
        weighted_exact_fraction / total_weight if total_weight else 0.0,
    )


def as_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_points(points: list[list[float]], limit: int = 6) -> str:
    return json.dumps(points[:limit], ensure_ascii=True)


def build_distribution_summary(results: list[dict]) -> dict[str, float | int]:
    def point_count_ratio_metric(row: dict) -> float:
        metrics = row["metrics"]
        raw_count_ratio = metrics.get("series_point_count_ratio_raw")
        if raw_count_ratio is not None:
            return float(raw_count_ratio)
        return float(metrics["series_point_count_ratio"])

    point_values = [row["metrics"]["series_point_value"] for row in results]
    zero_point_value = [row for row in results if row["metrics"]["series_point_value"] == 0.0]
    zero_and_perfect_names = [
        row for row in zero_point_value if row["metrics"]["series_name_f1"] == 1.0
    ]
    zero_perfect_names_high_count = [
        row
        for row in zero_point_value
        if row["metrics"]["series_name_f1"] == 1.0
        and point_count_ratio_metric(row) >= 0.8
    ]
    zero_perfect_names_perfect_count = [
        row
        for row in zero_point_value
        if row["metrics"]["series_name_f1"] == 1.0
        and point_count_ratio_metric(row) == 1.0
    ]

    return {
        "count": len(results),
        "avg_point_value": statistics.mean(point_values),
        "zero_point_value": len(zero_point_value),
        "lt_0_05": sum(1 for value in point_values if value < 0.05),
        "lt_0_1": sum(1 for value in point_values if value < 0.1),
        "gt_0_5": sum(1 for value in point_values if value > 0.5),
        "zero_and_perfect_names": len(zero_and_perfect_names),
        "zero_perfect_names_high_count": len(zero_perfect_names_high_count),
        "zero_perfect_names_perfect_count": len(zero_perfect_names_perfect_count),
    }


def build_strict_zero_case_summary(results: list[dict]) -> dict[str, float | int]:
    strict_rows = [
        row
        for row in results
        if row["metrics"]["series_point_value"] == 0.0
        and row["metrics"]["series_name_f1"] == 1.0
        and row["metrics"]["series_point_count_ratio"] == 1.0
    ]

    order_scores: list[float] = []
    x_step_ratios: list[float] = []
    strong_shape_cases = 0

    for row in strict_rows:
        predicted_answer = parse_answer(row["completion"][0]["content"])
        order_score, x_step_ratio, _ = weighted_example_diagnostics(row, predicted_answer)
        order_scores.append(order_score)
        if x_step_ratio is not None:
            x_step_ratios.append(x_step_ratio)
        if order_score >= 0.8 and x_step_ratio is not None and x_step_ratio <= 0.25:
            strong_shape_cases += 1

    return {
        "count": len(strict_rows),
        "mean_order_aligned_y_score": statistics.mean(order_scores),
        "median_order_aligned_y_score": statistics.median(order_scores),
        "mean_x_step_ratio": statistics.mean(x_step_ratios),
        "median_x_step_ratio": statistics.median(x_step_ratios),
        "strong_shape_cases": strong_shape_cases,
    }


def build_example_diagnostics(
    results_by_id: dict[int, dict],
    dataset_split,
    example_ids: list[int],
    asset_dir: Path,
) -> list[ExampleDiagnostics]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[ExampleDiagnostics] = []

    for example_id in example_ids:
        row = results_by_id[example_id]
        predicted_answer = parse_answer(row["completion"][0]["content"])
        order_score, x_ratio, exact_fraction = weighted_example_diagnostics(row, predicted_answer)

        dataset_row = dataset_split[example_id]
        image_path = asset_dir / row["info"]["file_name"]
        dataset_row["image"].save(image_path)

        diagnostics.append(
            ExampleDiagnostics(
                example_id=example_id,
                file_name=row["info"]["file_name"],
                reward=float(row["reward"]),
                series_name_f1=float(row["metrics"]["series_name_f1"]),
                series_point_count_ratio=float(
                    row["metrics"].get(
                        "series_point_count_ratio_raw",
                        row["metrics"]["series_point_count_ratio"],
                    )
                ),
                series_point_value=float(row["metrics"]["series_point_value"]),
                order_aligned_y_score=order_score,
                mean_x_step_ratio=x_ratio,
                exact_x_match_fraction=exact_fraction,
                title=row["info"]["title"],
                x_axis_label=row["info"]["x_axis_label"],
                y_axis_label=row["info"]["y_axis_label"],
                image_path=image_path,
                gold_series=row["info"]["series"],
                predicted_series=predicted_answer["series"],
            )
        )

    return diagnostics


def build_behavior_note(example: ExampleDiagnostics) -> str:
    if example.series_name_f1 < 1.0:
        return (
            "This looks like a genuine extraction miss: at least one series name was typoed or dropped, "
            "so the reward never even gets to compare those points."
        )

    if example.mean_x_step_ratio is not None and example.mean_x_step_ratio <= 0.25:
        return (
            "This is the clearest 'same shape, wrong x grid' pattern. The model is tracking the series "
            "well, but it snaps x values onto a nearby cleaner grid."
        )

    if example.series_point_count_ratio >= 0.8:
        return (
            "The model still captures most of the chart structure, but the predicted x locations are "
            "quantized onto a coarser interval than the gold labels."
        )

    return (
        "This case mixes x-grid mismatch with under-extraction, so the zero score is not only about the "
        "exact-x rule."
    )


def build_example_section(example: ExampleDiagnostics, report_path: Path) -> str:
    predicted_by_name = {
        series["name"]: series.get("points", [])
        for series in example.predicted_series
        if series.get("name")
    }
    representative_series = example.gold_series[:2]
    relative_image_path = example.image_path.relative_to(report_path.parent)

    lines = [
        f"## Example {example.example_id}: `{example.file_name}`",
        "",
        f"![Chart {example.example_id}]({relative_image_path.as_posix()})",
        "",
        f"- Title: `{example.title}`",
        f"- Axes: `x = {example.x_axis_label}` and `y = {example.y_axis_label}`",
        f"- Reward tuple: `reward={format_float(example.reward)}`, `series_name_f1={format_float(example.series_name_f1)}`, `series_point_count_ratio={format_float(example.series_point_count_ratio)}`, `series_point_value={format_float(example.series_point_value)}`",
        f"- Diagnostic tuple: `order_aligned_y_score={format_float(example.order_aligned_y_score)}`, `mean_x_step_ratio={format_ratio(example.mean_x_step_ratio)}`, `exact_x_match_fraction={format_float(example.exact_x_match_fraction)}`",
        f"- Read: {build_behavior_note(example)}",
        "",
    ]

    for gold_series in representative_series:
        predicted_points = predicted_by_name.get(gold_series["name"], [])
        lines.extend(
            [
                f"Representative series `{gold_series['name']}`",
                "",
                "Gold first 6 points:",
                "```json",
                format_points(gold_series["points"]),
                "```",
                "Predicted first 6 points:",
                "```json",
                format_points(predicted_points),
                "```",
                "",
            ]
        )

    if example.exact_x_match_fraction == 0.0:
        lines.append(
            "Why the reward collapses to zero here: the implementation only scores a point when the predicted `x` exactly equals a gold `x`. In this example there are effectively no exact `x` matches across the matched series, so every point contributes zero credit."
        )
    else:
        lines.append(
            "Why the reward stays very low here: only a small fraction of gold `x` values appear exactly in the prediction, so most points receive zero credit before `y` similarity even matters."
        )

    lines.append("")
    return "\n".join(lines)


def render_report(
    metadata: dict,
    distribution_summary: dict[str, float | int],
    strict_zero_summary: dict[str, float | int],
    examples: list[ExampleDiagnostics],
    report_path: Path,
    run_dir: Path,
) -> str:
    count = int(distribution_summary["count"])
    zero_count = int(distribution_summary["zero_point_value"])
    strict_zero_count = int(strict_zero_summary["count"])

    sections = [
        "# `series_point_value` low-score analysis",
        "",
        f"Generated from eval run `{run_dir}`.",
        "",
        "## Run summary",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Number of samples: `{metadata['num_examples']}`",
        f"- Average total reward: `{format_float(metadata['avg_reward'])}`",
        f"- Average `series_point_value`: `{format_float(distribution_summary['avg_point_value'])}`",
        "",
        "## Distribution snapshot",
        "",
        f"- `series_point_value = 0` on `{zero_count} / {count}` samples ({as_pct(zero_count / count)})",
        f"- `series_point_value < 0.05` on `{distribution_summary['lt_0_05']} / {count}` samples ({as_pct(float(distribution_summary['lt_0_05']) / count)})",
        f"- `series_point_value < 0.1` on `{distribution_summary['lt_0_1']} / {count}` samples ({as_pct(float(distribution_summary['lt_0_1']) / count)})",
        f"- Only `{distribution_summary['gt_0_5']}` samples scored above `0.5` on this reward",
        f"- Among the zero-score cases, `{distribution_summary['zero_and_perfect_names']}` still had perfect `series_name_f1`",
        f"- `{distribution_summary['zero_perfect_names_high_count']}` zero-score cases still had perfect names and `series_point_count_ratio >= 0.8`",
        f"- `{distribution_summary['zero_perfect_names_perfect_count']}` zero-score cases still had perfect names and perfect point counts",
        "",
        "## What the model seems to be doing",
        "",
        f"- In the strict subset where `series_point_value = 0`, `series_name_f1 = 1`, and `series_point_count_ratio = 1`, there are `{strict_zero_count}` samples.",
        f"- On those `{strict_zero_count}` samples, an order-aligned diagnostic y score averages `{format_float(strict_zero_summary['mean_order_aligned_y_score'])}` with median `{format_float(strict_zero_summary['median_order_aligned_y_score'])}`.",
        f"- Their mean x offset is only `{format_ratio(float(strict_zero_summary['mean_x_step_ratio']))}` gold x-steps, with median `{format_ratio(float(strict_zero_summary['median_x_step_ratio']))}`.",
        f"- `{strict_zero_summary['strong_shape_cases']}` of those `{strict_zero_count}` strict zero-score cases have both `order_aligned_y_score >= 0.8` and `mean_x_step_ratio <= 0.25`.",
        "",
        "Interpretation:",
        "",
        "- A large share of the zero scores are not 'the model found the wrong series' failures.",
        "- The common pattern is that the model recovers the right series names and roughly the right y trajectory, but snaps x values onto a nearby cleaner grid such as `60, 80, 100` instead of `58, 78, 98`.",
        "- There are still genuine misses too, especially when a series name is typoed or a series is omitted, but the exact-x matching rule is masking a lot of near-miss behavior that would be useful signal during RL.",
        "",
        "## Representative low-score examples",
        "",
        "_Each example below comes from the saved 2,000-sample run and is paired with the original chart image from the Hugging Face test split._",
        "",
    ]

    for example in examples:
        sections.append(build_example_section(example, report_path))

    sections.extend(
        [
            "## Initial takeaway",
            "",
            "- The current reward is excellent at detecting exact coordinate agreement, but it is harsh as a learning signal because a small x shift erases otherwise useful partial credit.",
            "- Before changing the reward, the model behavior to keep in mind is: names are often correct, counts are often close, y shapes are often close, and x values are frequently quantized or shifted onto a nearby regular grid.",
            "- That makes a tolerance-aware or alignment-aware variant of `series_point_value` a promising next step.",
            "",
        ]
    )

    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    report_path = args.report_path.resolve()
    asset_dir = args.asset_dir.resolve()
    example_ids = args.example_ids or DEFAULT_EXAMPLE_IDS

    metadata = load_metadata(run_dir)
    results = load_results(run_dir)
    results_by_id = {row["example_id"]: row for row in results}

    missing_example_ids = [example_id for example_id in example_ids if example_id not in results_by_id]
    if missing_example_ids:
        missing_ids = ", ".join(str(example_id) for example_id in missing_example_ids)
        raise KeyError(f"Missing example ids in results.jsonl: {missing_ids}")

    distribution_summary = build_distribution_summary(results)
    strict_zero_summary = build_strict_zero_case_summary(results)

    dataset_split = load_dataset("13point5/line-ex", split="test")
    examples = build_example_diagnostics(
        results_by_id=results_by_id,
        dataset_split=dataset_split,
        example_ids=example_ids,
        asset_dir=asset_dir,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_contents = render_report(
        metadata=metadata,
        distribution_summary=distribution_summary,
        strict_zero_summary=strict_zero_summary,
        examples=examples,
        report_path=report_path,
        run_dir=run_dir,
    )
    report_path.write_text(report_contents, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
