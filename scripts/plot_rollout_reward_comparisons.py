#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = REPO_ROOT / "environments/chart_extraction"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from rubric.rewards.series_name_f1 import f1_score  # noqa: E402
from rubric.rewards.series_point_values import (  # noqa: E402
    DEFAULT_SERIES_POINT_VALUE_CONFIG,
    LEGACY_AXIS_SPAN_CONFIG,
    series_point_value_chart_score,
)
from rubric.rewards.series_points import point_count_ratio  # noqa: E402
from schemas import CanonicalChart, CanonicalPoint, parse_chart_extraction  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "analysis/rollout_reward_gallery"
PLOTS_DIR = OUTPUT_DIR / "plots"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
ALL_SAMPLES_CSV_PATH = OUTPUT_DIR / "all_samples.csv"
SELECTED_SAMPLES_CSV_PATH = OUTPUT_DIR / "selected_samples.csv"
RECENT_RUNS_JSON_PATH = OUTPUT_DIR / "recent_stopped_runs.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "contact-sheet.png"

RUN_LIST_PAGE_SIZE = 20
NUM_RECENT_STOPPED_RUNS = 3
MAX_PLOTS = 8
MIN_SAMPLES_PER_RUN = 1

FORMAT_WEIGHT = 1.0
NAME_WEIGHT = 1.0
COUNT_WEIGHT = 2.0
POINT_WEIGHT = 2.0


@dataclass(frozen=True, slots=True)
class RunSelection:
    run_id: str
    run_name: str
    version: str
    latest_sample_step: int
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    format_reward: float
    name_f1: float
    point_count_ratio: float
    point_value_old: float
    point_value_new: float
    total_old: float
    total_new: float


@dataclass(frozen=True, slots=True)
class SampleAnalysis:
    run_id: str
    run_name: str
    version: str
    step: int
    row_index: int
    sample_id: int
    problem_id: int
    file_name: str
    image_id: int | None
    schema_version: str
    logged_reward: float | None
    logged_format_reward: float | None
    logged_name_f1: float | None
    logged_point_count_ratio: float | None
    logged_point_value: float | None
    reward: RewardBreakdown
    plot_path: Path


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_prime_command(args: list[str]) -> str:
    completed = subprocess.run(
        ["prime", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def repair_json_like(text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                repaired.append(ch)
                escape = False
            elif ch == "\\":
                repaired.append(ch)
                escape = True
            elif ch == '"':
                repaired.append(ch)
                in_string = False
            elif ch == "\n":
                repaired.append("\\n")
            elif ch == "\r":
                repaired.append("\\r")
            elif ch == "\t":
                repaired.append("\\t")
            else:
                repaired.append(ch)
        else:
            repaired.append(ch)
            if ch == '"':
                in_string = True

    return "".join(repaired)


def prime_json(args: list[str]) -> dict[str, Any]:
    return json.loads(repair_json_like(run_prime_command(args)))


def loads_repaired(text: str) -> Any:
    return json.loads(repair_json_like(text))


def recent_stopped_runs() -> list[RunSelection]:
    data = prime_json(["rl", "list", "-n", str(RUN_LIST_PAGE_SIZE), "-o", "json"])
    RECENT_RUNS_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    selections: list[RunSelection] = []
    for run in data.get("runs", []):
        if run.get("status") != "STOPPED":
            continue

        progress = prime_json(["rl", "progress", run["id"]])
        sample_steps = progress.get("steps_with_samples", [])
        if len(sample_steps) < MIN_SAMPLES_PER_RUN:
            continue

        envs = run.get("environments", [])
        version = envs[0]["version"] if envs else "unknown"
        selections.append(
            RunSelection(
                run_id=run["id"],
                run_name=run["name"],
                version=version,
                latest_sample_step=int(sample_steps[-1]),
                completed_at=run.get("completed_at"),
            )
        )
        if len(selections) >= NUM_RECENT_STOPPED_RUNS:
            break

    return selections


def extract_answer_text(completion: str) -> str | None:
    completion_text = completion
    try:
        parsed_completion = loads_repaired(completion)
    except json.JSONDecodeError:
        parsed_completion = None

    if isinstance(parsed_completion, list):
        for message in parsed_completion:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                completion_text = content
                break

    match = re.search(r"<answer>(.*?)</answer>", completion_text, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def parse_rollout_chart(answer_text: str | None, schema_version: str) -> CanonicalChart | None:
    if not answer_text:
        return None

    try:
        return parse_chart_extraction(loads_repaired(answer_text), schema_version=schema_version).to_canonical()
    except Exception:
        return None


def canonical_series_map(chart: CanonicalChart | None) -> dict[str, list[CanonicalPoint]]:
    if chart is None:
        return {}
    return {series.name: series.points for series in chart.series if series.name}


def weighted_point_count_ratio(
    predicted_series: dict[str, list[CanonicalPoint]],
    gold_series: dict[str, list[CanonicalPoint]],
) -> float:
    if not gold_series:
        return 1.0 if not predicted_series else 0.0

    weighted_score_sum = 0.0
    total_weight = 0
    for name, gold_points in gold_series.items():
        weight = max(len(gold_points), 1)
        weighted_score_sum += point_count_ratio(predicted_series.get(name, []), gold_points) * weight
        total_weight += weight
    return weighted_score_sum / total_weight if total_weight else 0.0


def compute_reward_breakdown(
    predicted_chart: CanonicalChart | None,
    gold_chart: CanonicalChart,
) -> RewardBreakdown:
    predicted_series = canonical_series_map(predicted_chart)
    gold_series = canonical_series_map(gold_chart)

    predicted_names = set(predicted_series)
    gold_names = set(gold_series)
    format_reward = 1.0 if predicted_chart is not None else 0.0
    name_f1 = f1_score(predicted_names, gold_names) if predicted_chart is not None else 0.0
    count_ratio = (
        weighted_point_count_ratio(predicted_series, gold_series)
        if predicted_chart is not None
        else 0.0
    )
    point_value_old = (
        series_point_value_chart_score(
            predicted_series,
            gold_series,
            config=LEGACY_AXIS_SPAN_CONFIG,
        )
        if predicted_chart is not None
        else 0.0
    )
    point_value_new = (
        series_point_value_chart_score(
            predicted_series,
            gold_series,
            config=DEFAULT_SERIES_POINT_VALUE_CONFIG,
        )
        if predicted_chart is not None
        else 0.0
    )

    total_old = (
        (FORMAT_WEIGHT * format_reward)
        + (NAME_WEIGHT * name_f1)
        + (COUNT_WEIGHT * count_ratio)
        + (POINT_WEIGHT * point_value_old)
    )
    total_new = (
        (FORMAT_WEIGHT * format_reward)
        + (NAME_WEIGHT * name_f1)
        + (COUNT_WEIGHT * count_ratio)
        + (POINT_WEIGHT * point_value_new)
    )

    return RewardBreakdown(
        format_reward=format_reward,
        name_f1=name_f1,
        point_count_ratio=count_ratio,
        point_value_old=point_value_old,
        point_value_new=point_value_new,
        total_old=total_old,
        total_new=total_new,
    )


def series_axis_bounds(
    gold_chart: CanonicalChart,
    predicted_chart: CanonicalChart | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for chart in (gold_chart, predicted_chart):
        if chart is None:
            continue
        for series in chart.series:
            for point in series.points:
                xs.append(float(point.x))
                ys.append(float(point.y))

    if not xs or not ys:
        return (0.0, 1.0), (0.0, 1.0)

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def build_color_map(
    gold_chart: CanonicalChart,
    predicted_chart: CanonicalChart | None,
) -> dict[str, Any]:
    ordered_names: list[str] = []
    for chart in (gold_chart, predicted_chart):
        if chart is None:
            continue
        for series in chart.series:
            if series.name and series.name not in ordered_names:
                ordered_names.append(series.name)

    cmap = plt.get_cmap("tab20")
    return {name: cmap(index % 20) for index, name in enumerate(ordered_names)}


def add_chart_plot(
    ax: Any,
    chart: CanonicalChart | None,
    *,
    title: str,
    colors: dict[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    line_style: str = "-",
) -> None:
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if chart is None or not chart.series:
        ax.text(0.5, 0.5, "No parsed chart", ha="center", va="center", transform=ax.transAxes)
        return

    for series in chart.series:
        if not series.points:
            continue
        xs = [point.x for point in series.points]
        ys = [point.y for point in series.points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            linestyle=line_style,
            color=colors.get(series.name),
            label=f"{series.name} ({len(series.points)})",
        )
    ax.legend(fontsize=7, loc="best")


def add_reward_plot(ax: Any, reward: RewardBreakdown) -> None:
    categories = ["format", "name", "count x2", "point x2", "total"]
    old_values = [
        FORMAT_WEIGHT * reward.format_reward,
        NAME_WEIGHT * reward.name_f1,
        COUNT_WEIGHT * reward.point_count_ratio,
        POINT_WEIGHT * reward.point_value_old,
        reward.total_old,
    ]
    new_values = [
        FORMAT_WEIGHT * reward.format_reward,
        NAME_WEIGHT * reward.name_f1,
        COUNT_WEIGHT * reward.point_count_ratio,
        POINT_WEIGHT * reward.point_value_new,
        reward.total_new,
    ]

    positions = list(range(len(categories)))
    width = 0.36
    ax.bar([pos - (width / 2) for pos in positions], old_values, width=width, label="Old", color="#c97b63")
    ax.bar([pos + (width / 2) for pos in positions], new_values, width=width, label="New", color="#4b7f8c")
    ax.set_xticks(positions, categories, rotation=20)
    ax.set_ylim(0.0, max(6.2, max(old_values + new_values) + 0.4))
    ax.set_title("Reward Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.98,
        (
            f"Point value old: {reward.point_value_old:.3f}\n"
            f"Point value new: {reward.point_value_new:.3f}\n"
            f"Delta total: {reward.total_new - reward.total_old:+.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )


def add_overlay_plot(
    ax: Any,
    *,
    gold_chart: CanonicalChart,
    predicted_chart: CanonicalChart | None,
    colors: dict[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_title("Overlay")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for series in gold_chart.series:
        if not series.points:
            continue
        xs = [point.x for point in series.points]
        ys = [point.y for point in series.points]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4.0,
            linewidth=2.0,
            linestyle="-",
            color=colors.get(series.name),
            alpha=0.95,
            label=f"GT {series.name}",
        )

    if predicted_chart is not None:
        for series in predicted_chart.series:
            if not series.points:
                continue
            xs = [point.x for point in series.points]
            ys = [point.y for point in series.points]
            ax.plot(
                xs,
                ys,
                marker="x",
                markersize=4.0,
                linewidth=1.8,
                linestyle="--",
                color=colors.get(series.name),
                alpha=0.8,
                label=f"Pred {series.name}",
            )

    ax.legend(fontsize=6.5, loc="best")


def plot_sample(
    *,
    sample: dict[str, Any],
    row_index: int,
    sample_index: int,
    run_selection: RunSelection,
    gold_chart: CanonicalChart,
    predicted_chart: CanonicalChart | None,
    reward: RewardBreakdown,
) -> Path:
    file_name = sample["file_name"]
    safe_stem = Path(file_name).stem
    prefix = (
        f"{run_selection.run_id[:6]}_step{run_selection.latest_sample_step}"
        f"_row{row_index:02d}"
        f"_sample{sample_index:02d}_{safe_stem}"
    )
    plot_path = PLOTS_DIR / f"{prefix}_comparison.png"

    xlim, ylim = series_axis_bounds(gold_chart, predicted_chart)
    colors = build_color_map(gold_chart, predicted_chart)

    fig: Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        (
            f"{run_selection.run_id[:6]}  step {run_selection.latest_sample_step}  "
            f"row {row_index}  sample {sample_index}  file {file_name}"
        ),
        fontsize=14,
    )

    add_chart_plot(
        axes[0, 0],
        gold_chart,
        title="Ground Truth",
        colors=colors,
        xlim=xlim,
        ylim=ylim,
        line_style="-",
    )
    add_chart_plot(
        axes[0, 1],
        predicted_chart,
        title="Prediction From Rollout",
        colors=colors,
        xlim=xlim,
        ylim=ylim,
        line_style="--",
    )
    add_overlay_plot(
        axes[1, 0],
        gold_chart=gold_chart,
        predicted_chart=predicted_chart,
        colors=colors,
        xlim=xlim,
        ylim=ylim,
    )
    add_reward_plot(axes[1, 1], reward)

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relpath_from_output(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


def make_contact_sheet(selected_samples: list[SampleAnalysis]) -> None:
    if not selected_samples:
        return

    columns = 2
    rows = math.ceil(len(selected_samples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 5), constrained_layout=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax in axes_list:
        ax.axis("off")

    for ax, sample in zip(axes_list, selected_samples):
        image = Image.open(sample.plot_path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            (
                f"{sample.run_id[:6]} step {sample.step} row {sample.row_index}\n"
                f"old {sample.reward.total_old:.2f}  new {sample.reward.total_new:.2f}"
            ),
            fontsize=10,
        )

    figure.savefig(CONTACT_SHEET_PATH, dpi=150)
    plt.close(figure)


def markdown_gallery(samples: list[SampleAnalysis], selected_samples: list[SampleAnalysis]) -> str:
    lines = [
        "# Rollout Reward Comparison Gallery",
        "",
        "This gallery uses rollout samples from the most recent stopped Prime RL runs with saved sample steps.",
        "For each example:",
        "- `Ground Truth` comes from the rollout `info.expected_answer` payload.",
        "- `Prediction From Rollout` comes from the rollout completion.",
        "- `Overlay` makes it easier to see whether point errors are mostly shifts, count mismatches, or name mismatches.",
        "- `Reward Comparison` shows the old and new reward totals using the current local code.",
        "",
        f"Saved {len(samples)} total analyzed samples from {NUM_RECENT_STOPPED_RUNS} recent stopped runs.",
        "",
        f"![Top reward deltas]({CONTACT_SHEET_PATH})",
        "",
        "## Selected Examples",
        "",
        "The selected set includes the biggest reward increases, one case where the new reward is harsher, and one near-neutral case.",
        "",
    ]

    for sample in selected_samples:
        lines.extend(
            [
                (
                    f"### {sample.run_id[:6]} step {sample.step} row {sample.row_index} "
                    f"sample {sample.sample_id} ({sample.file_name})"
                ),
                "",
                (
                    f"- Total reward: old `{sample.reward.total_old:.3f}` -> "
                    f"new `{sample.reward.total_new:.3f}` "
                    f"(`{sample.reward.total_new - sample.reward.total_old:+.3f}`)"
                ),
                (
                    f"- Point value: old `{sample.reward.point_value_old:.3f}` -> "
                    f"new `{sample.reward.point_value_new:.3f}`"
                ),
                (
                    f"- Shared components: format `{sample.reward.format_reward:.1f}`, "
                    f"name F1 `{sample.reward.name_f1:.3f}`, "
                    f"count ratio `{sample.reward.point_count_ratio:.3f}`"
                ),
                "",
                f"![{sample.file_name}]({sample.plot_path})",
                "",
            ]
        )

    lines.extend(
        [
            "## Files",
            "",
            f"- All samples CSV: `{ALL_SAMPLES_CSV_PATH}`",
            f"- Selected samples CSV: `{SELECTED_SAMPLES_CSV_PATH}`",
            f"- Plot directory: `{PLOTS_DIR}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_run(run_selection: RunSelection) -> list[SampleAnalysis]:
    rollout_data = prime_json(
        ["rl", "rollouts", run_selection.run_id, "-s", str(run_selection.latest_sample_step), "-n", "100"]
    )
    analyzed_samples: list[SampleAnalysis] = []

    for row_index, sample in enumerate(rollout_data.get("samples", [])):
        metrics = loads_repaired(sample["metrics"])
        info = loads_repaired(sample["info"])

        schema_version = info.get("schema_version", "v1")
        gold_answer = info.get(
            "expected_answer",
            {
                "title": info.get("title", ""),
                "x_axis_label": info.get("x_axis_label", ""),
                "y_axis_label": info.get("y_axis_label", ""),
                "series": info.get("series", []),
            },
        )
        gold_chart = parse_chart_extraction(gold_answer, schema_version=schema_version).to_canonical()
        predicted_chart = parse_rollout_chart(extract_answer_text(sample["completion"]), schema_version=schema_version)
        reward = compute_reward_breakdown(predicted_chart, gold_chart)

        plot_path = plot_sample(
            sample=info,
            row_index=row_index,
            sample_index=int(sample["sample_id"]),
            run_selection=run_selection,
            gold_chart=gold_chart,
            predicted_chart=predicted_chart,
            reward=reward,
        )

        analyzed_samples.append(
            SampleAnalysis(
                run_id=run_selection.run_id,
                run_name=run_selection.run_name,
                version=run_selection.version,
                step=run_selection.latest_sample_step,
                row_index=row_index,
                sample_id=int(sample["sample_id"]),
                problem_id=int(sample["problem_id"]),
                file_name=str(info.get("file_name", "unknown.png")),
                image_id=int(info["image_id"]) if info.get("image_id") is not None else None,
                schema_version=str(schema_version),
                logged_reward=float(sample["reward"]) if sample.get("reward") is not None else None,
                logged_format_reward=(
                    float(metrics["format_reward_func"])
                    if metrics.get("format_reward_func") is not None
                    else None
                ),
                logged_name_f1=(
                    float(metrics["series_name_f1"])
                    if metrics.get("series_name_f1") is not None
                    else None
                ),
                logged_point_count_ratio=(
                    float(metrics["series_point_count_ratio"])
                    if metrics.get("series_point_count_ratio") is not None
                    else None
                ),
                logged_point_value=(
                    float(metrics["series_point_value"])
                    if metrics.get("series_point_value") is not None
                    else None
                ),
                reward=reward,
                plot_path=plot_path,
            )
        )

    return analyzed_samples


def analysis_rows(samples: list[SampleAnalysis]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "run_id": sample.run_id,
                "run_name": sample.run_name,
                "version": sample.version,
                "step": sample.step,
                "row_index": sample.row_index,
                "sample_id": sample.sample_id,
                "problem_id": sample.problem_id,
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "schema_version": sample.schema_version,
                "logged_reward": sample.logged_reward,
                "logged_format_reward": sample.logged_format_reward,
                "logged_name_f1": sample.logged_name_f1,
                "logged_point_count_ratio": sample.logged_point_count_ratio,
                "logged_point_value": sample.logged_point_value,
                "format_reward": sample.reward.format_reward,
                "name_f1": sample.reward.name_f1,
                "point_count_ratio": sample.reward.point_count_ratio,
                "point_value_old": sample.reward.point_value_old,
                "point_value_new": sample.reward.point_value_new,
                "total_old": sample.reward.total_old,
                "total_new": sample.reward.total_new,
                "delta_total": sample.reward.total_new - sample.reward.total_old,
                "delta_point_value": sample.reward.point_value_new - sample.reward.point_value_old,
                "plot_path": str(sample.plot_path),
            }
        )
    return rows


def choose_selected_samples(samples: list[SampleAnalysis]) -> list[SampleAnalysis]:
    positive_sorted = sorted(
        samples,
        key=lambda sample: (
            sample.reward.total_new - sample.reward.total_old,
            sample.reward.point_value_new - sample.reward.point_value_old,
        ),
        reverse=True,
    )
    negative_sorted = sorted(
        samples,
        key=lambda sample: (
            sample.reward.total_new - sample.reward.total_old,
            sample.reward.point_value_new - sample.reward.point_value_old,
        ),
    )
    neutral_sorted = sorted(
        samples,
        key=lambda sample: (
            abs(sample.reward.total_new - sample.reward.total_old),
            -sample.reward.total_old,
        ),
    )

    selected: list[SampleAnalysis] = []
    seen_paths: set[Path] = set()

    def add_candidates(candidates: list[SampleAnalysis], count: int) -> None:
        for candidate in candidates:
            if len(selected) >= MAX_PLOTS:
                return
            if candidate.plot_path in seen_paths:
                continue
            selected.append(candidate)
            seen_paths.add(candidate.plot_path)
            if count > 0:
                count -= 1
            if count == 0:
                return

    add_candidates(positive_sorted, max(MAX_PLOTS - 2, 1))
    add_candidates(negative_sorted, 1)
    add_candidates(neutral_sorted, 1)
    add_candidates(positive_sorted, MAX_PLOTS)
    return selected[:MAX_PLOTS]


def main() -> int:
    ensure_output_dirs()
    runs = recent_stopped_runs()
    if not runs:
        raise RuntimeError("Did not find any recent stopped runs with rollout samples.")

    all_samples: list[SampleAnalysis] = []
    for run in runs:
        all_samples.extend(analyze_run(run))

    if not all_samples:
        raise RuntimeError("No rollout samples were analyzed.")

    all_rows = analysis_rows(all_samples)
    write_csv(ALL_SAMPLES_CSV_PATH, all_rows)

    selected_samples = choose_selected_samples(all_samples)
    selected_rows = analysis_rows(selected_samples)
    write_csv(SELECTED_SAMPLES_CSV_PATH, selected_rows)
    make_contact_sheet(selected_samples)
    SUMMARY_PATH.write_text(markdown_gallery(all_samples, selected_samples), encoding="utf-8")

    print(f"Wrote gallery to {SUMMARY_PATH}")
    print(f"Wrote {len(all_samples)} analyzed samples to {ALL_SAMPLES_CSV_PATH}")
    print(f"Wrote {len(selected_samples)} selected plots to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
