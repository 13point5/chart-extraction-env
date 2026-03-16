#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = REPO_ROOT / "environments/chart_extraction"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from rubric.rewards.series_point_values import (  # noqa: E402
    RELAXED_LINE_DISTANCE_RATIO,
    SeriesPointValueConfig,
    series_point_value_chart_score,
)
from schemas import CanonicalPoint  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "analysis/series_point_value_sweep"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
DETAILS_PATH = OUTPUT_DIR / "results.json"
CURATED_IMAGE_IDS = {
    "very_flat_sparse": 762,
    "very_flat_dense": 1829,
    "tall_sparse": 1657,
    "tall_dense": 901,
    "many_series": 29,
    "few_series_dense": 40,
    "small_image": 6,
    "large_image": 57,
}
SCENARIO_TARGETS = {
    "exact": 1.00,
    "x_shift_0.10_step": 0.90,
    "x_shift_0.25_step": 0.60,
    "x_shift_0.50_step": 0.15,
    "y_shift_abs_0.10": 0.90,
    "y_shift_abs_0.25": 0.60,
    "y_shift_0.02_span": 0.90,
    "y_shift_0.05_span": 0.55,
    "segment_midpoint": 0.35,
}


@dataclass(frozen=True)
class CuratedChart:
    bucket: str
    image_id: int
    file_name: str
    width: int
    height: int
    num_series: int
    total_points: int
    x_span: float
    y_span: float
    flatness: float
    series: dict[str, list[CanonicalPoint]]


def make_points(raw_series_xy: list[float]) -> list[CanonicalPoint]:
    return [
        CanonicalPoint(index=index, x=float(raw_series_xy[2 * index]), y=float(raw_series_xy[(2 * index) + 1]))
        for index in range(len(raw_series_xy) // 2)
    ]


def clone_series(series: dict[str, list[CanonicalPoint]]) -> dict[str, list[CanonicalPoint]]:
    return {
        name: [CanonicalPoint(index=point.index, x=point.x, y=point.y) for point in points]
        for name, points in series.items()
    }


def series_x_step(points: list[CanonicalPoint]) -> float:
    steps = [
        abs(points[index + 1].x - points[index].x)
        for index in range(len(points) - 1)
        if points[index + 1].x != points[index].x
    ]
    if not steps:
        return 0.0
    return float(statistics.median(steps))


def chart_y_span(series: dict[str, list[CanonicalPoint]]) -> float:
    ys = [point.y for points in series.values() for point in points]
    if not ys:
        return 0.0
    return float(max(ys) - min(ys))


def shift_x_by_step_fraction(
    series: dict[str, list[CanonicalPoint]],
    fraction: float,
) -> dict[str, list[CanonicalPoint]]:
    shifted: dict[str, list[CanonicalPoint]] = {}
    for name, points in series.items():
        step = series_x_step(points)
        delta = fraction * step
        shifted[name] = [
            CanonicalPoint(index=point.index, x=point.x + delta, y=point.y)
            for point in points
        ]
    return shifted


def shift_y_by_span_fraction(
    series: dict[str, list[CanonicalPoint]],
    fraction: float,
) -> dict[str, list[CanonicalPoint]]:
    delta = fraction * chart_y_span(series)
    return shift_y_by_absolute_delta(series, delta)


def shift_y_by_absolute_delta(
    series: dict[str, list[CanonicalPoint]],
    delta: float,
) -> dict[str, list[CanonicalPoint]]:
    return {
        name: [
            CanonicalPoint(index=point.index, x=point.x, y=point.y + delta)
            for point in points
        ]
        for name, points in series.items()
    }


def midpoint_series(points: list[CanonicalPoint]) -> list[CanonicalPoint]:
    if len(points) < 2:
        return [CanonicalPoint(index=point.index, x=point.x, y=point.y) for point in points]

    shifted: list[CanonicalPoint] = []
    for index, point in enumerate(points):
        if index == 0 or index == len(points) - 1:
            shifted.append(CanonicalPoint(index=point.index, x=point.x, y=point.y))
            continue

        next_point = points[index + 1]
        shifted.append(
            CanonicalPoint(
                index=point.index,
                x=(point.x + next_point.x) / 2.0,
                y=(point.y + next_point.y) / 2.0,
            )
        )

    return shifted


def move_points_to_midpoints(
    series: dict[str, list[CanonicalPoint]],
) -> dict[str, list[CanonicalPoint]]:
    return {name: midpoint_series(points) for name, points in series.items()}


def config_label(config: SeriesPointValueConfig) -> str:
    parts = [
        config.scale_mode,
        f"k={config.oks_k:.3f}",
        f"thr={config.oks_threshold:.2f}",
    ]
    if config.relaxed_line_distance_ratio is not None:
        parts.append(f"relaxed={config.relaxed_line_distance_ratio:.3f}")
    else:
        parts.append("relaxed=off")
    return " | ".join(parts)


def build_configs() -> list[SeriesPointValueConfig]:
    configs: list[SeriesPointValueConfig] = []
    for scale_mode in ("axis_spans", "chart_area"):
        for oks_k in (0.025, 0.035, 0.05):
            for threshold in (0.5, 0.4):
                configs.append(
                    SeriesPointValueConfig(
                        scale_mode=scale_mode,
                        oks_k=oks_k,
                        oks_threshold=threshold,
                    )
                )
                if scale_mode == "chart_area":
                    configs.append(
                        SeriesPointValueConfig(
                            scale_mode=scale_mode,
                            oks_k=oks_k,
                            oks_threshold=threshold,
                            relaxed_line_distance_ratio=RELAXED_LINE_DISTANCE_RATIO,
                        )
                    )
    return configs


def load_curated_charts() -> list[CuratedChart]:
    dataset = load_dataset("13point5/line-ex", split="test[:3000]")
    by_id = {row["image_id"]: row for row in dataset}

    charts: list[CuratedChart] = []
    for bucket, image_id in CURATED_IMAGE_IDS.items():
        row = by_id[image_id]
        series = {
            str(line["line_name"]).strip(): make_points(line["raw_series_xy"])
            for line in row["lines"]
            if str(line["line_name"]).strip()
        }
        xs = [point.x for points in series.values() for point in points]
        ys = [point.y for points in series.values() for point in points]
        charts.append(
            CuratedChart(
                bucket=bucket,
                image_id=int(row["image_id"]),
                file_name=str(row["file_name"]),
                width=int(row["width"]),
                height=int(row["height"]),
                num_series=len(series),
                total_points=sum(len(points) for points in series.values()),
                x_span=float(max(xs) - min(xs)),
                y_span=float(max(ys) - min(ys)),
                flatness=float((max(ys) - min(ys)) / max(max(xs) - min(xs), 1.0)),
                series=series,
            )
        )

    return charts


def build_scenarios(
    chart: CuratedChart,
) -> dict[str, dict[str, list[CanonicalPoint]]]:
    base = clone_series(chart.series)
    return {
        "exact": base,
        "x_shift_0.10_step": shift_x_by_step_fraction(base, 0.10),
        "x_shift_0.25_step": shift_x_by_step_fraction(base, 0.25),
        "x_shift_0.50_step": shift_x_by_step_fraction(base, 0.50),
        "y_shift_abs_0.10": shift_y_by_absolute_delta(base, 0.10),
        "y_shift_abs_0.25": shift_y_by_absolute_delta(base, 0.25),
        "y_shift_0.02_span": shift_y_by_span_fraction(base, 0.02),
        "y_shift_0.05_span": shift_y_by_span_fraction(base, 0.05),
        "segment_midpoint": move_points_to_midpoints(base),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_results(
    charts: list[CuratedChart],
    configs: list[SeriesPointValueConfig],
) -> list[dict]:
    flat_buckets = {"very_flat_sparse", "very_flat_dense"}
    tall_buckets = {"tall_sparse", "tall_dense"}
    results: list[dict] = []

    for config in configs:
        scenario_scores: dict[str, list[float]] = {name: [] for name in SCENARIO_TARGETS}
        flat_scores: dict[str, list[float]] = {name: [] for name in SCENARIO_TARGETS}
        tall_scores: dict[str, list[float]] = {name: [] for name in SCENARIO_TARGETS}
        per_chart: dict[str, dict[str, float]] = {}

        for chart in charts:
            scenarios = build_scenarios(chart)
            chart_scores: dict[str, float] = {}
            for scenario_name, predicted_series in scenarios.items():
                score = series_point_value_chart_score(predicted_series, chart.series, config=config)
                scenario_scores[scenario_name].append(score)
                chart_scores[scenario_name] = score

                if chart.bucket in flat_buckets:
                    flat_scores[scenario_name].append(score)
                if chart.bucket in tall_buckets:
                    tall_scores[scenario_name].append(score)

            per_chart[chart.bucket] = chart_scores

        scenario_means = {name: mean(scores) for name, scores in scenario_scores.items()}
        flat_means = {name: mean(scores) for name, scores in flat_scores.items()}
        tall_means = {name: mean(scores) for name, scores in tall_scores.items()}

        target_error = mean(
            [abs(scenario_means[name] - target) for name, target in SCENARIO_TARGETS.items()]
        )
        flat_tall_gap = abs(flat_means["y_shift_abs_0.10"] - tall_means["y_shift_abs_0.10"])
        objective = target_error + (0.5 * flat_tall_gap)

        results.append(
            {
                "config": asdict(config),
                "label": config_label(config),
                "objective": objective,
                "target_error": target_error,
                "flat_tall_gap_y_shift_abs_0_10": flat_tall_gap,
                "scenario_means": scenario_means,
                "flat_means": flat_means,
                "tall_means": tall_means,
                "per_chart": per_chart,
            }
        )

    return sorted(results, key=lambda row: row["objective"])


def format_table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]

    lines: list[str] = []
    header = " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(rows[0]))
    separator = " | ".join("-" * width for width in widths)
    lines.append(header)
    lines.append(separator)
    for row in rows[1:]:
        lines.append(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return lines


def build_markdown(charts: list[CuratedChart], results: list[dict]) -> str:
    best = results[0]
    current = next(
        row
        for row in results
        if row["config"] == asdict(SeriesPointValueConfig())
    )

    lines = [
        "# `series_point_value` parameter sweep",
        "",
        "This sweep uses curated LineEX charts and controlled perturbations derived from gold series.",
        "",
        "Curated charts:",
        "",
    ]

    chart_rows = [["Bucket", "Image", "Size", "Series", "Points", "y/x span"]]
    for chart in charts:
        chart_rows.append(
            [
                chart.bucket,
                f"{chart.image_id} ({chart.file_name})",
                f"{chart.width}x{chart.height}",
                str(chart.num_series),
                str(chart.total_points),
                f"{chart.flatness:.4f}",
            ]
        )
    lines.extend(format_table(chart_rows))
    lines.extend(
        [
            "",
            "Scenario targets used for ranking configs:",
            "",
        ]
    )
    for scenario_name, target in SCENARIO_TARGETS.items():
        lines.append(f"- `{scenario_name}` target mean: `{target:.2f}`")

    lines.extend(
        [
            "",
            "Top configs by heuristic objective:",
            "",
        ]
    )
    top_rows = [[
        "Rank",
        "Config",
        "Objective",
        "Exact",
        "x0.25",
        "x0.50",
        "y_abs0.10",
        "Segment",
        "Flat/Tall gap",
    ]]
    for rank, row in enumerate(results[:8], start=1):
        means = row["scenario_means"]
        top_rows.append(
            [
                str(rank),
                row["label"],
                f"{row['objective']:.4f}",
                f"{means['exact']:.3f}",
                f"{means['x_shift_0.25_step']:.3f}",
                f"{means['x_shift_0.50_step']:.3f}",
                f"{means['y_shift_abs_0.10']:.3f}",
                f"{means['segment_midpoint']:.3f}",
                f"{row['flat_tall_gap_y_shift_abs_0_10']:.3f}",
            ]
        )
    lines.extend(format_table(top_rows))

    lines.extend(
        [
            "",
            "Current default vs best heuristic config:",
            "",
        ]
    )
    compare_rows = [[
        "Scenario",
        "Current",
        "Best",
    ]]
    for scenario_name in SCENARIO_TARGETS:
        compare_rows.append(
            [
                scenario_name,
                f"{current['scenario_means'][scenario_name]:.3f}",
                f"{best['scenario_means'][scenario_name]:.3f}",
            ]
        )
    lines.extend(format_table(compare_rows))

    lines.extend(
        [
            "",
            f"Best config: `{best['label']}`",
            "",
            "Notes:",
            "",
            "- `axis_spans` is the current reward behavior.",
            "- `chart_area` is a more LineEX-like variant that uses a single chart-scale term analogous to `sqrt(h*w)` rather than per-axis normalization.",
            "- `relaxed=0.007` adds a segment fallback inspired by the original LineEX relaxed metric.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = load_curated_charts()
    results = summarize_results(charts, build_configs())

    SUMMARY_PATH.write_text(build_markdown(charts, results), encoding="utf-8")
    DETAILS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    best = results[0]
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILS_PATH}")
    print(f"Best config: {best['label']}")


if __name__ == "__main__":
    main()
