#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = REPO_ROOT / "environments/chart_extraction"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from schemas import CanonicalChart, CanonicalPoint, parse_chart_extraction  # noqa: E402
from rubric.rewards.series_name_f1 import f1_score  # noqa: E402
from rubric.rewards.series_point_values import OKS_K, OKS_THRESHOLD  # noqa: E402
from rubric.rewards.series_points import point_count_ratio  # noqa: E402


OUTPUT_PATH = REPO_ROOT / "analysis/series_point_value_rollout_playground.html"
RUN_ID = "n4di6f7er5n292nihoknhksf"

FORMAT_WEIGHT = 1.0
NAME_WEIGHT = 1.0
COUNT_WEIGHT = 2.0
POINT_WEIGHT = 2.0

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone HTML playground for chart series_point_value rollouts.",
    )
    parser.add_argument(
        "--run-id",
        default=RUN_ID,
        help="Prime RL run id to pull rollout samples from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output HTML path.",
    )
    parser.add_argument(
        "--num-per-step",
        type=int,
        default=100,
        help="How many rollout samples to request per sampled step.",
    )
    return parser.parse_args()


def run_prime_command(args: list[str]) -> str:
    attempts = 3
    last_error: subprocess.CalledProcessError | None = None

    for attempt in range(attempts):
        try:
            completed = subprocess.run(
                ["prime", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Prime command failed without raising CalledProcessError: {args}") from last_error


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

    match = ANSWER_RE.search(completion_text)
    if match is None:
        return None
    return match.group(1).strip()


def canonical_from_info(info: dict[str, Any]) -> CanonicalChart:
    schema_version = info.get("schema_version", "v1")
    payload = info.get(
        "expected_answer",
        {
            "title": info.get("title", ""),
            "x_axis_label": info.get("x_axis_label", ""),
            "y_axis_label": info.get("y_axis_label", ""),
            "series": info.get("series", []),
        },
    )
    return parse_chart_extraction(payload, schema_version=schema_version).to_canonical()


def canonical_from_answer(answer_text: str | None, schema_version: str) -> CanonicalChart | None:
    if not answer_text:
        return None

    try:
        raw_answer = loads_repaired(answer_text)
    except json.JSONDecodeError:
        return None

    try:
        return parse_chart_extraction(raw_answer, schema_version=schema_version).to_canonical()
    except Exception:
        return None


def chart_to_data(chart: CanonicalChart | None) -> dict[str, Any] | None:
    if chart is None:
        return None

    return {
        "title": chart.title,
        "x_axis_label": chart.x_axis_label,
        "y_axis_label": chart.y_axis_label,
        "series": [
            {
                "name": series.name,
                "points": [[float(point.x), float(point.y)] for point in series.points],
            }
            for series in chart.series
        ],
    }


def canonical_series_map(chart: CanonicalChart | None) -> dict[str, list[CanonicalPoint]]:
    if chart is None:
        return {}
    return {series.name: list(series.points) for series in chart.series if series.name}


def point_pairs(points: list[CanonicalPoint]) -> list[tuple[float, float]]:
    return [(float(point.x), float(point.y)) for point in points]


def normalize_point(
    point: tuple[float, float],
    *,
    x_min: float,
    x_scale: float,
    y_min: float,
    y_scale: float,
) -> tuple[float, float]:
    return ((point[0] - x_min) / x_scale, (point[1] - y_min) / y_scale)


def oks(distance: float, *, k: float) -> float:
    return math.exp(-(distance**2) / (2.0 * (k**2)))


def nearest_gold_index(
    predicted_point: tuple[float, float],
    gold_points: list[tuple[float, float]],
) -> tuple[float, int]:
    best_distance = math.inf
    best_index = -1

    for index, gold_point in enumerate(gold_points):
        distance = math.dist(predicted_point, gold_point)
        if distance < best_distance:
            best_distance = distance
            best_index = index

    return best_distance, best_index


def compute_chart_bounds(
    gold_series: dict[str, list[CanonicalPoint]],
) -> tuple[float, float, float, float]:
    all_gold_pairs = [
        pair
        for gold_points in gold_series.values()
        for pair in point_pairs(gold_points)
    ]

    if not all_gold_pairs:
        return 0.0, 1.0, 0.0, 1.0

    gold_xs = [x for x, _ in all_gold_pairs]
    gold_ys = [y for _, y in all_gold_pairs]
    x_min = min(gold_xs)
    y_min = min(gold_ys)
    x_scale = max(max(gold_xs) - x_min, 1.0)
    y_scale = max(max(gold_ys) - y_min, 1.0)
    return x_min, x_scale, y_min, y_scale


def current_series_point_value_detail(
    predicted_points: list[CanonicalPoint],
    gold_points: list[CanonicalPoint],
    *,
    x_min: float,
    x_scale: float,
    y_min: float,
    y_scale: float,
    oks_k: float,
    oks_threshold: float,
) -> dict[str, Any]:
    if not predicted_points and not gold_points:
        return {"score": 1.0, "matched_gold": 0, "gold_count": 0, "pred_count": 0}
    if not gold_points:
        return {
            "score": 1.0 if not predicted_points else 0.0,
            "matched_gold": 0,
            "gold_count": 0,
            "pred_count": len(predicted_points),
        }
    if not predicted_points:
        return {"score": 0.0, "matched_gold": 0, "gold_count": len(gold_points), "pred_count": 0}

    gold_pairs = point_pairs(gold_points)
    predicted_pairs = point_pairs(predicted_points)

    if not gold_pairs:
        return {
            "score": 1.0 if not predicted_pairs else 0.0,
            "matched_gold": 0,
            "gold_count": 0,
            "pred_count": len(predicted_pairs),
        }
    if not predicted_pairs:
        return {"score": 0.0, "matched_gold": 0, "gold_count": len(gold_pairs), "pred_count": 0}

    normalized_gold = [
        normalize_point(
            point,
            x_min=x_min,
            x_scale=x_scale,
            y_min=y_min,
            y_scale=y_scale,
        )
        for point in gold_pairs
    ]
    normalized_predicted = [
        normalize_point(
            point,
            x_min=x_min,
            x_scale=x_scale,
            y_min=y_min,
            y_scale=y_scale,
        )
        for point in predicted_pairs
    ]

    found_gold_indices: set[int] = set()
    for predicted_point in normalized_predicted:
        min_distance, gold_index = nearest_gold_index(predicted_point, normalized_gold)
        if gold_index < 0:
            continue

        if oks(min_distance, k=oks_k) > oks_threshold:
            found_gold_indices.add(gold_index)

    return {
        "score": len(found_gold_indices) / len(gold_pairs),
        "matched_gold": len(found_gold_indices),
        "gold_count": len(gold_pairs),
        "pred_count": len(predicted_pairs),
    }


def current_chart_point_value(
    predicted_series: dict[str, list[CanonicalPoint]],
    gold_series: dict[str, list[CanonicalPoint]],
    *,
    oks_k: float,
    oks_threshold: float,
) -> float:
    if not gold_series:
        return 1.0 if not predicted_series else 0.0

    x_min, x_scale, y_min, y_scale = compute_chart_bounds(gold_series)
    weighted_score_sum = 0.0
    total_weight = 0

    for name, gold_points in gold_series.items():
        weight = max(len(gold_points), 1)
        detail = current_series_point_value_detail(
            predicted_series.get(name, []),
            gold_points,
            x_min=x_min,
            x_scale=x_scale,
            y_min=y_min,
            y_scale=y_scale,
            oks_k=oks_k,
            oks_threshold=oks_threshold,
        )
        weighted_score_sum += detail["score"] * weight
        total_weight += weight

    return weighted_score_sum / total_weight if total_weight else 0.0


def legacy_series_point_value_detail(
    predicted_points: list[CanonicalPoint],
    gold_points: list[CanonicalPoint],
) -> dict[str, Any]:
    if not predicted_points and not gold_points:
        return {"score": 1.0, "matched_gold": 0, "gold_count": 0, "pred_count": 0}
    if not gold_points:
        return {
            "score": 1.0 if not predicted_points else 0.0,
            "matched_gold": 0,
            "gold_count": 0,
            "pred_count": len(predicted_points),
        }
    if not predicted_points:
        return {"score": 0.0, "matched_gold": 0, "gold_count": len(gold_points), "pred_count": 0}

    predicted_y_by_x = {float(point.x): float(point.y) for point in predicted_points}
    gold_pairs = point_pairs(gold_points)
    if not gold_pairs:
        return {
            "score": 1.0 if not predicted_y_by_x else 0.0,
            "matched_gold": 0,
            "gold_count": 0,
            "pred_count": len(predicted_y_by_x),
        }

    gold_ys = [gold_y for _, gold_y in gold_pairs]
    y_scale = max(max(gold_ys) - min(gold_ys), 1.0)

    total_score = 0.0
    matched_gold = 0
    for gold_x, gold_y in gold_pairs:
        predicted_y = predicted_y_by_x.get(gold_x)
        if predicted_y is None:
            continue

        matched_gold += 1
        normalized_y_error = abs(predicted_y - gold_y) / y_scale
        total_score += max(0.0, 1.0 - normalized_y_error)

    return {
        "score": total_score / len(gold_pairs),
        "matched_gold": matched_gold,
        "gold_count": len(gold_pairs),
        "pred_count": len(predicted_y_by_x),
    }


def legacy_chart_point_value(
    predicted_series: dict[str, list[CanonicalPoint]],
    gold_series: dict[str, list[CanonicalPoint]],
) -> float:
    if not gold_series:
        return 1.0 if not predicted_series else 0.0

    weighted_score_sum = 0.0
    total_weight = 0
    for name, gold_points in gold_series.items():
        weight = max(len(gold_points), 1)
        detail = legacy_series_point_value_detail(predicted_series.get(name, []), gold_points)
        weighted_score_sum += detail["score"] * weight
        total_weight += weight
    return weighted_score_sum / total_weight if total_weight else 0.0


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


def sort_point_pairs(points: list[CanonicalPoint]) -> list[tuple[float, float]]:
    return sorted(point_pairs(points), key=lambda pair: (pair[0], pair[1]))


def order_aligned_y_score(
    predicted_points: list[CanonicalPoint],
    gold_points: list[CanonicalPoint],
) -> float:
    predicted = sort_point_pairs(predicted_points)
    gold = sort_point_pairs(gold_points)

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
    predicted_points: list[CanonicalPoint],
    gold_points: list[CanonicalPoint],
) -> float | None:
    predicted = sort_point_pairs(predicted_points)
    gold = sort_point_pairs(gold_points)

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
    if reference_step == 0:
        return None

    mean_abs_x_offset = statistics.mean(
        abs(predicted[index][0] - gold[index][0])
        for index in range(len(gold))
    )
    return mean_abs_x_offset / reference_step


def exact_x_match_fraction(
    predicted_points: list[CanonicalPoint],
    gold_points: list[CanonicalPoint],
) -> float:
    predicted_xs = {float(point.x) for point in predicted_points}
    gold_xs = [float(point.x) for point in gold_points]

    if not gold_xs:
        return 1.0 if not predicted_xs else 0.0

    exact_matches = sum(1 for gold_x in gold_xs if gold_x in predicted_xs)
    return exact_matches / len(gold_xs)


def sample_diagnostics(
    predicted_series: dict[str, list[CanonicalPoint]],
    gold_series: dict[str, list[CanonicalPoint]],
) -> dict[str, Any]:
    total_weight = 0
    weighted_y_score = 0.0
    weighted_exact_fraction = 0.0
    x_step_ratios: list[float] = []

    for name, gold_points in gold_series.items():
        predicted_points = predicted_series.get(name, [])
        weight = max(len(gold_points), 1)
        weighted_y_score += order_aligned_y_score(predicted_points, gold_points) * weight
        weighted_exact_fraction += exact_x_match_fraction(predicted_points, gold_points) * weight
        total_weight += weight

        ratio = mean_x_step_ratio(predicted_points, gold_points)
        if ratio is not None:
            x_step_ratios.append(ratio)

    return {
        "order_aligned_y_score": (weighted_y_score / total_weight) if total_weight else 0.0,
        "exact_x_match_fraction": (weighted_exact_fraction / total_weight) if total_weight else 0.0,
        "mean_x_step_ratio": statistics.mean(x_step_ratios) if x_step_ratios else None,
    }


def build_sample_record(raw_sample: dict[str, Any], sequence_index: int) -> dict[str, Any]:
    info = loads_repaired(raw_sample["info"])
    metrics = loads_repaired(raw_sample["metrics"])
    schema_version = info.get("schema_version", "v1")

    gold_chart = canonical_from_info(info)
    answer_text = extract_answer_text(raw_sample.get("completion", ""))
    predicted_chart = canonical_from_answer(answer_text, schema_version=schema_version)

    predicted_series = canonical_series_map(predicted_chart)
    gold_series = canonical_series_map(gold_chart)

    predicted_names = set(predicted_series)
    gold_names = {str(name).strip() for name in info.get("legend_names", []) if str(name).strip()}
    if not gold_names:
        gold_names = set(gold_series)

    format_reward = 1.0 if predicted_chart is not None else 0.0
    name_f1 = f1_score(predicted_names, gold_names) if predicted_chart is not None else 0.0
    count_ratio = (
        weighted_point_count_ratio(predicted_series, gold_series)
        if predicted_chart is not None
        else 0.0
    )
    current_value = (
        current_chart_point_value(
            predicted_series,
            gold_series,
            oks_k=OKS_K,
            oks_threshold=OKS_THRESHOLD,
        )
        if predicted_chart is not None
        else 0.0
    )
    legacy_value = legacy_chart_point_value(predicted_series, gold_series) if predicted_chart is not None else 0.0

    diagnostics = sample_diagnostics(predicted_series, gold_series)

    total_reward_current = (
        (FORMAT_WEIGHT * format_reward)
        + (NAME_WEIGHT * name_f1)
        + (COUNT_WEIGHT * count_ratio)
        + (POINT_WEIGHT * current_value)
    )
    total_reward_legacy = (
        (FORMAT_WEIGHT * format_reward)
        + (NAME_WEIGHT * name_f1)
        + (COUNT_WEIGHT * count_ratio)
        + (POINT_WEIGHT * legacy_value)
    )

    point_value_delta = abs(current_value - float(metrics.get("series_point_value", 0.0)))
    reward_delta = abs(total_reward_current - float(raw_sample.get("reward", 0.0)))

    return {
        "id": f"s{int(raw_sample['step']):03d}-{sequence_index:04d}",
        "index": sequence_index,
        "step": int(raw_sample["step"]),
        "problem_id": raw_sample.get("problem_id"),
        "sample_id": raw_sample.get("sample_id"),
        "created_at": raw_sample.get("created_at"),
        "image_id": info.get("image_id"),
        "file_name": info.get("file_name", f"image-{info.get('image_id', 'unknown')}"),
        "schema_version": schema_version,
        "gold": {
            "legend_names": sorted(gold_names),
            "chart": chart_to_data(gold_chart),
        },
        "predicted": {
            "chart": chart_to_data(predicted_chart),
            "parse_ok": predicted_chart is not None,
        },
        "logged": {
            "reward": float(raw_sample.get("reward", 0.0) or 0.0),
            "format_reward": float(metrics.get("format_reward_func", 0.0) or 0.0),
            "name_f1": float(metrics.get("series_name_f1", 0.0) or 0.0),
            "count_ratio": float(metrics.get("series_point_count_ratio", 0.0) or 0.0),
            "point_value": float(metrics.get("series_point_value", 0.0) or 0.0),
        },
        "computed": {
            "format_reward": format_reward,
            "name_f1": name_f1,
            "count_ratio": count_ratio,
            "point_value_current": current_value,
            "point_value_legacy": legacy_value,
            "reward_current": total_reward_current,
            "reward_legacy": total_reward_legacy,
            "point_value_delta_vs_logged": point_value_delta,
            "reward_delta_vs_logged": reward_delta,
        },
        "diagnostics": diagnostics,
    }


def build_dataset(run_id: str, *, num_per_step: int) -> dict[str, Any]:
    progress = prime_json(["rl", "progress", run_id])
    steps = [int(step) for step in progress.get("steps_with_samples", [])]

    samples: list[dict[str, Any]] = []
    step_counts: list[dict[str, int]] = []
    sequence_index = 0

    for step in steps:
        rollouts = prime_json(["rl", "rollouts", run_id, "-s", str(step), "-n", str(num_per_step)])
        raw_samples = rollouts.get("samples", [])
        step_counts.append({"step": step, "count": len(raw_samples)})
        for raw_sample in raw_samples:
            samples.append(build_sample_record(raw_sample, sequence_index=sequence_index))
            sequence_index += 1

    point_value_deltas = [sample["computed"]["point_value_delta_vs_logged"] for sample in samples]
    reward_deltas = [sample["computed"]["reward_delta_vs_logged"] for sample in samples]
    point_value_exact_matches = sum(delta <= 1e-9 for delta in point_value_deltas)
    reward_exact_matches = sum(delta <= 1e-9 for delta in reward_deltas)
    current_beats_legacy = sum(
        abs(sample["logged"]["point_value"] - sample["computed"]["point_value_current"])
        <= abs(sample["logged"]["point_value"] - sample["computed"]["point_value_legacy"])
        for sample in samples
    )

    return {
        "meta": {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "latest_step": int(progress.get("latest_step", 0)),
            "steps_with_samples": steps,
            "step_counts": step_counts,
            "sample_count": len(samples),
            "weights": {
                "format": FORMAT_WEIGHT,
                "name": NAME_WEIGHT,
                "count": COUNT_WEIGHT,
                "point_value": POINT_WEIGHT,
            },
            "current_value_reward": {
                "mode": "oks_strict",
                "oks_k": OKS_K,
                "oks_threshold": OKS_THRESHOLD,
                "normalization": "full_gold_chart_span",
                "matching": "nearest_labeled_gold_point",
                "dedupe": "unique_gold_recall",
            },
            "legacy_value_reward": {
                "mode": "exact_x",
                "matching": "exact_gold_x_then_y_penalty",
            },
            "validation": {
                "max_point_value_delta_vs_logged": max(point_value_deltas) if point_value_deltas else 0.0,
                "max_total_reward_delta_vs_logged": max(reward_deltas) if reward_deltas else 0.0,
                "point_value_matches_logged_within_1e-9": all(delta <= 1e-9 for delta in point_value_deltas),
                "reward_matches_logged_within_1e-9": all(delta <= 1e-9 for delta in reward_deltas),
                "point_value_exact_match_count": point_value_exact_matches,
                "reward_exact_match_count": reward_exact_matches,
                "point_value_mean_delta_vs_logged": statistics.mean(point_value_deltas) if point_value_deltas else 0.0,
                "point_value_median_delta_vs_logged": statistics.median(point_value_deltas) if point_value_deltas else 0.0,
                "current_closer_than_legacy_count": current_beats_legacy,
            },
        },
        "samples": samples,
    }


def build_html(dataset: dict[str, Any]) -> str:
    data_json = json.dumps(dataset, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Series Point Value Rollout Playground</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {
        theme: {
          extend: {
            fontFamily: {
              sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
              mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
            },
            colors: {
              ink: '#10212b',
              sand: '#f4efe6',
              ember: '#c8643b',
              ocean: '#147d7e',
              pine: '#21403f',
              sun: '#d4a338',
            },
            boxShadow: {
              floaty: '0 24px 80px rgba(15, 23, 42, 0.12)',
            },
          },
        },
      };
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <style>
      :root {
        color-scheme: light;
      }
      body {
        background:
          radial-gradient(circle at top left, rgba(20, 125, 126, 0.18), transparent 28%),
          radial-gradient(circle at top right, rgba(212, 163, 56, 0.16), transparent 24%),
          linear-gradient(180deg, #fbf7f0 0%, #f5efe6 54%, #efe6d8 100%);
      }
      .grid-noise {
        background-image:
          linear-gradient(rgba(16, 33, 43, 0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(16, 33, 43, 0.05) 1px, transparent 1px);
        background-size: 28px 28px;
        mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.9), transparent 92%);
      }
      input[type="range"] {
        accent-color: #147d7e;
      }
      .chart-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.72));
        backdrop-filter: blur(14px);
      }
      .sample-row:hover {
        background: rgba(20, 125, 126, 0.08);
      }
      .sample-row.is-selected {
        background: rgba(20, 125, 126, 0.14);
      }
    </style>
  </head>
  <body class="min-h-screen font-sans text-ink antialiased">
    <div class="fixed inset-0 -z-10 grid-noise"></div>
    <script id="playground-data" type="application/json">__DATA_JSON__</script>

    <main class="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
      <section class="relative overflow-hidden rounded-[2rem] border border-white/70 bg-white/70 px-6 py-6 shadow-floaty sm:px-8 lg:px-10">
        <div class="absolute inset-0 bg-gradient-to-br from-white/70 via-white/20 to-teal-100/30"></div>
        <div class="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div class="max-w-4xl">
            <p class="mb-3 inline-flex items-center gap-2 rounded-full border border-amber-300/60 bg-amber-100/80 px-3 py-1 font-mono text-xs font-medium uppercase tracking-[0.25em] text-amber-900">
              Real Prime RL Rollouts
            </p>
            <h1 class="max-w-3xl text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
              How harsh is <span class="text-ember">`series_point_value`</span> on real rollout samples?
            </h1>
            <p class="mt-4 max-w-3xl text-sm leading-6 text-slate-700 sm:text-base">
              This page replays saved rollout samples from run <span class="font-mono text-xs sm:text-sm">__RUN_ID__</span>.
              The default controls follow the current checked-in reward in this repo: strict OKS, <span class="font-mono">k = __DEFAULT_OKS_K__</span>,
              threshold <span class="font-mono">__DEFAULT_OKS_THRESHOLD__</span>, full-chart span normalization, and unique matched-gold recall.
            </p>
            <p class="mt-3 max-w-3xl text-xs leading-5 text-slate-600 sm:text-sm">
              Replay check: <span class="font-mono">__POINT_VALUE_EXACT_MATCH_COUNT__/__SAMPLE_COUNT__</span> samples match the logged rollout point-value exactly with the current local code,
              median delta <span class="font-mono">__POINT_VALUE_MEDIAN_DELTA__</span>, max delta <span class="font-mono">__POINT_VALUE_MAX_DELTA__</span>.
              The page keeps both <span class="font-medium">logged</span> and <span class="font-medium">simulated</span> numbers visible so you can separate the historical run score from the current local replay when they drift.
            </p>
          </div>

          <div class="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:w-[560px]">
            <div class="rounded-2xl border border-white/70 bg-white/70 p-4">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Samples</div>
              <div id="hero-sample-count" class="mt-2 text-2xl font-semibold text-slate-900">-</div>
            </div>
            <div class="rounded-2xl border border-white/70 bg-white/70 p-4">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Steps</div>
              <div id="hero-step-count" class="mt-2 text-2xl font-semibold text-slate-900">-</div>
            </div>
            <div class="rounded-2xl border border-white/70 bg-white/70 p-4">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Latest Step</div>
              <div id="hero-latest-step" class="mt-2 text-2xl font-semibold text-slate-900">-</div>
            </div>
            <div class="rounded-2xl border border-white/70 bg-white/70 p-4">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Built</div>
              <div id="hero-generated-at" class="mt-2 text-sm font-medium text-slate-900">-</div>
            </div>
          </div>
        </div>
      </section>

      <section class="mt-6 grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
        <aside class="chart-card rounded-[1.75rem] border border-white/70 p-5 shadow-floaty">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold text-slate-900">Controls</h2>
            <button id="reset-button" class="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:border-slate-400 hover:text-slate-900">
              Reset
            </button>
          </div>

          <div class="mt-5">
            <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Presets</div>
            <div class="mt-3 grid grid-cols-2 gap-2">
              <button data-preset="current" class="preset-btn rounded-2xl border border-teal-300 bg-teal-50 px-3 py-2 text-left text-sm font-medium text-teal-900 transition hover:border-teal-400">Current live</button>
              <button data-preset="gentle" class="preset-btn rounded-2xl border border-amber-300 bg-amber-50 px-3 py-2 text-left text-sm font-medium text-amber-900 transition hover:border-amber-400">More forgiving</button>
              <button data-preset="wide" class="preset-btn rounded-2xl border border-orange-300 bg-orange-50 px-3 py-2 text-left text-sm font-medium text-orange-900 transition hover:border-orange-400">Wide OKS</button>
              <button data-preset="legacy" class="preset-btn rounded-2xl border border-slate-300 bg-slate-50 px-3 py-2 text-left text-sm font-medium text-slate-900 transition hover:border-slate-400">Legacy exact-x</button>
            </div>
          </div>

          <div class="mt-6 space-y-5">
            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <label for="value-mode" class="text-sm font-medium text-slate-900">Value reward mode</label>
              <select id="value-mode" class="mt-2 w-full rounded-2xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-teal-500">
                <option value="oks">Current strict OKS</option>
                <option value="legacy_exact_x">Legacy exact-x</option>
              </select>
              <p class="mt-2 text-xs leading-5 text-slate-600">
                Use <span class="font-mono">legacy exact-x</span> to compare against the pre-OKS reward that only gave credit when predicted and gold <span class="font-mono">x</span> matched exactly.
              </p>
            </div>

            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <div class="flex items-center justify-between text-sm font-medium text-slate-900">
                <label for="oks-k">OKS k</label>
                <span id="oks-k-value" class="font-mono text-xs text-slate-600">-</span>
              </div>
              <input id="oks-k" type="range" min="0.005" max="0.12" step="0.001" class="mt-3 w-full" />
              <p class="mt-2 text-xs leading-5 text-slate-600">
                Larger <span class="font-mono">k</span> makes the point tolerance wider before the score falls off.
              </p>
            </div>

            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <div class="flex items-center justify-between text-sm font-medium text-slate-900">
                <label for="oks-threshold">OKS threshold</label>
                <span id="oks-threshold-value" class="font-mono text-xs text-slate-600">-</span>
              </div>
              <input id="oks-threshold" type="range" min="0.00" max="0.99" step="0.01" class="mt-3 w-full" />
              <p class="mt-2 text-xs leading-5 text-slate-600">
                Lower thresholds accept more near-miss predicted points as matched gold points.
              </p>
            </div>
          </div>

          <div class="mt-6 space-y-5">
            <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Filters</div>

            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <label for="step-filter" class="text-sm font-medium text-slate-900">Sampled step</label>
              <select id="step-filter" class="mt-2 w-full rounded-2xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-teal-500"></select>
            </div>

            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <div class="flex items-center justify-between text-sm font-medium text-slate-900">
                <label for="name-filter">Min name F1</label>
                <span id="name-filter-value" class="font-mono text-xs text-slate-600">-</span>
              </div>
              <input id="name-filter" type="range" min="0" max="1" step="0.05" class="mt-3 w-full" />
            </div>

            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <div class="flex items-center justify-between text-sm font-medium text-slate-900">
                <label for="count-filter">Min count ratio</label>
                <span id="count-filter-value" class="font-mono text-xs text-slate-600">-</span>
              </div>
              <input id="count-filter" type="range" min="0" max="1" step="0.05" class="mt-3 w-full" />
            </div>

            <div class="rounded-3xl border border-slate-200/80 bg-white/80 p-4">
              <label for="search-filter" class="text-sm font-medium text-slate-900">Search title, file, or series</label>
              <input id="search-filter" type="search" placeholder="e.g. 66016 or ivogn" class="mt-2 w-full rounded-2xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-teal-500" />
              <label class="mt-3 flex items-center gap-3 text-sm text-slate-700">
                <input id="near-miss-only" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-teal-700 focus:ring-teal-600" />
                Only near-misses: perfect names and count ratio ≥ 0.8
              </label>
            </div>
          </div>
        </aside>

        <div class="space-y-6">
          <section class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div class="chart-card rounded-[1.5rem] border border-white/70 p-5 shadow-floaty">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Filtered samples</div>
              <div id="summary-filtered-count" class="mt-2 text-3xl font-semibold text-slate-900">-</div>
              <p class="mt-2 text-sm text-slate-600">Current table slice after filters.</p>
            </div>
            <div class="chart-card rounded-[1.5rem] border border-white/70 p-5 shadow-floaty">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Mean point value</div>
              <div class="mt-2 flex items-end gap-3">
                <div id="summary-mean-simulated" class="text-3xl font-semibold text-slate-900">-</div>
                <div id="summary-mean-delta" class="rounded-full px-2.5 py-1 text-xs font-medium">-</div>
              </div>
              <p class="mt-2 text-sm text-slate-600">Compared with logged rollout score.</p>
            </div>
            <div class="chart-card rounded-[1.5rem] border border-white/70 p-5 shadow-floaty">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Zero-score rate</div>
              <div class="mt-2 flex items-end gap-3">
                <div id="summary-zero-rate" class="text-3xl font-semibold text-slate-900">-</div>
                <div id="summary-zero-delta" class="rounded-full px-2.5 py-1 text-xs font-medium">-</div>
              </div>
              <p class="mt-2 text-sm text-slate-600">How often value reward fully collapses.</p>
            </div>
            <div class="chart-card rounded-[1.5rem] border border-white/70 p-5 shadow-floaty">
              <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Mean total reward</div>
              <div class="mt-2 flex items-end gap-3">
                <div id="summary-total-reward" class="text-3xl font-semibold text-slate-900">-</div>
                <div id="summary-total-delta" class="rounded-full px-2.5 py-1 text-xs font-medium">-</div>
              </div>
              <p class="mt-2 text-sm text-slate-600">Uses the fixed rubric weights from the env.</p>
            </div>
          </section>

          <section class="grid gap-6 xl:grid-cols-2">
            <div class="chart-card rounded-[1.75rem] border border-white/70 p-5 shadow-floaty">
              <div class="flex items-center justify-between">
                <div>
                  <h2 class="text-lg font-semibold text-slate-900">Step Trend</h2>
                  <p class="mt-1 text-sm text-slate-600">Average value reward at each sampled RL step.</p>
                </div>
                <div class="flex gap-2 text-xs">
                  <span class="rounded-full bg-amber-100 px-3 py-1 font-medium text-amber-900">Logged</span>
                  <span class="rounded-full bg-teal-100 px-3 py-1 font-medium text-teal-900">Current controls</span>
                </div>
              </div>
              <div id="trend-chart" class="mt-4 h-[280px] w-full overflow-hidden rounded-[1.25rem] border border-slate-200 bg-slate-50"></div>
            </div>

            <div class="chart-card rounded-[1.75rem] border border-white/70 p-5 shadow-floaty">
              <div class="flex items-center justify-between">
                <div>
                  <h2 class="text-lg font-semibold text-slate-900">Value Distribution</h2>
                  <p class="mt-1 text-sm text-slate-600">Histogram of logged versus simulated point-value scores.</p>
                </div>
                <div class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700">
                  20 bins
                </div>
              </div>
              <div id="histogram-chart" class="mt-4 h-[280px] w-full overflow-hidden rounded-[1.25rem] border border-slate-200 bg-slate-50"></div>
            </div>
          </section>

          <section class="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(420px,1.05fr)]">
            <div class="chart-card rounded-[1.75rem] border border-white/70 p-5 shadow-floaty">
              <div class="flex items-center justify-between gap-4">
                <div>
                  <h2 class="text-lg font-semibold text-slate-900">Samples</h2>
                  <p class="mt-1 text-sm text-slate-600">Sorted by the biggest change from the logged value reward.</p>
                </div>
                <div id="table-caption" class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700">-</div>
              </div>
              <div class="mt-4 overflow-hidden rounded-[1.25rem] border border-slate-200">
                <div class="max-h-[720px] overflow-auto">
                  <table class="min-w-full divide-y divide-slate-200 bg-white/80 text-left text-sm">
                    <thead class="sticky top-0 bg-slate-100/95 backdrop-blur">
                      <tr class="text-xs uppercase tracking-[0.16em] text-slate-600">
                        <th class="px-3 py-3 font-medium">Sample</th>
                        <th class="px-3 py-3 font-medium">Step</th>
                        <th class="px-3 py-3 font-medium">Logged</th>
                        <th class="px-3 py-3 font-medium">Simulated</th>
                        <th class="px-3 py-3 font-medium">Delta</th>
                        <th class="px-3 py-3 font-medium">Name</th>
                        <th class="px-3 py-3 font-medium">Count</th>
                        <th class="px-3 py-3 font-medium">Exact x</th>
                      </tr>
                    </thead>
                    <tbody id="sample-table-body" class="divide-y divide-slate-200"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div class="chart-card rounded-[1.75rem] border border-white/70 p-5 shadow-floaty">
              <div class="flex items-start justify-between gap-4">
                <div>
                  <h2 class="text-lg font-semibold text-slate-900">Selected Sample</h2>
                  <p id="detail-subtitle" class="mt-1 text-sm text-slate-600">Choose a sample from the table.</p>
                </div>
                <div id="detail-badges" class="flex flex-wrap justify-end gap-2"></div>
              </div>

              <div class="mt-4 grid gap-4 lg:grid-cols-3">
                <div class="rounded-[1.25rem] border border-slate-200 bg-white/85 p-4">
                  <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Logged value</div>
                  <div id="detail-logged-value" class="mt-2 text-2xl font-semibold text-slate-900">-</div>
                </div>
                <div class="rounded-[1.25rem] border border-slate-200 bg-white/85 p-4">
                  <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Simulated value</div>
                  <div id="detail-simulated-value" class="mt-2 text-2xl font-semibold text-slate-900">-</div>
                </div>
                <div class="rounded-[1.25rem] border border-slate-200 bg-white/85 p-4">
                  <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Value delta</div>
                  <div id="detail-value-delta" class="mt-2 text-2xl font-semibold text-slate-900">-</div>
                </div>
              </div>

              <div class="mt-4 overflow-hidden rounded-[1.5rem] border border-slate-200 bg-slate-50">
                <div id="overlay-chart" class="h-[380px] w-full"></div>
              </div>

              <div class="mt-4 grid gap-4 lg:grid-cols-2">
                <div class="rounded-[1.25rem] border border-slate-200 bg-white/85 p-4">
                  <h3 class="text-sm font-semibold text-slate-900">Reward Breakdown</h3>
                  <div id="detail-breakdown" class="mt-3 overflow-hidden rounded-2xl border border-slate-200"></div>
                </div>
                <div class="rounded-[1.25rem] border border-slate-200 bg-white/85 p-4">
                  <h3 class="text-sm font-semibold text-slate-900">Diagnostics</h3>
                  <dl id="detail-diagnostics" class="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 text-sm"></dl>
                </div>
              </div>

              <div class="mt-4 rounded-[1.25rem] border border-slate-200 bg-white/85 p-4">
                <div class="flex items-center justify-between">
                  <h3 class="text-sm font-semibold text-slate-900">Per-series detail</h3>
                  <div class="text-xs text-slate-500">Gold names are the scoring anchors.</div>
                </div>
                <div id="detail-series-table" class="mt-3 overflow-hidden rounded-2xl border border-slate-200"></div>
              </div>
            </div>
          </section>
        </div>
      </section>
    </main>

    <script>
      const PLAYGROUND = JSON.parse(document.getElementById('playground-data').textContent);

      const state = {
        valueMode: 'oks',
        oksK: PLAYGROUND.meta.current_value_reward.oks_k,
        oksThreshold: PLAYGROUND.meta.current_value_reward.oks_threshold,
        step: 'all',
        minNameF1: 0,
        minCountRatio: 0,
        search: '',
        nearMissOnly: false,
        selectedId: PLAYGROUND.samples[0] ? PLAYGROUND.samples[0].id : null,
      };

      const elements = {
        heroSampleCount: document.getElementById('hero-sample-count'),
        heroStepCount: document.getElementById('hero-step-count'),
        heroLatestStep: document.getElementById('hero-latest-step'),
        heroGeneratedAt: document.getElementById('hero-generated-at'),
        resetButton: document.getElementById('reset-button'),
        presetButtons: [...document.querySelectorAll('.preset-btn')],
        valueMode: document.getElementById('value-mode'),
        oksK: document.getElementById('oks-k'),
        oksKValue: document.getElementById('oks-k-value'),
        oksThreshold: document.getElementById('oks-threshold'),
        oksThresholdValue: document.getElementById('oks-threshold-value'),
        stepFilter: document.getElementById('step-filter'),
        nameFilter: document.getElementById('name-filter'),
        nameFilterValue: document.getElementById('name-filter-value'),
        countFilter: document.getElementById('count-filter'),
        countFilterValue: document.getElementById('count-filter-value'),
        searchFilter: document.getElementById('search-filter'),
        nearMissOnly: document.getElementById('near-miss-only'),
        summaryFilteredCount: document.getElementById('summary-filtered-count'),
        summaryMeanSimulated: document.getElementById('summary-mean-simulated'),
        summaryMeanDelta: document.getElementById('summary-mean-delta'),
        summaryZeroRate: document.getElementById('summary-zero-rate'),
        summaryZeroDelta: document.getElementById('summary-zero-delta'),
        summaryTotalReward: document.getElementById('summary-total-reward'),
        summaryTotalDelta: document.getElementById('summary-total-delta'),
        trendChart: document.getElementById('trend-chart'),
        histogramChart: document.getElementById('histogram-chart'),
        tableBody: document.getElementById('sample-table-body'),
        tableCaption: document.getElementById('table-caption'),
        detailSubtitle: document.getElementById('detail-subtitle'),
        detailBadges: document.getElementById('detail-badges'),
        detailLoggedValue: document.getElementById('detail-logged-value'),
        detailSimulatedValue: document.getElementById('detail-simulated-value'),
        detailValueDelta: document.getElementById('detail-value-delta'),
        overlayChart: document.getElementById('overlay-chart'),
        detailBreakdown: document.getElementById('detail-breakdown'),
        detailDiagnostics: document.getElementById('detail-diagnostics'),
        detailSeriesTable: document.getElementById('detail-series-table'),
      };

      const DEFAULT_PRESETS = {
        current: {
          valueMode: 'oks',
          oksK: PLAYGROUND.meta.current_value_reward.oks_k,
          oksThreshold: PLAYGROUND.meta.current_value_reward.oks_threshold,
        },
        gentle: {
          valueMode: 'oks',
          oksK: Math.max(PLAYGROUND.meta.current_value_reward.oks_k, 0.04),
          oksThreshold: 0.4,
        },
        wide: {
          valueMode: 'oks',
          oksK: 0.07,
          oksThreshold: 0.3,
        },
        legacy: {
          valueMode: 'legacy_exact_x',
          oksK: PLAYGROUND.meta.current_value_reward.oks_k,
          oksThreshold: PLAYGROUND.meta.current_value_reward.oks_threshold,
        },
      };

      function fmt(value, digits = 3) {
        if (value === null || value === undefined || Number.isNaN(value)) {
          return 'n/a';
        }
        return Number(value).toFixed(digits);
      }

      function escapeHtml(value) {
        return String(value ?? '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#39;');
      }

      function pct(value, digits = 1) {
        if (value === null || value === undefined || Number.isNaN(value)) {
          return 'n/a';
        }
        return `${(Number(value) * 100).toFixed(digits)}%`;
      }

      function mean(values) {
        if (!values.length) return 0;
        return values.reduce((sum, value) => sum + value, 0) / values.length;
      }

      function median(values) {
        if (!values.length) return null;
        const sorted = [...values].sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 1) return sorted[middle];
        return (sorted[middle - 1] + sorted[middle]) / 2;
      }

      function unique(values) {
        return [...new Set(values)];
      }

      function sampleSearchText(sample) {
        const goldSeriesNames = (sample.gold.chart?.series || []).map((series) => series.name).join(' ');
        const predSeriesNames = (sample.predicted.chart?.series || []).map((series) => series.name).join(' ');
        return [
          sample.file_name,
          sample.image_id,
          sample.gold.chart?.title || '',
          goldSeriesNames,
          predSeriesNames,
        ].join(' ').toLowerCase();
      }

      function seriesMap(chart) {
        const map = new Map();
        if (!chart || !chart.series) return map;
        for (const series of chart.series) {
          if (series && series.name) {
            map.set(series.name, series.points || []);
          }
        }
        return map;
      }

      function pointCountRatio(predictedPoints, goldPoints) {
        const predictedCount = predictedPoints.length;
        const goldCount = goldPoints.length;
        if (predictedCount === 0 && goldCount === 0) return 1;
        if (predictedCount === 0 || goldCount === 0) return 0;
        return Math.min(predictedCount, goldCount) / Math.max(predictedCount, goldCount);
      }

      function weightedPointCountRatio(predictedSeries, goldSeries) {
        if (goldSeries.size === 0) return predictedSeries.size === 0 ? 1 : 0;
        let weighted = 0;
        let totalWeight = 0;
        for (const [name, goldPoints] of goldSeries.entries()) {
          const predictedPoints = predictedSeries.get(name) || [];
          const weight = Math.max(goldPoints.length, 1);
          weighted += pointCountRatio(predictedPoints, goldPoints) * weight;
          totalWeight += weight;
        }
        return totalWeight ? weighted / totalWeight : 0;
      }

      function f1Score(predicted, gold) {
        if (gold.size === 0 && predicted.size === 0) return 1;
        if (gold.size === 0 || predicted.size === 0) return 0;

        let truePositives = 0;
        for (const value of predicted) {
          if (gold.has(value)) truePositives += 1;
        }
        if (truePositives === 0) return 0;

        const precision = truePositives / predicted.size;
        const recall = truePositives / gold.size;
        return (2 * precision * recall) / (precision + recall);
      }

      function chartBounds(goldSeries) {
        const xs = [];
        const ys = [];
        for (const points of goldSeries.values()) {
          for (const [x, y] of points) {
            xs.push(Number(x));
            ys.push(Number(y));
          }
        }
        if (!xs.length || !ys.length) {
          return { xMin: 0, xScale: 1, yMin: 0, yScale: 1 };
        }
        const xMin = Math.min(...xs);
        const yMin = Math.min(...ys);
        return {
          xMin,
          xScale: Math.max(Math.max(...xs) - xMin, 1),
          yMin,
          yScale: Math.max(Math.max(...ys) - yMin, 1),
        };
      }

      function normalizePoint(point, bounds) {
        return [
          (Number(point[0]) - bounds.xMin) / bounds.xScale,
          (Number(point[1]) - bounds.yMin) / bounds.yScale,
        ];
      }

      function nearestGoldIndex(predictedPoint, goldPoints) {
        let bestDistance = Infinity;
        let bestIndex = -1;
        for (let index = 0; index < goldPoints.length; index += 1) {
          const goldPoint = goldPoints[index];
          const dx = predictedPoint[0] - goldPoint[0];
          const dy = predictedPoint[1] - goldPoint[1];
          const distance = Math.hypot(dx, dy);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
          }
        }
        return { distance: bestDistance, index: bestIndex };
      }

      function oks(distance, k) {
        return Math.exp(-(distance ** 2) / (2 * (k ** 2)));
      }

      function currentSeriesValueDetail(predictedPoints, goldPoints, params, bounds) {
        if (predictedPoints.length === 0 && goldPoints.length === 0) {
          return { score: 1, matchedGold: 0, goldCount: 0, predCount: 0 };
        }
        if (goldPoints.length === 0) {
          return { score: predictedPoints.length === 0 ? 1 : 0, matchedGold: 0, goldCount: 0, predCount: predictedPoints.length };
        }
        if (predictedPoints.length === 0) {
          return { score: 0, matchedGold: 0, goldCount: goldPoints.length, predCount: 0 };
        }

        const normalizedGold = goldPoints.map((point) => normalizePoint(point, bounds));
        const normalizedPredicted = predictedPoints.map((point) => normalizePoint(point, bounds));
        const matched = new Set();

        for (const predictedPoint of normalizedPredicted) {
          const nearest = nearestGoldIndex(predictedPoint, normalizedGold);
          if (nearest.index < 0) continue;
          if (oks(nearest.distance, params.oksK) > params.oksThreshold) {
            matched.add(nearest.index);
          }
        }

        return {
          score: matched.size / goldPoints.length,
          matchedGold: matched.size,
          goldCount: goldPoints.length,
          predCount: predictedPoints.length,
        };
      }

      function legacySeriesValueDetail(predictedPoints, goldPoints) {
        if (predictedPoints.length === 0 && goldPoints.length === 0) {
          return { score: 1, matchedGold: 0, goldCount: 0, predCount: 0 };
        }
        if (goldPoints.length === 0) {
          return { score: predictedPoints.length === 0 ? 1 : 0, matchedGold: 0, goldCount: 0, predCount: predictedPoints.length };
        }
        if (predictedPoints.length === 0) {
          return { score: 0, matchedGold: 0, goldCount: goldPoints.length, predCount: 0 };
        }

        const predictedYByX = new Map(predictedPoints.map((point) => [Number(point[0]), Number(point[1])]));
        const goldYs = goldPoints.map((point) => Number(point[1]));
        const yScale = Math.max(Math.max(...goldYs) - Math.min(...goldYs), 1);

        let totalScore = 0;
        let matchedGold = 0;
        for (const [goldX, goldY] of goldPoints) {
          if (!predictedYByX.has(Number(goldX))) continue;
          matchedGold += 1;
          const predictedY = predictedYByX.get(Number(goldX));
          const normalizedError = Math.abs(predictedY - Number(goldY)) / yScale;
          totalScore += Math.max(0, 1 - normalizedError);
        }

        return {
          score: totalScore / goldPoints.length,
          matchedGold,
          goldCount: goldPoints.length,
          predCount: predictedPoints.length,
        };
      }

      function pointValueChartDetail(sample, params) {
        const predictedSeries = seriesMap(sample.predicted.chart);
        const goldSeries = seriesMap(sample.gold.chart);
        if (goldSeries.size === 0) {
          return { score: predictedSeries.size === 0 ? 1 : 0, bySeries: [] };
        }

        const bounds = chartBounds(goldSeries);
        const details = [];
        let weighted = 0;
        let totalWeight = 0;

        for (const [name, goldPoints] of goldSeries.entries()) {
          const predictedPoints = predictedSeries.get(name) || [];
          const detail = params.valueMode === 'legacy_exact_x'
            ? legacySeriesValueDetail(predictedPoints, goldPoints)
            : currentSeriesValueDetail(predictedPoints, goldPoints, params, bounds);
          const weight = Math.max(goldPoints.length, 1);
          weighted += detail.score * weight;
          totalWeight += weight;
          details.push({
            name,
            score: detail.score,
            matchedGold: detail.matchedGold,
            goldCount: detail.goldCount,
            predCount: detail.predCount,
          });
        }

        return {
          score: totalWeight ? weighted / totalWeight : 0,
          bySeries: details,
        };
      }

      function sortPointPairs(points) {
        return [...points].map((point) => [Number(point[0]), Number(point[1])]).sort((a, b) => {
          if (a[0] !== b[0]) return a[0] - b[0];
          return a[1] - b[1];
        });
      }

      function orderAlignedYScore(predictedPoints, goldPoints) {
        const predicted = sortPointPairs(predictedPoints);
        const gold = sortPointPairs(goldPoints);
        if (gold.length === 0 && predicted.length === 0) return 1;
        if (gold.length === 0 || predicted.length === 0) return 0;

        const goldYs = gold.map((point) => point[1]);
        const yScale = Math.max(Math.max(...goldYs) - Math.min(...goldYs), 1);
        const matchedCount = Math.min(predicted.length, gold.length);
        let totalScore = 0;
        for (let index = 0; index < matchedCount; index += 1) {
          totalScore += Math.max(0, 1 - Math.abs(predicted[index][1] - gold[index][1]) / yScale);
        }
        return totalScore / gold.length;
      }

      function meanXStepRatio(predictedPoints, goldPoints) {
        const predicted = sortPointPairs(predictedPoints);
        const gold = sortPointPairs(goldPoints);
        if (predicted.length !== gold.length || gold.length < 2) return null;

        const goldSteps = [];
        for (let index = 0; index < gold.length - 1; index += 1) {
          const step = Math.abs(gold[index + 1][0] - gold[index][0]);
          if (step !== 0) goldSteps.push(step);
        }
        if (!goldSteps.length) return null;
        const referenceStep = median(goldSteps);
        if (!referenceStep) return null;

        const offsets = gold.map((point, index) => Math.abs(predicted[index][0] - point[0]));
        return mean(offsets) / referenceStep;
      }

      function exactXMatchFraction(predictedPoints, goldPoints) {
        const predictedXs = new Set(predictedPoints.map((point) => Number(point[0])));
        const goldXs = goldPoints.map((point) => Number(point[0]));
        if (!goldXs.length) return predictedXs.size === 0 ? 1 : 0;
        let exact = 0;
        for (const goldX of goldXs) {
          if (predictedXs.has(goldX)) exact += 1;
        }
        return exact / goldXs.length;
      }

      function sampleDiagnostics(sample) {
        const predictedSeries = seriesMap(sample.predicted.chart);
        const goldSeries = seriesMap(sample.gold.chart);
        const yScores = [];
        const xRatios = [];
        const exactFractions = [];
        const weights = [];

        for (const [name, goldPoints] of goldSeries.entries()) {
          const predictedPoints = predictedSeries.get(name) || [];
          const weight = Math.max(goldPoints.length, 1);
          yScores.push(orderAlignedYScore(predictedPoints, goldPoints));
          exactFractions.push(exactXMatchFraction(predictedPoints, goldPoints));
          weights.push(weight);
          const ratio = meanXStepRatio(predictedPoints, goldPoints);
          if (ratio !== null) xRatios.push(ratio);
        }

        const weightedAverage = (values) => {
          if (!values.length || !weights.length) return 0;
          let total = 0;
          let totalWeight = 0;
          for (let index = 0; index < values.length; index += 1) {
            total += values[index] * weights[index];
            totalWeight += weights[index];
          }
          return totalWeight ? total / totalWeight : 0;
        };

        return {
          orderAlignedYScore: weightedAverage(yScores),
          exactXMatchFraction: weightedAverage(exactFractions),
          meanXStepRatio: xRatios.length ? mean(xRatios) : null,
        };
      }

      function computeSample(sample, params) {
        const predictedSeries = seriesMap(sample.predicted.chart);
        const goldSeries = seriesMap(sample.gold.chart);
        const predictedNames = new Set([...predictedSeries.keys()]);
        const goldNames = new Set(sample.gold.legend_names || [...goldSeries.keys()]);

        const formatReward = sample.predicted.parse_ok ? 1 : 0;
        const nameF1 = sample.predicted.parse_ok ? f1Score(predictedNames, goldNames) : 0;
        const countRatio = sample.predicted.parse_ok ? weightedPointCountRatio(predictedSeries, goldSeries) : 0;
        const pointValueDetail = sample.predicted.parse_ok ? pointValueChartDetail(sample, params) : { score: 0, bySeries: [] };
        const totalReward = (1 * formatReward) + (1 * nameF1) + (2 * countRatio) + (2 * pointValueDetail.score);

        return {
          sample,
          pointValue: pointValueDetail.score,
          totalReward,
          formatReward,
          nameF1,
          countRatio,
          valueDelta: pointValueDetail.score - sample.logged.point_value,
          totalRewardDelta: totalReward - sample.logged.reward,
          pointValueDetail,
          diagnostics: sampleDiagnostics(sample),
        };
      }

      function filteredSamples(computedSamples) {
        const query = state.search.trim().toLowerCase();
        return computedSamples
          .filter((entry) => state.step === 'all' || entry.sample.step === Number(state.step))
          .filter((entry) => entry.nameF1 >= state.minNameF1)
          .filter((entry) => entry.countRatio >= state.minCountRatio)
          .filter((entry) => !state.nearMissOnly || (entry.nameF1 >= 0.999 && entry.countRatio >= 0.8))
          .filter((entry) => !query || sampleSearchText(entry.sample).includes(query))
          .sort((a, b) => Math.abs(b.valueDelta) - Math.abs(a.valueDelta) || b.sample.step - a.sample.step);
      }

      function badgeClass(delta) {
        if (delta > 0.0005) {
          return 'bg-teal-100 text-teal-900 border border-teal-200';
        }
        if (delta < -0.0005) {
          return 'bg-amber-100 text-amber-900 border border-amber-200';
        }
        return 'bg-slate-100 text-slate-700 border border-slate-200';
      }

      function renderDeltaChip(element, delta, prefix = '') {
        element.className = `rounded-full px-2.5 py-1 text-xs font-medium ${badgeClass(delta)}`;
        const sign = delta > 0 ? '+' : '';
        element.textContent = `${prefix}${sign}${delta.toFixed(3)}`;
      }

      function renderHero() {
        elements.heroSampleCount.textContent = PLAYGROUND.meta.sample_count;
        elements.heroStepCount.textContent = PLAYGROUND.meta.steps_with_samples.length;
        elements.heroLatestStep.textContent = PLAYGROUND.meta.latest_step;
        elements.heroGeneratedAt.textContent = new Date(PLAYGROUND.meta.generated_at).toLocaleString();
      }

      function renderControls() {
        elements.valueMode.value = state.valueMode;
        elements.oksK.value = state.oksK;
        elements.oksThreshold.value = state.oksThreshold;
        elements.oksKValue.textContent = fmt(state.oksK, 3);
        elements.oksThresholdValue.textContent = fmt(state.oksThreshold, 2);
        elements.nameFilter.value = state.minNameF1;
        elements.nameFilterValue.textContent = fmt(state.minNameF1, 2);
        elements.countFilter.value = state.minCountRatio;
        elements.countFilterValue.textContent = fmt(state.minCountRatio, 2);
        elements.searchFilter.value = state.search;
        elements.nearMissOnly.checked = state.nearMissOnly;
        const disableOKS = state.valueMode !== 'oks';
        elements.oksK.disabled = disableOKS;
        elements.oksThreshold.disabled = disableOKS;
        elements.oksK.classList.toggle('opacity-50', disableOKS);
        elements.oksThreshold.classList.toggle('opacity-50', disableOKS);
      }

      function populateStepFilter() {
        elements.stepFilter.innerHTML = '';
        const allOption = document.createElement('option');
        allOption.value = 'all';
        allOption.textContent = 'All sampled steps';
        elements.stepFilter.appendChild(allOption);

        for (const step of PLAYGROUND.meta.steps_with_samples) {
          const option = document.createElement('option');
          option.value = String(step);
          option.textContent = `Step ${step}`;
          elements.stepFilter.appendChild(option);
        }
        elements.stepFilter.value = state.step;
      }

      function renderSummary(filtered) {
        const loggedValues = filtered.map((entry) => entry.sample.logged.point_value);
        const simulatedValues = filtered.map((entry) => entry.pointValue);
        const loggedRewards = filtered.map((entry) => entry.sample.logged.reward);
        const simulatedRewards = filtered.map((entry) => entry.totalReward);

        const meanLoggedValue = mean(loggedValues);
        const meanSimulatedValue = mean(simulatedValues);
        const meanLoggedReward = mean(loggedRewards);
        const meanSimulatedReward = mean(simulatedRewards);
        const loggedZeroRate = loggedValues.length ? loggedValues.filter((value) => value <= 1e-9).length / loggedValues.length : 0;
        const simulatedZeroRate = simulatedValues.length ? simulatedValues.filter((value) => value <= 1e-9).length / simulatedValues.length : 0;

        elements.summaryFilteredCount.textContent = filtered.length;
        elements.summaryMeanSimulated.textContent = fmt(meanSimulatedValue, 3);
        renderDeltaChip(elements.summaryMeanDelta, meanSimulatedValue - meanLoggedValue, 'vs logged ');
        elements.summaryZeroRate.textContent = pct(simulatedZeroRate, 1);
        renderDeltaChip(elements.summaryZeroDelta, simulatedZeroRate - loggedZeroRate, 'vs logged ');
        elements.summaryTotalReward.textContent = fmt(meanSimulatedReward, 3);
        renderDeltaChip(elements.summaryTotalDelta, meanSimulatedReward - meanLoggedReward, 'vs logged ');
      }

      function svgShell(width, height, inner) {
        return `
          <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
            ${inner}
          </svg>
        `;
      }

      function renderTrendChart(filtered) {
        const byStep = new Map();
        for (const entry of filtered) {
          if (!byStep.has(entry.sample.step)) {
            byStep.set(entry.sample.step, { logged: [], simulated: [] });
          }
          byStep.get(entry.sample.step).logged.push(entry.sample.logged.point_value);
          byStep.get(entry.sample.step).simulated.push(entry.pointValue);
        }

        const steps = [...byStep.keys()].sort((a, b) => a - b);
        if (!steps.length) {
          elements.trendChart.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-500">No samples match the current filters.</div>';
          return;
        }

        const points = steps.map((step) => ({
          step,
          logged: mean(byStep.get(step).logged),
          simulated: mean(byStep.get(step).simulated),
        }));

        const width = 760;
        const height = 280;
        const margin = { top: 20, right: 18, bottom: 32, left: 44 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const xMin = Math.min(...steps);
        const xMax = Math.max(...steps);
        const yMax = Math.max(0.12, ...points.flatMap((point) => [point.logged, point.simulated]));

        const xScale = (step) => {
          if (xMax === xMin) return margin.left + plotWidth / 2;
          return margin.left + ((step - xMin) / (xMax - xMin)) * plotWidth;
        };
        const yScale = (value) => margin.top + plotHeight - ((value / yMax) * plotHeight);

        const linePath = (key) => points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${xScale(point.step).toFixed(2)} ${yScale(point[key]).toFixed(2)}`).join(' ');

        const gridLines = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = margin.top + ratio * plotHeight;
          const label = ((1 - ratio) * yMax).toFixed(2);
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(15,23,42,0.12)" stroke-dasharray="4 6" />
            <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#475569">${label}</text>
          `;
        }).join('');

        const xLabels = points.map((point) => `
          <text x="${xScale(point.step)}" y="${height - 10}" text-anchor="middle" font-size="11" fill="#475569">${point.step}</text>
        `).join('');

        const loggedDots = points.map((point) => `
          <circle cx="${xScale(point.step)}" cy="${yScale(point.logged)}" r="4" fill="#c8643b" />
        `).join('');
        const simulatedDots = points.map((point) => `
          <circle cx="${xScale(point.step)}" cy="${yScale(point.simulated)}" r="4" fill="#147d7e" />
        `).join('');

        elements.trendChart.innerHTML = svgShell(width, height, `
          <rect x="0" y="0" width="${width}" height="${height}" fill="#f8fafc" />
          ${gridLines}
          <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />
          <path d="${linePath('logged')}" fill="none" stroke="#c8643b" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
          <path d="${linePath('simulated')}" fill="none" stroke="#147d7e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
          ${loggedDots}
          ${simulatedDots}
          ${xLabels}
          <text x="${width / 2}" y="${height - 2}" text-anchor="middle" font-size="12" fill="#475569">sampled RL step</text>
          <text x="14" y="${height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 14 ${height / 2})">mean point value</text>
        `);
      }

      function renderHistogram(filtered) {
        if (!filtered.length) {
          elements.histogramChart.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-500">No samples match the current filters.</div>';
          return;
        }

        const bins = 20;
        const loggedCounts = Array.from({ length: bins }, () => 0);
        const simulatedCounts = Array.from({ length: bins }, () => 0);
        const toIndex = (value) => Math.min(bins - 1, Math.max(0, Math.floor(value * bins)));

        for (const entry of filtered) {
          loggedCounts[toIndex(entry.sample.logged.point_value)] += 1;
          simulatedCounts[toIndex(entry.pointValue)] += 1;
        }

        const maxCount = Math.max(...loggedCounts, ...simulatedCounts, 1);
        const width = 760;
        const height = 280;
        const margin = { top: 18, right: 18, bottom: 34, left: 44 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const slotWidth = plotWidth / bins;
        const barWidth = Math.max(8, slotWidth * 0.36);
        const yScale = (count) => margin.top + plotHeight - (count / maxCount) * plotHeight;

        const gridLines = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = margin.top + ratio * plotHeight;
          const label = Math.round((1 - ratio) * maxCount);
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(15,23,42,0.12)" stroke-dasharray="4 6" />
            <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#475569">${label}</text>
          `;
        }).join('');

        const bars = loggedCounts.map((count, index) => {
          const xBase = margin.left + index * slotWidth;
          const loggedHeight = plotHeight - (yScale(count) - margin.top);
          const simulatedHeight = plotHeight - (yScale(simulatedCounts[index]) - margin.top);
          return `
            <rect x="${xBase + slotWidth * 0.1}" y="${yScale(count)}" width="${barWidth}" height="${loggedHeight}" rx="4" fill="#c8643b" fill-opacity="0.82" />
            <rect x="${xBase + slotWidth * 0.54}" y="${yScale(simulatedCounts[index])}" width="${barWidth}" height="${simulatedHeight}" rx="4" fill="#147d7e" fill-opacity="0.82" />
            ${index < bins ? `<text x="${xBase + slotWidth / 2}" y="${height - 10}" text-anchor="middle" font-size="10" fill="#475569">${(index / bins).toFixed(1)}</text>` : ''}
          `;
        }).join('');

        elements.histogramChart.innerHTML = svgShell(width, height, `
          <rect x="0" y="0" width="${width}" height="${height}" fill="#f8fafc" />
          ${gridLines}
          <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />
          ${bars}
          <text x="${width / 2}" y="${height - 2}" text-anchor="middle" font-size="12" fill="#475569">point value</text>
          <text x="14" y="${height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 14 ${height / 2})">sample count</text>
        `);
      }

      function renderTable(filtered) {
        elements.tableCaption.textContent = `${filtered.length} visible`;
        elements.tableBody.innerHTML = '';
        if (!filtered.length) {
          elements.tableBody.innerHTML = '<tr><td colspan="8" class="px-4 py-8 text-center text-sm text-slate-500">No samples match the current filters.</td></tr>';
          return;
        }

        for (const entry of filtered) {
          const row = document.createElement('tr');
          row.className = `sample-row cursor-pointer ${entry.sample.id === state.selectedId ? 'is-selected' : ''}`;
          row.dataset.sampleId = entry.sample.id;

          const sampleLabel = `
            <div class="font-medium text-slate-900">${escapeHtml(entry.sample.file_name)}</div>
            <div class="mt-1 font-mono text-xs text-slate-500">${escapeHtml(entry.sample.gold.chart?.title || 'untitled')} • ${escapeHtml(entry.sample.id)}</div>
          `;

          row.innerHTML = `
            <td class="px-3 py-3 align-top">${sampleLabel}</td>
            <td class="px-3 py-3 font-mono text-xs text-slate-700">${entry.sample.step}</td>
            <td class="px-3 py-3 font-mono text-xs text-slate-700">${fmt(entry.sample.logged.point_value, 3)}</td>
            <td class="px-3 py-3 font-mono text-xs text-slate-700">${fmt(entry.pointValue, 3)}</td>
            <td class="px-3 py-3"><span class="rounded-full px-2.5 py-1 text-xs font-medium ${badgeClass(entry.valueDelta)}">${entry.valueDelta > 0 ? '+' : ''}${fmt(entry.valueDelta, 3)}</span></td>
            <td class="px-3 py-3 font-mono text-xs text-slate-700">${fmt(entry.nameF1, 2)}</td>
            <td class="px-3 py-3 font-mono text-xs text-slate-700">${fmt(entry.countRatio, 2)}</td>
            <td class="px-3 py-3 font-mono text-xs text-slate-700">${fmt(entry.diagnostics.exactXMatchFraction, 2)}</td>
          `;

          row.addEventListener('click', () => {
            state.selectedId = entry.sample.id;
            refresh();
          });
          elements.tableBody.appendChild(row);
        }
      }

      function colorForSeries(name, index) {
        const palette = ['#147d7e', '#c8643b', '#d4a338', '#6b8e23', '#5a67d8', '#db2777', '#0f766e', '#9a3412'];
        const hash = [...name].reduce((acc, char) => acc + char.charCodeAt(0), 0);
        return palette[(hash + index) % palette.length];
      }

      function renderOverlayChart(entry) {
        if (!entry) {
          elements.overlayChart.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-500">Choose a sample from the table.</div>';
          return;
        }

        const goldSeries = entry.sample.gold.chart?.series || [];
        const predictedSeries = entry.sample.predicted.chart?.series || [];
        const allSeries = [...goldSeries, ...predictedSeries];
        const xs = allSeries.flatMap((series) => (series.points || []).map((point) => Number(point[0])));
        const ys = allSeries.flatMap((series) => (series.points || []).map((point) => Number(point[1])));

        if (!xs.length || !ys.length) {
          elements.overlayChart.innerHTML = '<div class="flex h-full items-center justify-center text-sm text-slate-500">No chart data to plot for this sample.</div>';
          return;
        }

        const width = 840;
        const height = 380;
        const margin = { top: 18, right: 16, bottom: 38, left: 50 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const xMin = Math.min(...xs);
        const xMax = Math.max(...xs);
        const yMin = Math.min(...ys);
        const yMax = Math.max(...ys);
        const xPad = Math.max((xMax - xMin) * 0.05, 1e-6);
        const yPad = Math.max((yMax - yMin) * 0.08, 1e-6);
        const xScale = (value) => margin.left + ((Number(value) - (xMin - xPad)) / ((xMax + xPad) - (xMin - xPad))) * plotWidth;
        const yScale = (value) => margin.top + plotHeight - ((Number(value) - (yMin - yPad)) / ((yMax + yPad) - (yMin - yPad))) * plotHeight;

        const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = margin.top + ratio * plotHeight;
          const value = ((1 - ratio) * ((yMax + yPad) - (yMin - yPad)) + (yMin - yPad)).toFixed(1);
          return `
            <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(15,23,42,0.10)" stroke-dasharray="4 6" />
            <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#475569">${value}</text>
          `;
        }).join('');

        const unionNames = unique([
          ...goldSeries.map((series) => series.name),
          ...predictedSeries.map((series) => series.name),
        ]);

        const lines = unionNames.flatMap((name, index) => {
          const color = colorForSeries(name, index);
          const gold = goldSeries.find((series) => series.name === name);
          const predicted = predictedSeries.find((series) => series.name === name);
          const pieces = [];

          if (gold && gold.points.length) {
            const path = gold.points.map((point, pointIndex) => `${pointIndex === 0 ? 'M' : 'L'} ${xScale(point[0]).toFixed(2)} ${yScale(point[1]).toFixed(2)}`).join(' ');
            pieces.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" />`);
            pieces.push(gold.points.map((point) => `<circle cx="${xScale(point[0])}" cy="${yScale(point[1])}" r="3.8" fill="${color}" />`).join(''));
          }

          if (predicted && predicted.points.length) {
            const path = predicted.points.map((point, pointIndex) => `${pointIndex === 0 ? 'M' : 'L'} ${xScale(point[0]).toFixed(2)} ${yScale(point[1]).toFixed(2)}`).join(' ');
            pieces.push(`<path d="${path}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="7 6" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" />`);
            pieces.push(predicted.points.map((point) => `<path d="M ${xScale(point[0]) - 3.4} ${yScale(point[1]) - 3.4} L ${xScale(point[0]) + 3.4} ${yScale(point[1]) + 3.4} M ${xScale(point[0]) - 3.4} ${yScale(point[1]) + 3.4} L ${xScale(point[0]) + 3.4} ${yScale(point[1]) - 3.4}" stroke="${color}" stroke-width="1.7" stroke-linecap="round" />`).join(''));
          }

          return pieces;
        }).join('');

        const legend = unionNames.map((name, index) => {
          const color = colorForSeries(name, index);
          return `
            <div class="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-700">
              <span class="inline-block h-2.5 w-2.5 rounded-full" style="background:${color}"></span>
              <span>${escapeHtml(name)}</span>
            </div>
          `;
        }).join('');

        elements.overlayChart.innerHTML = `
          <div class="flex h-full flex-col">
            <div class="flex flex-wrap gap-2 border-b border-slate-200 px-4 py-3">${legend}</div>
            <div class="min-h-0 flex-1">
              ${svgShell(width, height, `
                <rect x="0" y="0" width="${width}" height="${height}" fill="#f8fafc" />
                ${grid}
                <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />
                <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />
                ${lines}
                <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#475569">${escapeHtml(entry.sample.gold.chart?.x_axis_label || 'x')}</text>
                <text x="14" y="${height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 14 ${height / 2})">${escapeHtml(entry.sample.gold.chart?.y_axis_label || 'y')}</text>
              `)}
            </div>
            <div class="border-t border-slate-200 px-4 py-3 text-xs text-slate-500">
              Solid circles = gold points. Dashed lines and x-marks = rollout prediction.
            </div>
          </div>
        `;
      }

      function renderBreakdown(entry) {
        if (!entry) {
          elements.detailBreakdown.innerHTML = '<div class="px-4 py-6 text-sm text-slate-500">Choose a sample from the table.</div>';
          return;
        }

        const rows = [
          ['format_reward', entry.sample.logged.format_reward, entry.formatReward],
          ['series_name_f1', entry.sample.logged.name_f1, entry.nameF1],
          ['series_point_count_ratio', entry.sample.logged.count_ratio, entry.countRatio],
          ['series_point_value', entry.sample.logged.point_value, entry.pointValue],
          ['total_reward', entry.sample.logged.reward, entry.totalReward],
        ];

        const body = rows.map(([label, logged, simulated]) => `
          <tr class="border-t border-slate-200">
            <td class="px-4 py-3 font-mono text-xs text-slate-600">${label}</td>
            <td class="px-4 py-3 font-mono text-xs text-slate-700">${fmt(logged, 3)}</td>
            <td class="px-4 py-3 font-mono text-xs text-slate-700">${fmt(simulated, 3)}</td>
            <td class="px-4 py-3">
              <span class="rounded-full px-2.5 py-1 text-xs font-medium ${badgeClass(simulated - logged)}">${simulated - logged > 0 ? '+' : ''}${fmt(simulated - logged, 3)}</span>
            </td>
          </tr>
        `).join('');

        elements.detailBreakdown.innerHTML = `
          <table class="min-w-full bg-white text-left text-sm">
            <thead class="bg-slate-100/90">
              <tr class="text-xs uppercase tracking-[0.16em] text-slate-600">
                <th class="px-4 py-3 font-medium">Metric</th>
                <th class="px-4 py-3 font-medium">Logged</th>
                <th class="px-4 py-3 font-medium">Simulated</th>
                <th class="px-4 py-3 font-medium">Delta</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        `;
      }

      function renderDiagnostics(entry) {
        if (!entry) {
          elements.detailDiagnostics.innerHTML = '<div class="col-span-2 text-sm text-slate-500">Choose a sample from the table.</div>';
          return;
        }

        const details = [
          ['Order-aligned y score', fmt(entry.diagnostics.orderAlignedYScore, 3)],
          ['Exact x match fraction', fmt(entry.diagnostics.exactXMatchFraction, 3)],
          ['Mean x-step ratio', fmt(entry.diagnostics.meanXStepRatio, 3)],
          ['Predicted parse ok', entry.sample.predicted.parse_ok ? 'yes' : 'no'],
          ['Gold series count', String(entry.sample.gold.chart?.series?.length || 0)],
          ['Predicted series count', String(entry.sample.predicted.chart?.series?.length || 0)],
        ];

        elements.detailDiagnostics.innerHTML = details.map(([label, value]) => `
          <div class="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3">
            <dt class="text-xs uppercase tracking-[0.14em] text-slate-500">${label}</dt>
            <dd class="mt-1 font-mono text-sm text-slate-900">${value}</dd>
          </div>
        `).join('');
      }

      function renderSeriesTable(entry) {
        if (!entry) {
          elements.detailSeriesTable.innerHTML = '<div class="px-4 py-6 text-sm text-slate-500">Choose a sample from the table.</div>';
          return;
        }

        const predictedSeries = seriesMap(entry.sample.predicted.chart);
        const goldSeries = seriesMap(entry.sample.gold.chart);
        const detailsByName = new Map(entry.pointValueDetail.bySeries.map((detail) => [detail.name, detail]));

        const rows = [...goldSeries.entries()].map(([name, goldPoints]) => {
          const predictedPoints = predictedSeries.get(name) || [];
          const detail = detailsByName.get(name) || { score: 0, matchedGold: 0 };
          return `
            <tr class="border-t border-slate-200">
              <td class="px-4 py-3">
                <div class="font-medium text-slate-900">${escapeHtml(name)}</div>
              </td>
              <td class="px-4 py-3 font-mono text-xs text-slate-700">${goldPoints.length}</td>
              <td class="px-4 py-3 font-mono text-xs text-slate-700">${predictedPoints.length}</td>
              <td class="px-4 py-3 font-mono text-xs text-slate-700">${fmt(detail.score, 3)}</td>
              <td class="px-4 py-3 font-mono text-xs text-slate-700">${detail.matchedGold} / ${detail.goldCount}</td>
              <td class="px-4 py-3 font-mono text-xs text-slate-700">${fmt(exactXMatchFraction(predictedPoints, goldPoints), 3)}</td>
              <td class="px-4 py-3 font-mono text-xs text-slate-700">${fmt(orderAlignedYScore(predictedPoints, goldPoints), 3)}</td>
            </tr>
          `;
        }).join('');

        elements.detailSeriesTable.innerHTML = `
          <table class="min-w-full bg-white text-left text-sm">
            <thead class="bg-slate-100/90">
              <tr class="text-xs uppercase tracking-[0.16em] text-slate-600">
                <th class="px-4 py-3 font-medium">Series</th>
                <th class="px-4 py-3 font-medium">Gold pts</th>
                <th class="px-4 py-3 font-medium">Pred pts</th>
                <th class="px-4 py-3 font-medium">Value</th>
                <th class="px-4 py-3 font-medium">Matched gold</th>
                <th class="px-4 py-3 font-medium">Exact x</th>
                <th class="px-4 py-3 font-medium">Y shape</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      }

      function renderSelectedDetail(filtered, computedSamples) {
        let entry = filtered.find((item) => item.sample.id === state.selectedId) || computedSamples.find((item) => item.sample.id === state.selectedId) || filtered[0] || computedSamples[0] || null;
        if (entry) {
          state.selectedId = entry.sample.id;
        }

        if (!entry) {
          elements.detailSubtitle.textContent = 'Choose a sample from the table.';
          elements.detailBadges.innerHTML = '';
          elements.detailLoggedValue.textContent = '-';
          elements.detailSimulatedValue.textContent = '-';
          elements.detailValueDelta.textContent = '-';
          renderOverlayChart(null);
          renderBreakdown(null);
          renderDiagnostics(null);
          renderSeriesTable(null);
          return;
        }

        elements.detailSubtitle.textContent = `${entry.sample.file_name} • ${entry.sample.gold.chart?.title || 'untitled'} • step ${entry.sample.step}`;
        elements.detailBadges.innerHTML = `
          <span class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700">image ${entry.sample.image_id ?? 'n/a'}</span>
          <span class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700">name F1 ${fmt(entry.nameF1, 2)}</span>
          <span class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700">count ${fmt(entry.countRatio, 2)}</span>
          <span class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700">schema ${entry.sample.schema_version}</span>
        `;
        elements.detailLoggedValue.textContent = fmt(entry.sample.logged.point_value, 3);
        elements.detailSimulatedValue.textContent = fmt(entry.pointValue, 3);
        elements.detailValueDelta.textContent = `${entry.valueDelta > 0 ? '+' : ''}${fmt(entry.valueDelta, 3)}`;
        renderOverlayChart(entry);
        renderBreakdown(entry);
        renderDiagnostics(entry);
        renderSeriesTable(entry);
      }

      function refresh() {
        renderControls();
        const computed = PLAYGROUND.samples.map((sample) => computeSample(sample, state));
        const filtered = filteredSamples(computed);
        renderSummary(filtered);
        renderTrendChart(filtered);
        renderHistogram(filtered);
        renderTable(filtered);
        renderSelectedDetail(filtered, computed);
      }

      function applyPreset(name) {
        const preset = DEFAULT_PRESETS[name];
        if (!preset) return;
        state.valueMode = preset.valueMode;
        state.oksK = preset.oksK;
        state.oksThreshold = preset.oksThreshold;
        refresh();
      }

      function bindEvents() {
        elements.resetButton.addEventListener('click', () => {
          state.valueMode = 'oks';
          state.oksK = PLAYGROUND.meta.current_value_reward.oks_k;
          state.oksThreshold = PLAYGROUND.meta.current_value_reward.oks_threshold;
          state.step = 'all';
          state.minNameF1 = 0;
          state.minCountRatio = 0;
          state.search = '';
          state.nearMissOnly = false;
          refresh();
        });

        for (const button of elements.presetButtons) {
          button.addEventListener('click', () => applyPreset(button.dataset.preset));
        }

        elements.valueMode.addEventListener('change', (event) => {
          state.valueMode = event.target.value;
          refresh();
        });
        elements.oksK.addEventListener('input', (event) => {
          state.oksK = Number(event.target.value);
          refresh();
        });
        elements.oksThreshold.addEventListener('input', (event) => {
          state.oksThreshold = Number(event.target.value);
          refresh();
        });
        elements.stepFilter.addEventListener('change', (event) => {
          state.step = event.target.value;
          refresh();
        });
        elements.nameFilter.addEventListener('input', (event) => {
          state.minNameF1 = Number(event.target.value);
          refresh();
        });
        elements.countFilter.addEventListener('input', (event) => {
          state.minCountRatio = Number(event.target.value);
          refresh();
        });
        elements.searchFilter.addEventListener('input', (event) => {
          state.search = event.target.value;
          refresh();
        });
        elements.nearMissOnly.addEventListener('change', (event) => {
          state.nearMissOnly = event.target.checked;
          refresh();
        });
      }

      renderHero();
      populateStepFilter();
      bindEvents();
      refresh();
    </script>
  </body>
</html>
"""

    return (
        template
        .replace("__DATA_JSON__", data_json)
        .replace("__RUN_ID__", dataset["meta"]["run_id"])
        .replace("__DEFAULT_OKS_K__", f"{dataset['meta']['current_value_reward']['oks_k']:.3f}")
        .replace("__DEFAULT_OKS_THRESHOLD__", f"{dataset['meta']['current_value_reward']['oks_threshold']:.2f}")
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.run_id, num_per_step=args.num_per_step)
    html = build_html(dataset)
    args.output.write_text(html, encoding="utf-8")

    validation = dataset["meta"]["validation"]
    print(f"Wrote {args.output}")
    print(f"Samples: {dataset['meta']['sample_count']}")
    print(f"Steps: {dataset['meta']['steps_with_samples']}")
    print(
        "Validation:",
        json.dumps(validation, indent=2),
    )


if __name__ == "__main__":
    main()
