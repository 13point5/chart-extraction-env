from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Image as HFImage, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SRC = REPO_ROOT / "environments" / "chart_extraction"
if str(ENV_SRC) not in sys.path:
    sys.path.insert(0, str(ENV_SRC))

from chart_extraction_env.dataset.models import CanonicalAnswer, OutputMode  # noqa: E402
from chart_extraction_env.environment import point_component_scores  # noqa: E402
from chart_extraction_env.parsing import (  # noqa: E402
    extract_assistant_text,
    normalized_label,
    parse_response,
)

NUMERIC_X_TOLERANCE = 1e-3
MATERIAL_Y_ERROR_THRESHOLD = 0.05
SEVERE_Y_ERROR_THRESHOLD = 0.10


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_id: str
    output_mode: OutputMode


@dataclass(frozen=True)
class DatasetRef:
    split: str
    index: int
    row_id: str
    info: dict[str, Any]


@dataclass(frozen=True)
class PointError:
    target_index: int
    target_x: float | str
    predicted_x: float | str
    target_y: float
    predicted_y: float
    abs_x_error: float
    abs_y_error: float
    normalized_y_error: float


@dataclass(frozen=True)
class MissingPoint:
    target_index: int
    target_x: float | str
    target_y: float


@dataclass(frozen=True)
class ExtraPoint:
    predicted_index: int
    predicted_x: float | str
    predicted_y: float


@dataclass(frozen=True)
class SampleAnalysis:
    run_label: str
    run_id: str
    step: int
    problem_id: int
    sample_id: int
    reward: float
    row_id: str
    split: str
    data_profile: str
    style_profile: str
    image_size_profile: str
    num_points_target: int
    num_points_predicted: int
    matched_points: int
    missing_points: int
    extra_points: int
    duplicate_predicted_x: int
    material_y_errors: int
    severe_y_errors: int
    exact_x_matches: int
    y_mae: float
    normalized_y_mae: float
    max_normalized_y_error: float
    point_count_score: float
    point_value_score: float
    point_x_score: float
    point_y_score: float
    reward_metrics: dict[str, Any]
    parse_valid: bool
    parse_error: str | None
    input_image_relpath: str
    comparison_relpath: str | None
    prediction_relpath: str
    target_relpath: str
    point_errors: list[PointError]
    missing_target_points: list[MissingPoint]
    extra_predicted_points: list[ExtraPoint]
    raw_completion_excerpt: str
    raw_target_excerpt: str

    @property
    def materially_wrong(self) -> bool:
        return (
            not self.parse_valid
            or self.missing_points >= 10
            or self.extra_points >= 10
            or self.material_y_errors >= 10
            or self.normalized_y_mae >= MATERIAL_Y_ERROR_THRESHOLD
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-local-path",
        default=str(
            REPO_ROOT / "environments" / "chart_extraction" / "outputs" / "chart_extraction" / "dense_noisy_local_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "reports" / "chart_extraction_rollout_analysis"),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label:run_id:json|markdown",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional rollout step. If omitted, use the latest shared step with saved rollout samples.",
    )
    parser.add_argument(
        "--detailed-per-run",
        type=int,
        default=4,
        help="Number of detailed sample sections to include per run.",
    )
    return parser.parse_args()


def parse_run_spec(raw: str) -> RunSpec:
    label, run_id, mode = raw.split(":", 2)
    return RunSpec(label=label, run_id=run_id, output_mode=OutputMode(mode))


def extract_json_payload(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON payload in CLI output:\n{text}")
    return text[start : end + 1]


def escape_control_chars_in_json_strings(payload: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in payload:
        if in_string:
            if escaped:
                if char in {"\n", "\r"}:
                    # Prime CLI sometimes hard-wraps long JSON strings after a backslash.
                    # Preserve the escape so the following character stays escaped.
                    continue
                output.append(char)
                escaped = False
                continue
            if char == "\\":
                output.append(char)
                escaped = True
                continue
            if char == '"':
                output.append(char)
                in_string = False
                continue
            if char == "\n":
                output.append("\\n")
                continue
            if char == "\r":
                output.append("\\r")
                continue
            if char == "\t":
                output.append("\\t")
                continue
            if ord(char) < 0x20:
                output.append(f"\\u{ord(char):04x}")
                continue
            output.append(char)
            continue

        output.append(char)
        if char == '"':
            in_string = True

    return "".join(output)


def strip_control_chars_outside_json_strings(payload: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in payload:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char in {"\n", "\r", "\t"}:
            continue
        if ord(char) < 0x20:
            continue
        output.append(char)
        if char == '"':
            in_string = True

    return "".join(output)


def json_loads_lenient(text: str) -> Any:
    payload = escape_control_chars_in_json_strings(text)
    payload = strip_control_chars_outside_json_strings(payload)
    return json.loads(payload)


def run_prime_json(args: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    process = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    payload = extract_json_payload(process.stdout or process.stderr)
    return json_loads_lenient(payload)


def latest_shared_step(run_specs: list[RunSpec]) -> int:
    return shared_steps(run_specs)[-1]


def shared_steps(run_specs: list[RunSpec]) -> list[int]:
    step_sets: list[set[int]] = []
    for run in run_specs:
        progress = run_prime_json(["prime", "rl", "progress", run.run_id])
        step_sets.append(set(progress["steps_with_samples"]))
    shared = set.intersection(*step_sets)
    if not shared:
        raise ValueError("No shared rollout step found across runs.")
    return sorted(shared)


def fetch_all_rollouts(run_id: str, step: int) -> list[dict[str, Any]]:
    first_page = run_prime_json(
        ["prime", "rl", "rollouts", run_id, "--step", str(step), "--num", "100", "--page", "1"]
    )
    samples = list(first_page["samples"])
    total_pages = first_page.get("total_pages", 1)
    for page in range(2, total_pages + 1):
        page_payload = run_prime_json(
            ["prime", "rl", "rollouts", run_id, "--step", str(step), "--num", "100", "--page", str(page)]
        )
        samples.extend(page_payload["samples"])
    return samples


def build_answer_index(dataset_local_path: Path) -> tuple[dict[str, DatasetRef], Any]:
    dataset = load_from_disk(str(dataset_local_path))
    answer_index: dict[str, DatasetRef] = {}
    for split in ("train", "validation", "test"):
        split_ds = dataset[split].cast_column("image", HFImage(decode=False))
        for idx, row in enumerate(split_ds):
            answer_index[canonical_answer_key(row["answer"])] = DatasetRef(
                split=split,
                index=idx,
                row_id=row["id"],
                info=json.loads(row["info"]),
            )
    return answer_index, dataset


def load_embedded_json(text: str) -> Any:
    return json_loads_lenient(text)


def compact_embedded_json(text: str) -> str:
    return "".join(char for char in text if char not in {"\n", "\r", "\t"})


def parse_canonical_answer_lenient(payload: str) -> CanonicalAnswer:
    return CanonicalAnswer.model_validate(json_loads_lenient(compact_embedded_json(payload)))


def canonical_answer_key(payload: str) -> str:
    answer = parse_canonical_answer_lenient(payload)
    return answer.model_dump_json()


def completion_text(sample: dict[str, Any]) -> str:
    completion = load_embedded_json(sample["completion"])
    return extract_assistant_text(completion)


def parse_reward_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    return load_embedded_json(sample["metrics"])


def same_x(left: float | str, right: float | str) -> tuple[bool, float]:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return normalized_label(str(left)) == normalized_label(str(right)), 0.0
    x_error = abs(left_value - right_value)
    return x_error <= NUMERIC_X_TOLERANCE, x_error


def analyze_point_errors(parsed: CanonicalAnswer, target: CanonicalAnswer) -> tuple[
    list[PointError],
    list[MissingPoint],
    list[ExtraPoint],
    int,
    int,
    int,
    float,
    float,
    float,
]:
    predicted_points = list(parsed.series[0].points)
    target_points = list(target.series[0].points)
    y_values = [point.y for point in target_points]
    y_scale = max(max(y_values) - min(y_values), 1.0)

    matched_target_indices: set[int] = set()
    matched_errors: list[PointError] = []
    extra_points: list[ExtraPoint] = []
    duplicate_predicted_x = 0

    for predicted_index, predicted_point in enumerate(predicted_points):
        matched_index: int | None = None
        matched_x_error = 0.0
        for target_index, target_point in enumerate(target_points):
            if target_index in matched_target_indices:
                continue
            is_match, x_error = same_x(predicted_point.x, target_point.x)
            if not is_match:
                continue
            matched_index = target_index
            matched_x_error = x_error
            break

        if matched_index is None:
            extra_points.append(
                ExtraPoint(
                    predicted_index=predicted_index,
                    predicted_x=predicted_point.x,
                    predicted_y=predicted_point.y,
                )
            )
            continue

        if matched_index in matched_target_indices:
            duplicate_predicted_x += 1
            extra_points.append(
                ExtraPoint(
                    predicted_index=predicted_index,
                    predicted_x=predicted_point.x,
                    predicted_y=predicted_point.y,
                )
            )
            continue

        matched_target_indices.add(matched_index)
        target_point = target_points[matched_index]
        abs_y_error = abs(predicted_point.y - target_point.y)
        matched_errors.append(
            PointError(
                target_index=matched_index,
                target_x=target_point.x,
                predicted_x=predicted_point.x,
                target_y=target_point.y,
                predicted_y=predicted_point.y,
                abs_x_error=matched_x_error,
                abs_y_error=abs_y_error,
                normalized_y_error=abs_y_error / y_scale,
            )
        )

    missing_points = [
        MissingPoint(target_index=index, target_x=point.x, target_y=point.y)
        for index, point in enumerate(target_points)
        if index not in matched_target_indices
    ]
    matched_errors.sort(key=lambda point: point.normalized_y_error, reverse=True)
    abs_y_errors = [point.abs_y_error for point in matched_errors]
    normalized_y_errors = [point.normalized_y_error for point in matched_errors]
    y_mae = statistics.mean(abs_y_errors) if abs_y_errors else 0.0
    normalized_y_mae = statistics.mean(normalized_y_errors) if normalized_y_errors else 1.0
    max_normalized_y_error = max(normalized_y_errors, default=1.0)
    exact_x_matches = sum(1 for point in matched_errors if point.abs_x_error <= NUMERIC_X_TOLERANCE)
    material_y_errors = sum(1 for point in matched_errors if point.normalized_y_error >= MATERIAL_Y_ERROR_THRESHOLD)
    severe_y_errors = sum(1 for point in matched_errors if point.normalized_y_error >= SEVERE_Y_ERROR_THRESHOLD)

    return (
        matched_errors,
        missing_points,
        extra_points,
        duplicate_predicted_x,
        exact_x_matches,
        material_y_errors,
        severe_y_errors,
        y_mae,
        normalized_y_mae,
        max_normalized_y_error,
    )


def save_text(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path.name


def excerpt(text: str, limit: int = 1800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...<truncated>..."


def plot_series(ax: Any, answer: CanonicalAnswer, title: str) -> None:
    points = answer.series[0].points
    x_values = [float(point.x) for point in points]
    y_values = [point.y for point in points]
    line_width = 1.4 if len(points) >= 120 else 1.8
    alpha = 0.95 if len(points) < 250 else 0.9
    ax.plot(x_values, y_values, color="#2563eb", linewidth=line_width, alpha=alpha)
    if len(points) <= 80:
        ax.scatter(x_values, y_values, color="#1d4ed8", s=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)


def render_comparison_image(
    *,
    target: CanonicalAnswer,
    parsed: CanonicalAnswer | None,
    destination: Path,
    predicted_points: int,
) -> str | None:
    if parsed is None:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5), constrained_layout=True)
    plot_series(axes[0], target, f"Ground Truth ({len(target.series[0].points)} pts)")
    plot_series(axes[1], parsed, f"Model Output ({predicted_points} pts)")

    y_values = [point.y for point in target.series[0].points] + [point.y for point in parsed.series[0].points]
    x_values = [float(point.x) for point in target.series[0].points] + [float(point.x) for point in parsed.series[0].points]
    if x_values:
        x_min = min(x_values)
        x_max = max(x_values)
        x_pad = max((x_max - x_min) * 0.03, 1.0)
        axes[0].set_xlim(x_min - x_pad, x_max + x_pad)
        axes[1].set_xlim(x_min - x_pad, x_max + x_pad)
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        y_pad = max((y_max - y_min) * 0.08, 0.5)
        axes[0].set_ylim(y_min - y_pad, y_max + y_pad)
        axes[1].set_ylim(y_min - y_pad, y_max + y_pad)

    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination.name


def analyze_sample(
    *,
    sample: dict[str, Any],
    run_spec: RunSpec,
    step: int,
    dataset: Any,
    answer_index: dict[str, DatasetRef],
    assets_dir: Path,
) -> SampleAnalysis:
    target = parse_canonical_answer_lenient(sample["answer"])
    reward_metrics = parse_reward_metrics(sample)
    ref = answer_index[canonical_answer_key(sample["answer"])]
    completion = completion_text(sample)

    prediction_name = save_text(
        assets_dir / f"{run_spec.label}_step{step}_problem{sample['problem_id']}_sample{sample['sample_id']}_prediction.txt",
        completion,
    )
    target_name = save_text(
        assets_dir / f"{run_spec.label}_step{step}_problem{sample['problem_id']}_sample{sample['sample_id']}_target.json",
        sample["answer"],
    )

    image_path = assets_dir / f"{run_spec.label}_step{step}_problem{sample['problem_id']}_sample{sample['sample_id']}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    dataset[ref.split][ref.index]["image"].save(image_path)

    parsed: CanonicalAnswer | None = None
    parse_error: str | None = None
    try:
        parsed = parse_response(completion, run_spec.output_mode)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    point_errors: list[PointError] = []
    missing_target_points: list[MissingPoint] = [
        MissingPoint(target_index=index, target_x=point.x, target_y=point.y)
        for index, point in enumerate(target.series[0].points)
    ]
    extra_predicted_points: list[ExtraPoint] = []
    duplicate_predicted_x = 0
    exact_x_matches = 0
    material_y_errors = 0
    severe_y_errors = 0
    y_mae = 0.0
    normalized_y_mae = 1.0 if parsed is None else 0.0
    max_normalized_y_error = 1.0 if parsed is None else 0.0
    comparison_name: str | None = None

    if parsed is not None:
        (
            point_errors,
            missing_target_points,
            extra_predicted_points,
            duplicate_predicted_x,
            exact_x_matches,
            material_y_errors,
            severe_y_errors,
            y_mae,
            normalized_y_mae,
            max_normalized_y_error,
        ) = analyze_point_errors(parsed, target)
        comparison_name = render_comparison_image(
            target=target,
            parsed=parsed,
            destination=assets_dir
            / f"{run_spec.label}_step{step}_problem{sample['problem_id']}_sample{sample['sample_id']}_comparison.png",
            predicted_points=len(parsed.series[0].points),
        )

    point_metrics = point_component_scores(parsed, target) if parsed is not None else {
        "point_value_score": 0.0,
        "point_x_score": 0.0,
        "point_y_score": 0.0,
        "normalized_y_mae": 1.0,
    }

    return SampleAnalysis(
        run_label=run_spec.label,
        run_id=run_spec.run_id,
        step=step,
        problem_id=sample["problem_id"],
        sample_id=sample["sample_id"],
        reward=float(sample["reward"]),
        row_id=ref.row_id,
        split=ref.split,
        data_profile=str(ref.info.get("data_profile", "")),
        style_profile=str(ref.info.get("style_profile", "")),
        image_size_profile=str(ref.info.get("image_size_profile", "")),
        num_points_target=len(target.series[0].points),
        num_points_predicted=(len(parsed.series[0].points) if parsed is not None else 0),
        matched_points=len(point_errors),
        missing_points=len(missing_target_points),
        extra_points=len(extra_predicted_points),
        duplicate_predicted_x=duplicate_predicted_x,
        material_y_errors=material_y_errors,
        severe_y_errors=severe_y_errors,
        exact_x_matches=exact_x_matches,
        y_mae=y_mae,
        normalized_y_mae=normalized_y_mae,
        max_normalized_y_error=max_normalized_y_error,
        point_count_score=float(reward_metrics["point_count_score"]),
        point_value_score=float(reward_metrics["point_value_score"]),
        point_x_score=float(reward_metrics["point_x_score_metric"]),
        point_y_score=float(reward_metrics["point_y_score_metric"]),
        reward_metrics=reward_metrics,
        parse_valid=parsed is not None,
        parse_error=parse_error,
        input_image_relpath=str(Path("assets") / image_path.name),
        comparison_relpath=(str(Path("assets") / comparison_name) if comparison_name is not None else None),
        prediction_relpath=str(Path("assets") / prediction_name),
        target_relpath=str(Path("assets") / target_name),
        point_errors=point_errors,
        missing_target_points=missing_target_points,
        extra_predicted_points=extra_predicted_points,
        raw_completion_excerpt=excerpt(completion),
        raw_target_excerpt=excerpt(sample["answer"]),
    )


def aggregate_samples(samples: list[SampleAnalysis]) -> dict[str, float]:
    materially_wrong = [sample for sample in samples if sample.materially_wrong]
    high_reward_wrong = [sample for sample in samples if sample.reward >= 0.9 and sample.materially_wrong]
    return {
        "mean_reward": statistics.mean(sample.reward for sample in samples),
        "mean_point_count_score": statistics.mean(sample.point_count_score for sample in samples),
        "mean_point_value_score": statistics.mean(sample.point_value_score for sample in samples),
        "mean_missing_points": statistics.mean(sample.missing_points for sample in samples),
        "mean_extra_points": statistics.mean(sample.extra_points for sample in samples),
        "mean_material_y_errors": statistics.mean(sample.material_y_errors for sample in samples),
        "mean_normalized_y_mae": statistics.mean(sample.normalized_y_mae for sample in samples),
        "max_missing_points": max(sample.missing_points for sample in samples),
        "parse_fail_rate": sum(1 for sample in samples if not sample.parse_valid) / len(samples),
        "materially_wrong_rate": len(materially_wrong) / len(samples),
        "high_reward_wrong_rate": len(high_reward_wrong) / len(samples),
    }


def render_full_table(samples: list[SampleAnalysis]) -> list[str]:
    rows = [
        "| Row ID | Profile | Reward | Target pts | Pred pts | Missing | Extra | Material y errs | Severe y errs | point_count_score | point_value_score | norm y MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample in sorted(samples, key=lambda item: item.reward):
        rows.append(
            f"| `{sample.row_id}` | `{sample.data_profile}` | {sample.reward:.4f} | "
            f"{sample.num_points_target} | {sample.num_points_predicted} | {sample.missing_points} | "
            f"{sample.extra_points} | {sample.material_y_errors} | {sample.severe_y_errors} | "
            f"{sample.point_count_score:.4f} | {sample.point_value_score:.4f} | {sample.normalized_y_mae:.4f} |"
        )
    return rows


def render_point_error_table(errors: list[PointError], limit: int = 15) -> str:
    rows = [
        "| target idx | target x | pred x | target y | pred y | abs x err | abs y err | norm y err |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in errors[:limit]:
        rows.append(
            f"| {point.target_index} | {point.target_x} | {point.predicted_x} | {point.target_y:.4f} | "
            f"{point.predicted_y:.4f} | {point.abs_x_error:.4f} | {point.abs_y_error:.4f} | "
            f"{point.normalized_y_error:.4f} |"
        )
    return "\n".join(rows)


def render_missing_points(points: list[MissingPoint], limit: int = 25) -> str:
    preview = ", ".join(f"{point.target_x}" for point in points[:limit])
    suffix = " ..." if len(points) > limit else ""
    return f"{len(points)} missing target x values: `{preview}{suffix}`"


def render_extra_points(points: list[ExtraPoint], limit: int = 25) -> str:
    preview = ", ".join(f"{point.predicted_x}" for point in points[:limit])
    suffix = " ..." if len(points) > limit else ""
    return f"{len(points)} extra predicted x values: `{preview}{suffix}`"


def select_detailed_samples(samples: list[SampleAnalysis], limit: int) -> list[SampleAnalysis]:
    candidates: list[SampleAnalysis] = []
    used_keys: set[tuple[int, int]] = set()

    def add(sample: SampleAnalysis | None) -> None:
        if sample is None:
            return
        key = (sample.problem_id, sample.sample_id)
        if key in used_keys:
            return
        used_keys.add(key)
        candidates.append(sample)

    ordered = sorted(samples, key=lambda sample: sample.reward)
    add(ordered[0] if ordered else None)
    add(max(samples, key=lambda sample: sample.missing_points, default=None))
    add(max(samples, key=lambda sample: sample.normalized_y_mae, default=None))

    high_reward_wrong = [
        sample
        for sample in samples
        if sample.reward >= 0.9 and (sample.missing_points > 0 or sample.material_y_errors > 0 or sample.extra_points > 0)
    ]
    add(max(high_reward_wrong, key=lambda sample: sample.material_y_errors + sample.missing_points, default=None))

    for sample in ordered:
        if len(candidates) >= limit:
            break
        add(sample)

    return candidates[:limit]


def render_detailed_section(sample: SampleAnalysis) -> list[str]:
    lines: list[str] = []
    lines.append(f"#### `{sample.row_id}`")
    lines.append("")
    lines.append("Input chart:")
    lines.append("")
    lines.append(f"![{sample.row_id} input]({sample.input_image_relpath})")
    lines.append("")
    if sample.comparison_relpath is not None:
        lines.append("Ground-truth vs model replot:")
        lines.append("")
        lines.append(f"![{sample.row_id} comparison]({sample.comparison_relpath})")
        lines.append("")

    lines.extend(
        [
            "| Metric | Value |",
            "| --- | --- |",
            f"| reward | `{sample.reward:.4f}` |",
            f"| data profile | `{sample.data_profile}` |",
            f"| style profile | `{sample.style_profile}` |",
            f"| image size profile | `{sample.image_size_profile}` |",
            f"| target points | `{sample.num_points_target}` |",
            f"| predicted points | `{sample.num_points_predicted}` |",
            f"| matched x values | `{sample.matched_points}` |",
            f"| missing points | `{sample.missing_points}` |",
            f"| extra points | `{sample.extra_points}` |",
            f"| exact x matches | `{sample.exact_x_matches}` |",
            f"| material y errors | `{sample.material_y_errors}` |",
            f"| severe y errors | `{sample.severe_y_errors}` |",
            f"| point_count_score | `{sample.point_count_score:.4f}` |",
            f"| point_value_score | `{sample.point_value_score:.4f}` |",
            f"| point_x_score_metric | `{sample.point_x_score:.4f}` |",
            f"| point_y_score_metric | `{sample.point_y_score:.4f}` |",
            f"| normalized_y_mae | `{sample.normalized_y_mae:.4f}` |",
            f"| max_normalized_y_error | `{sample.max_normalized_y_error:.4f}` |",
        ]
    )
    lines.append("")

    if sample.parse_error is not None:
        lines.append(f"Parse error: `{sample.parse_error}`")
        lines.append("")
    if sample.missing_target_points:
        lines.append(render_missing_points(sample.missing_target_points))
        lines.append("")
    if sample.extra_predicted_points:
        lines.append(render_extra_points(sample.extra_predicted_points))
        lines.append("")
    if sample.point_errors:
        lines.append("Largest y-magnitude errors at matched x values:")
        lines.append("")
        lines.append(render_point_error_table(sample.point_errors))
        lines.append("")

    lines.append("Prediction excerpt:")
    lines.append("")
    lines.append("```text")
    lines.append(sample.raw_completion_excerpt)
    lines.append("```")
    lines.append("")
    lines.append("Ground-truth excerpt:")
    lines.append("")
    lines.append("```text")
    lines.append(sample.raw_target_excerpt)
    lines.append("```")
    lines.append("")
    lines.append(f"Full files: [prediction]({sample.prediction_relpath}) | [target]({sample.target_relpath})")
    lines.append("")
    return lines


def render_report(
    *,
    run_specs: list[RunSpec],
    steps: list[int],
    analyses_by_run: dict[str, list[SampleAnalysis]],
    output_path: Path,
    generated_at: str,
    detailed_per_run: int,
) -> None:
    lines: list[str] = []
    lines.append("# Chart Extraction RL Rollout Analysis")
    lines.append("")
    lines.append(f"Generated at `{generated_at}`.")
    lines.append("")
    lines.append(
        f"This report inspects Prime RL rollout samples for the completed JSON and Markdown runs across `{len(steps)}` shared saved steps."
    )
    lines.append("")
    lines.append(f"Analyzed steps: `{', '.join(str(step) for step in steps)}`")
    lines.append("")

    lines.append("## Run Summary")
    lines.append("")
    lines.append(
        "| Run | Run ID | Samples | Mean reward | Mean point_count_score | Mean point_value_score | Mean missing points | Mean material y errors | Mean normalized y MAE | Parse fail rate | Materially wrong rate | High-reward wrong rate |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for run in run_specs:
        aggregate = aggregate_samples(analyses_by_run[run.label])
        lines.append(
            f"| `{run.label}` | `{run.run_id}` | {len(analyses_by_run[run.label])} | "
            f"{aggregate['mean_reward']:.4f} | {aggregate['mean_point_count_score']:.4f} | "
            f"{aggregate['mean_point_value_score']:.4f} | {aggregate['mean_missing_points']:.2f} | "
            f"{aggregate['mean_material_y_errors']:.2f} | {aggregate['mean_normalized_y_mae']:.4f} | "
            f"{aggregate['parse_fail_rate']:.2%} | {aggregate['materially_wrong_rate']:.2%} | "
            f"{aggregate['high_reward_wrong_rate']:.2%} |"
        )
    lines.append("")

    lines.append("## Reward Gap")
    lines.append("")
    lines.append(
        "- From the environment code: `point_count_score = min(pred_count, target_count) / max(pred_count, target_count)`."
    )
    lines.append(
        "- From the environment code: `point_value_score = mean(0.4 * x_score + 0.6 * y_score) * coverage`, where `coverage = overlap / max(pred_count, target_count)`."
    )
    lines.append(
        "- `point_count_score` is only a length overlap ratio, so dropping 5 points and dropping 150 points can both look moderate rather than catastrophic."
    )
    lines.append(
        "- `point_value_score` only evaluates the aligned overlap prefix from the environment parser, so it under-prices missing tails when the model gets early points roughly right."
    )
    lines.append(
        "- The tables below quantify the true x-aligned error: missing target x values, extra predicted x values, and normalized y error at each matched x."
    )
    lines.append("")

    for run in run_specs:
        samples = analyses_by_run[run.label]
        lines.append(f"## {run.label} Samples")
        lines.append("")
        unique_rows = sorted({sample.row_id for sample in samples})
        lines.append(f"Unique chart examples represented: `{len(unique_rows)}`")
        lines.append("")
        lines.extend(render_full_table(samples))
        lines.append("")

        high_reward_wrong = sorted(
            [sample for sample in samples if sample.reward >= 0.9 and sample.materially_wrong],
            key=lambda sample: (sample.reward, -(sample.missing_points + sample.material_y_errors)),
            reverse=True,
        )
        if high_reward_wrong:
            lines.append("### High-Reward Samples That Are Still Materially Wrong")
            lines.append("")
            lines.append(
                "| Row ID | Reward | Missing | Extra | Material y errs | Severe y errs | point_count_score | point_value_score | norm y MAE |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for sample in high_reward_wrong[:10]:
                lines.append(
                    f"| `{sample.row_id}` | {sample.reward:.4f} | {sample.missing_points} | {sample.extra_points} | "
                    f"{sample.material_y_errors} | {sample.severe_y_errors} | {sample.point_count_score:.4f} | "
                    f"{sample.point_value_score:.4f} | {sample.normalized_y_mae:.4f} |"
                )
            lines.append("")

        data_profile_counts = Counter(sample.data_profile for sample in samples)
        lines.append("### Data Profile Mix In Saved Rollouts")
        lines.append("")
        lines.append(
            ", ".join(f"`{profile}`: {count}" for profile, count in sorted(data_profile_counts.items()))
        )
        lines.append("")

        lines.append(f"### Detailed Sample Analysis ({min(detailed_per_run, len(samples))} cases)")
        lines.append("")
        for sample in select_detailed_samples(samples, detailed_per_run):
            lines.extend(render_detailed_section(sample))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_specs = [parse_run_spec(raw) for raw in args.run]
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    dataset_local_path = Path(args.dataset_local_path)

    steps = [args.step] if args.step is not None else shared_steps(run_specs)
    print(f"Analyzing shared steps: {steps}")
    answer_index, dataset = build_answer_index(dataset_local_path)

    analyses_by_run: dict[str, list[SampleAnalysis]] = {}
    for run in run_specs:
        print(f"Processing run {run.label} ({run.run_id})")
        run_samples: list[SampleAnalysis] = []
        for step in steps:
            print(f"  step {step}")
            rollouts = fetch_all_rollouts(run.run_id, step)
            run_samples.extend(
                analyze_sample(
                    sample=sample,
                    run_spec=run,
                    step=step,
                    dataset=dataset,
                    answer_index=answer_index,
                    assets_dir=assets_dir,
                )
                for sample in rollouts
            )
        analyses_by_run[run.label] = run_samples

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    render_report(
        run_specs=run_specs,
        steps=steps,
        analyses_by_run=analyses_by_run,
        output_path=output_dir / "REPORT.md",
        generated_at=generated_at,
        detailed_per_run=args.detailed_per_run,
    )


if __name__ == "__main__":
    main()
