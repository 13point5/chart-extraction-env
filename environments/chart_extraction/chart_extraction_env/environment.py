from __future__ import annotations

from datasets import Dataset
import verifiers as vf

from chart_extraction_env.dataset.hf import load_chart_dataset
from chart_extraction_env.dataset.models import CanonicalAnswer, DatasetVariant, OutputMode, XType
from chart_extraction_env.parsing import (
    extract_assistant_text,
    normalized_label,
    parse_canonical_answer_json,
    parse_response,
)
from chart_extraction_env.prompts import build_user_prompt


def load_environment(
    *,
    split: str = "train",
    eval_split: str | None = None,
    output_mode: str = OutputMode.JSON.value,
    dataset_variant: str = DatasetVariant.V1.value,
    chart_type_filter: str | None = None,
    dataset_local_path: str | None = None,
    dataset_repo_id: str | None = None,
    max_examples: int = -1,
    seed: int = 7,
    default_examples: int = 64,
) -> vf.Environment:
    resolved_output_mode = OutputMode(output_mode)
    resolved_dataset_variant = DatasetVariant(dataset_variant)
    resolved_eval_split = eval_split or ("validation" if split == "train" else split)
    train_dataset_source = load_chart_dataset(
        split=split,
        local_path=dataset_local_path,
        repo_id=dataset_repo_id,
        seed=seed,
        default_examples=default_examples,
        variant=resolved_dataset_variant,
        chart_type_filter=chart_type_filter,
    )
    eval_dataset_source = load_chart_dataset(
        split=resolved_eval_split,
        local_path=dataset_local_path,
        repo_id=dataset_repo_id,
        seed=seed,
        default_examples=default_examples,
        variant=resolved_dataset_variant,
        chart_type_filter=chart_type_filter,
    )
    if max_examples > 0:
        train_limit = min(max_examples, len(train_dataset_source))
        eval_limit = min(max_examples, len(eval_dataset_source))
        train_dataset_source = train_dataset_source.select(range(train_limit))
        eval_dataset_source = eval_dataset_source.select(range(eval_limit))

    dataset = Dataset.from_list(
        [
            {
                "prompt": build_user_prompt(image=row["image"], output_mode=resolved_output_mode),
                "answer": row["answer"],
                "info": row["info"],
            }
            for row in train_dataset_source
        ]
    )
    eval_dataset = Dataset.from_list(
        [
            {
                "prompt": build_user_prompt(image=row["image"], output_mode=resolved_output_mode),
                "answer": row["answer"],
                "info": row["info"],
            }
            for row in eval_dataset_source
        ]
    )

    async def format_valid(completion) -> float:
        try:
            parse_response(extract_assistant_text(completion), resolved_output_mode)
        except Exception:
            return 0.0
        return 1.0

    async def chart_type_score(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        return 1.0 if parsed.chart_type == target.chart_type else 0.0

    async def series_name_score(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        parsed_name = normalized_label(parsed.series[0].name)
        target_name = normalized_label(target.series[0].name)
        return 1.0 if parsed_name == target_name else 0.0

    async def point_count_score(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        parsed_count = len(parsed.series[0].points)
        target_count = len(target.series[0].points)
        return min(parsed_count, target_count) / max(parsed_count, target_count)

    async def point_value_score(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        return score_points(parsed, target)

    async def x_type_score_metric(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        return 1.0 if parsed.x_type == target.x_type else 0.0

    async def point_x_score_metric(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        return point_component_scores(parsed, target)["point_x_score"]

    async def point_y_score_metric(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        return point_component_scores(parsed, target)["point_y_score"]

    async def exact_answer_score_metric(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 0.0
        return 1.0 if parsed.model_dump(mode="json") == target.model_dump(mode="json") else 0.0

    async def normalized_y_mae_metric(completion, answer) -> float:
        parsed, target = _parse_pair(completion, answer, resolved_output_mode)
        if parsed is None or target is None:
            return 1.0
        return point_component_scores(parsed, target)["normalized_y_mae"]

    rubric = vf.Rubric(
        funcs=[
            format_valid,
            chart_type_score,
            series_name_score,
            point_count_score,
            point_value_score,
            x_type_score_metric,
            point_x_score_metric,
            point_y_score_metric,
            exact_answer_score_metric,
            normalized_y_mae_metric,
        ],
        weights=[0.1, 0.15, 0.15, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(dataset=dataset, eval_dataset=eval_dataset, rubric=rubric)


def score_points(parsed: CanonicalAnswer, target: CanonicalAnswer) -> float:
    return point_component_scores(parsed, target)["point_value_score"]


def point_component_scores(parsed: CanonicalAnswer, target: CanonicalAnswer) -> dict[str, float]:
    parsed_points = parsed.series[0].points
    target_points = target.series[0].points
    overlap = min(len(parsed_points), len(target_points))
    if overlap == 0:
        return {
            "point_value_score": 0.0,
            "point_x_score": 0.0,
            "point_y_score": 0.0,
            "normalized_y_mae": 1.0,
        }

    y_values = [point.y for point in target_points]
    y_scale = max(max(y_values) - min(y_values), 1.0)

    point_scores: list[float] = []
    x_scores: list[float] = []
    y_scores: list[float] = []
    y_errors: list[float] = []
    for parsed_point, target_point in zip(parsed_points[:overlap], target_points[:overlap], strict=True):
        if target.x_type == XType.NUMERIC:
            x_error = abs(float(parsed_point.x) - float(target_point.x))
            x_score = 1.0 if x_error <= 1e-3 else max(0.0, 1.0 - x_error)
        else:
            x_score = 1.0 if normalized_label(str(parsed_point.x)) == normalized_label(str(target_point.x)) else 0.0
        y_error = abs(parsed_point.y - target_point.y)
        y_score = max(0.0, 1.0 - (y_error / y_scale))
        x_scores.append(x_score)
        y_scores.append(y_score)
        y_errors.append(y_error / y_scale)
        point_scores.append((0.4 * x_score) + (0.6 * y_score))

    coverage = overlap / max(len(parsed_points), len(target_points))
    return {
        "point_value_score": (sum(point_scores) / len(point_scores)) * coverage,
        "point_x_score": (sum(x_scores) / len(x_scores)) * coverage,
        "point_y_score": (sum(y_scores) / len(y_scores)) * coverage,
        "normalized_y_mae": sum(y_errors) / len(y_errors),
    }


def _parse_pair(
    completion,
    answer: str,
    output_mode: OutputMode,
) -> tuple[CanonicalAnswer | None, CanonicalAnswer | None]:
    try:
        parsed = parse_response(extract_assistant_text(completion), output_mode)
        target = parse_canonical_answer_json(answer)
    except Exception:
        return None, None
    return parsed, target
