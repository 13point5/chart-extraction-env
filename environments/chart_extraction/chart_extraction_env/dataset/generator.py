from __future__ import annotations

import json
import random
from collections.abc import Mapping

import numpy as np

from .models import (
    CanonicalAnswer,
    ChartRecipe,
    ChartSpec,
    ChartType,
    ExampleInfo,
    GeneratedExample,
    LabelProfile,
    RenderPlan,
    SeriesData,
    SeriesPoint,
    StyleProfile,
    XType,
    YScaleProfile,
)
from .render import render_chart
from .sampling import (
    FIGURE_SIZES,
    sample_category_labels,
    sample_chart_spec,
    sample_color,
    sample_title,
    sample_y_axis_label,
)


def build_split_examples(
    *,
    split: str,
    num_examples: int,
    seed: int,
) -> list[GeneratedExample]:
    split_rng = random.Random(_split_seed(seed, split))
    examples: list[GeneratedExample] = []
    for example_index in range(num_examples):
        example_seed = split_rng.randrange(1_000_000_000)
        recipe = build_recipe(seed=example_seed, example_index=example_index)
        image_bytes = render_chart(recipe)
        info = ExampleInfo(
            split=split,
            chart_type=recipe.spec.chart_type,
            data_profile=recipe.spec.data_profile,
            x_mode=recipe.spec.x_mode,
            label_profile=recipe.spec.label_profile,
            style_profile=recipe.spec.style_profile,
            annotation_profile=recipe.spec.annotation_profile,
            layout_profile=recipe.spec.layout_profile,
            image_size_profile=recipe.spec.image_size_profile,
            y_scale_profile=recipe.spec.y_scale_profile,
            num_points=recipe.spec.num_points,
            figure_size_inches=recipe.render_plan.figure_size_inches,
            dpi=recipe.render_plan.dpi,
            seed=recipe.spec.seed,
        )
        examples.append(
            GeneratedExample(
                example_id=f"{split}-{example_index:05d}",
                image_bytes=image_bytes,
                answer=recipe.answer,
                info=info,
            )
        )
    return examples


def build_recipe(seed: int, example_index: int) -> ChartRecipe:
    rng = random.Random(seed)
    spec = sample_chart_spec(rng, example_index)
    render_plan = build_render_plan(spec=spec, rng=rng)
    series_name = "series_1"
    answer = build_answer(spec=spec, rng=rng, series_name=series_name)
    return ChartRecipe(
        spec=spec,
        render_plan=render_plan,
        series_name=series_name,
        answer=answer,
    )


def build_answer(
    *,
    spec: ChartSpec,
    rng: random.Random,
    series_name: str,
) -> CanonicalAnswer:
    values = generate_y_values(spec=spec, rng=rng)
    if spec.chart_type == ChartType.LINE:
        points = [
            SeriesPoint(x=float(index), y=value)
            for index, value in enumerate(values)
        ]
        return CanonicalAnswer(
            chart_type=spec.chart_type,
            x_type=XType.NUMERIC,
            series=[SeriesData(name=series_name, points=points)],
        )

    categories = sample_category_labels(rng, spec.label_profile, spec.num_points)
    points = [
        SeriesPoint(x=category, y=value)
        for category, value in zip(categories, values, strict=True)
    ]
    return CanonicalAnswer(
        chart_type=spec.chart_type,
        x_type=XType.CATEGORICAL,
        series=[SeriesData(name=series_name, points=points)],
    )


def build_render_plan(*, spec: ChartSpec, rng: random.Random) -> RenderPlan:
    figure_size_inches, dpi = FIGURE_SIZES[spec.image_size_profile]
    show_grid = spec.style_profile in {StyleProfile.GRID, StyleProfile.GRID_MARKERS}
    show_markers = spec.chart_type == ChartType.LINE and spec.style_profile in {
        StyleProfile.MARKERS,
        StyleProfile.GRID_MARKERS,
    }
    rotate_x_labels = spec.chart_type == ChartType.BAR and spec.label_profile == LabelProfile.LONG
    return RenderPlan(
        figure_size_inches=figure_size_inches,
        dpi=dpi,
        show_grid=show_grid,
        show_markers=show_markers,
        rotate_x_labels=rotate_x_labels,
        color_hex=sample_color(rng),
        title=sample_title(rng, spec.chart_type),
        x_label="Index" if spec.chart_type == ChartType.LINE else "Category",
        y_label=sample_y_axis_label(rng),
    )


def generate_y_values(*, spec: ChartSpec, rng: random.Random) -> list[float]:
    raw = base_profile_values(profile=spec.data_profile, num_points=spec.num_points, rng=rng)
    scaled = apply_y_scale_profile(values=raw, profile=spec.y_scale_profile, rng=rng)
    rounded = [round(value, 3) for value in scaled]
    if spec.chart_type == ChartType.BAR:
        return [round(max(value, 0.2), 3) for value in rounded]
    return rounded


def base_profile_values(
    *,
    profile,
    num_points: int,
    rng: random.Random,
) -> np.ndarray:
    x = np.arange(num_points, dtype=float)

    if profile.name == "MONO_UP":
        increments = np.array([rng.uniform(0.4, 1.3) for _ in range(num_points)])
        return np.cumsum(increments)

    if profile.name == "MONO_DOWN":
        increments = np.array([rng.uniform(0.4, 1.3) for _ in range(num_points)])
        return np.cumsum(increments)[::-1]

    if profile.name == "SINGLE_PEAK":
        center = (num_points - 1) / 2.0
        values = 1.0 - np.abs(x - center) / max(center, 1.0)
        noise = np.array([rng.uniform(-0.05, 0.05) for _ in range(num_points)])
        return values + noise

    if profile.name == "SINGLE_VALLEY":
        center = (num_points - 1) / 2.0
        values = np.abs(x - center) / max(center, 1.0)
        noise = np.array([rng.uniform(-0.05, 0.05) for _ in range(num_points)])
        return values + noise

    if profile.name == "ZIGZAG":
        baseline = np.linspace(0.2, 1.0, num_points)
        offsets = np.array([0.28 if index % 2 == 0 else -0.22 for index in range(num_points)])
        noise = np.array([rng.uniform(-0.03, 0.03) for _ in range(num_points)])
        return baseline + offsets + noise

    if profile.name == "FLAT":
        center = rng.uniform(0.45, 0.55)
        noise = np.array([rng.uniform(-0.03, 0.03) for _ in range(num_points)])
        trend = np.linspace(0.0, rng.uniform(-0.03, 0.03), num_points)
        return np.full(num_points, center) + noise + trend

    if profile.name == "RANDOM_WALK":
        increments = np.array([rng.uniform(-0.18, 0.18) for _ in range(num_points)])
        return np.cumsum(increments)

    raise ValueError(f"Unsupported data profile: {profile}")


def apply_y_scale_profile(
    *,
    values: np.ndarray,
    profile: YScaleProfile,
    rng: random.Random,
) -> list[float]:
    minimum = float(values.min())
    maximum = float(values.max())
    spread = max(maximum - minimum, 1e-6)
    normalized = (values - minimum) / spread

    profile_ranges: dict[YScaleProfile, tuple[float, float]] = {
        YScaleProfile.NEAR_FLAT: (8.0, 8.8),
        YScaleProfile.SMALL_RANGE: (2.0, 6.0),
        YScaleProfile.MEDIUM_RANGE: (10.0, 25.0),
        YScaleProfile.LARGE_RANGE: (20.0, 90.0),
    }
    lower, upper = profile_ranges[profile]
    amplitude = upper - lower
    base = lower + rng.uniform(0.0, max(amplitude * 0.15, 0.05))
    scaled = base + normalized * amplitude
    if profile == YScaleProfile.NEAR_FLAT:
        scaled = base + normalized * rng.uniform(0.15, 0.6)
    return scaled.tolist()


def canonical_answer_json(answer: CanonicalAnswer) -> str:
    return json.dumps(answer.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def canonical_info_json(info: ExampleInfo) -> str:
    return json.dumps(info.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _split_seed(seed: int, split: str) -> int:
    offsets: Mapping[str, int] = {
        "train": 0,
        "validation": 10_000,
        "test": 20_000,
    }
    return seed + offsets.get(split, 100_000)
