from __future__ import annotations

import json
import random
from collections.abc import Mapping

import numpy as np

from .models import (
    AnnotationProfile,
    CanonicalAnswer,
    ChartRecipe,
    ChartSpec,
    ChartType,
    DataProfile,
    DatasetVariant,
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
    sample_x_axis_label,
    sample_y_axis_label,
)


def build_split_examples(
    *,
    split: str,
    num_examples: int,
    seed: int,
    variant: DatasetVariant = DatasetVariant.V1,
    version: str | None = None,
) -> list[GeneratedExample]:
    split_rng = random.Random(_split_seed(seed, split))
    resolved_version = version or variant.value
    examples: list[GeneratedExample] = []
    for example_index in range(num_examples):
        example_seed = split_rng.randrange(1_000_000_000)
        recipe = build_recipe(
            seed=example_seed,
            example_index=example_index,
            variant=variant,
        )
        image_bytes = render_chart(recipe)
        info = ExampleInfo(
            version=resolved_version,
            variant=recipe.spec.variant,
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


def build_recipe(
    *,
    seed: int,
    example_index: int,
    variant: DatasetVariant = DatasetVariant.V1,
) -> ChartRecipe:
    rng = random.Random(seed)
    spec = sample_chart_spec(rng, example_index, variant=variant)
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

    categories = sample_category_labels(
        rng,
        spec.label_profile,
        spec.num_points,
        variant=spec.variant,
    )
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
    show_grid = spec.style_profile in {
        StyleProfile.GRID,
        StyleProfile.GRID_MARKERS,
        StyleProfile.ML_DENSE,
        StyleProfile.BUSINESS_GRID,
        StyleProfile.BUSINESS_LABELS,
    }
    show_markers = spec.chart_type == ChartType.LINE and spec.style_profile in {
        StyleProfile.MARKERS,
        StyleProfile.GRID_MARKERS,
    }
    if spec.variant == DatasetVariant.DENSE_NOISY_V1 and spec.num_points > 48:
        show_markers = False

    rotate_x_labels = spec.chart_type == ChartType.BAR and (
        spec.label_profile == LabelProfile.LONG or spec.num_points > 12
    )
    line_width = 2.2
    marker_size = 6.0
    if spec.chart_type == ChartType.LINE:
        if spec.num_points >= 320:
            line_width = 1.0
            marker_size = 0.0
        elif spec.num_points >= 160:
            line_width = 1.2
            marker_size = 2.0
        elif spec.num_points >= 80:
            line_width = 1.5
            marker_size = 3.0

    return RenderPlan(
        figure_size_inches=figure_size_inches,
        dpi=dpi,
        show_grid=show_grid,
        show_markers=show_markers,
        rotate_x_labels=rotate_x_labels,
        color_hex=sample_color(rng),
        line_width=line_width,
        marker_size=marker_size,
        title=sample_title(
            rng,
            spec.chart_type,
            data_profile=spec.data_profile,
            variant=spec.variant,
        ),
        x_label=sample_x_axis_label(
            rng,
            spec.chart_type,
            data_profile=spec.data_profile,
            variant=spec.variant,
        ),
        y_label=sample_y_axis_label(
            rng,
            chart_type=spec.chart_type,
            data_profile=spec.data_profile,
            variant=spec.variant,
        ),
    )


def generate_y_values(*, spec: ChartSpec, rng: random.Random) -> list[float]:
    if spec.chart_type == ChartType.LINE:
        raw = line_profile_values(profile=spec.data_profile, num_points=spec.num_points, rng=rng)
    else:
        raw = bar_profile_values(profile=spec.data_profile, num_points=spec.num_points, rng=rng)
    scaled = apply_y_scale_profile(
        values=raw,
        profile=spec.y_scale_profile,
        rng=rng,
        variant=spec.variant,
    )
    rounded = [round(value, 3) for value in scaled]
    if spec.chart_type == ChartType.BAR:
        return [round(max(value, 0.1), 3) for value in rounded]
    return rounded


def line_profile_values(
    *,
    profile: DataProfile,
    num_points: int,
    rng: random.Random,
) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randrange(1_000_000_000))
    x = np.linspace(0.0, 1.0, num_points, dtype=float)

    if profile == DataProfile.MONO_UP:
        increments = np_rng.uniform(0.4, 1.3, size=num_points)
        return np.cumsum(increments)

    if profile == DataProfile.MONO_DOWN:
        increments = np_rng.uniform(0.4, 1.3, size=num_points)
        return np.cumsum(increments)[::-1]

    if profile == DataProfile.SINGLE_PEAK:
        center = 0.5
        values = 1.0 - np.abs(x - center) / max(center, 1e-6)
        noise = np_rng.normal(0.0, 0.05, size=num_points)
        return values + noise

    if profile == DataProfile.SINGLE_VALLEY:
        center = 0.5
        values = np.abs(x - center) / max(center, 1e-6)
        noise = np_rng.normal(0.0, 0.05, size=num_points)
        return values + noise

    if profile == DataProfile.ZIGZAG:
        baseline = np.linspace(0.2, 1.0, num_points)
        offsets = np.array([0.28 if index % 2 == 0 else -0.22 for index in range(num_points)])
        noise = np_rng.normal(0.0, 0.03, size=num_points)
        return baseline + offsets + noise

    if profile == DataProfile.FLAT:
        center = rng.uniform(0.45, 0.55)
        noise = np_rng.normal(0.0, 0.03, size=num_points)
        trend = np.linspace(0.0, rng.uniform(-0.03, 0.03), num_points)
        return np.full(num_points, center) + noise + trend

    if profile == DataProfile.RANDOM_WALK:
        drift = np.linspace(rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), num_points)
        increments = np_rng.normal(0.0, 0.08, size=num_points)
        return np.cumsum(increments) + drift

    if profile == DataProfile.NOISY_TREND_UP:
        trend = np.linspace(0.0, 1.0, num_points)
        wave = 0.10 * np.sin((2.0 * np.pi * rng.uniform(1.0, 3.0) * x) + rng.uniform(0.0, 2.0 * np.pi))
        noise = np_rng.normal(0.0, 0.07, size=num_points)
        return trend + wave + noise

    if profile == DataProfile.NOISY_TREND_DOWN:
        trend = np.linspace(1.0, 0.0, num_points)
        wave = 0.10 * np.sin((2.0 * np.pi * rng.uniform(1.0, 3.0) * x) + rng.uniform(0.0, 2.0 * np.pi))
        noise = np_rng.normal(0.0, 0.07, size=num_points)
        return trend + wave + noise

    if profile == DataProfile.LOSS_DECAY:
        decay_rate = rng.uniform(2.4, 5.8)
        curve = np.exp(-decay_rate * x)
        noise = np_rng.normal(0.0, 0.03, size=num_points)
        return curve + noise

    if profile == DataProfile.GROWTH_SATURATING:
        growth_rate = rng.uniform(2.0, 5.0)
        curve = 1.0 - np.exp(-growth_rate * x)
        noise = np_rng.normal(0.0, 0.03, size=num_points)
        return curve + noise

    if profile == DataProfile.SEASONAL:
        cycles = rng.uniform(2.0, 6.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        seasonal = 0.35 * np.sin((2.0 * np.pi * cycles * x) + phase)
        harmonic = 0.12 * np.sin((2.0 * np.pi * cycles * 2.0 * x) + (phase / 2.0))
        trend = np.linspace(rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18), num_points)
        noise = np_rng.normal(0.0, 0.05, size=num_points)
        return seasonal + harmonic + trend + noise

    if profile == DataProfile.SPIKY:
        baseline = np.cumsum(np_rng.normal(0.0, 0.03, size=num_points))
        spike_count = max(2, num_points // 60)
        spike_positions = np_rng.choice(num_points, size=spike_count, replace=False)
        spikes = np.zeros(num_points)
        spikes[spike_positions] = np_rng.normal(0.85, 0.35, size=spike_count) * np_rng.choice(
            [-1.0, 1.0],
            size=spike_count,
        )
        kernel = np.array([0.25, 0.9, 0.25])
        return baseline + np.convolve(spikes, kernel, mode="same") + np_rng.normal(0.0, 0.03, size=num_points)

    if profile == DataProfile.STEP_CHANGE:
        values = np.zeros(num_points)
        cut_count = 2 if num_points >= 60 else 1
        candidate_positions = np.arange(max(3, num_points // 8), max(num_points - 3, 4))
        cut_positions = sorted(np_rng.choice(candidate_positions, size=cut_count, replace=False).tolist())
        boundaries = [0, *cut_positions, num_points]
        current_level = rng.uniform(0.15, 0.35)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            current_level += rng.uniform(-0.18, 0.38)
            values[left:right] = current_level
        trend = np.linspace(0.0, rng.uniform(-0.08, 0.08), num_points)
        noise = np_rng.normal(0.0, 0.025, size=num_points)
        return values + trend + noise

    raise ValueError(f"Unsupported line data profile: {profile}")


def bar_profile_values(
    *,
    profile: DataProfile,
    num_points: int,
    rng: random.Random,
) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randrange(1_000_000_000))

    if profile == DataProfile.BUSINESS_RANDOM:
        baseline = rng.uniform(0.45, 0.8)
        return baseline + np_rng.normal(0.0, 0.18, size=num_points)

    if profile == DataProfile.BUSINESS_SORTED:
        values = np_rng.normal(0.75, 0.2, size=num_points)
        values = np.sort(values)
        if rng.random() < 0.5:
            values = values[::-1]
        return values

    if profile == DataProfile.BUSINESS_OUTLIER:
        values = np_rng.normal(0.55, 0.12, size=num_points)
        standout_count = 1 if num_points < 24 else 2
        standout_indices = np_rng.choice(num_points, size=standout_count, replace=False)
        values[standout_indices] += np_rng.uniform(0.7, 1.3, size=standout_count)
        return values

    if profile == DataProfile.MONO_UP:
        values = np.sort(np_rng.normal(0.6, 0.18, size=num_points))
        return values

    if profile == DataProfile.MONO_DOWN:
        values = np.sort(np_rng.normal(0.6, 0.18, size=num_points))[::-1]
        return values

    if profile == DataProfile.SINGLE_PEAK:
        x = np.linspace(-1.0, 1.0, num_points)
        return 1.1 - np.abs(x) + np_rng.normal(0.0, 0.05, size=num_points)

    if profile == DataProfile.SINGLE_VALLEY:
        x = np.linspace(-1.0, 1.0, num_points)
        return np.abs(x) + np_rng.normal(0.0, 0.05, size=num_points)

    if profile == DataProfile.ZIGZAG:
        base = np.linspace(0.3, 1.0, num_points)
        offsets = np.array([0.15 if index % 2 == 0 else -0.1 for index in range(num_points)])
        return base + offsets + np_rng.normal(0.0, 0.03, size=num_points)

    if profile == DataProfile.FLAT:
        return np.full(num_points, rng.uniform(0.5, 0.7)) + np_rng.normal(0.0, 0.04, size=num_points)

    raise ValueError(f"Unsupported bar data profile: {profile}")


def apply_y_scale_profile(
    *,
    values: np.ndarray,
    profile: YScaleProfile,
    rng: random.Random,
    variant: DatasetVariant,
) -> list[float]:
    minimum = float(values.min())
    maximum = float(values.max())
    spread = max(maximum - minimum, 1e-6)
    normalized = (values - minimum) / spread

    if variant == DatasetVariant.DENSE_NOISY_V1:
        profile_ranges: dict[YScaleProfile, tuple[float, float]] = {
            YScaleProfile.NEAR_FLAT: (0.2, 1.1),
            YScaleProfile.SMALL_RANGE: (1.0, 12.0),
            YScaleProfile.MEDIUM_RANGE: (15.0, 180.0),
            YScaleProfile.LARGE_RANGE: (200.0, 1500.0),
        }
    else:
        profile_ranges = {
            YScaleProfile.NEAR_FLAT: (8.0, 8.8),
            YScaleProfile.SMALL_RANGE: (2.0, 6.0),
            YScaleProfile.MEDIUM_RANGE: (10.0, 25.0),
            YScaleProfile.LARGE_RANGE: (20.0, 90.0),
        }
    lower, upper = profile_ranges[profile]
    amplitude = upper - lower
    base = lower + rng.uniform(0.0, max(amplitude * 0.12, 0.05))
    scaled = base + normalized * amplitude
    if profile == YScaleProfile.NEAR_FLAT:
        flat_amplitude = rng.uniform(0.05, 0.18) if variant == DatasetVariant.DENSE_NOISY_V1 else rng.uniform(0.15, 0.6)
        scaled = base + normalized * flat_amplitude
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
