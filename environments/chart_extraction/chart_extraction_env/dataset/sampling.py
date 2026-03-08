from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

from .models import (
    AnnotationProfile,
    ChartSpec,
    ChartType,
    DataProfile,
    ImageSizeProfile,
    LabelProfile,
    LayoutProfile,
    StyleProfile,
    XMode,
    YScaleProfile,
)

T = TypeVar("T")

CHART_TYPE_PATTERN = [
    ChartType.LINE,
    ChartType.BAR,
]

DATA_PROFILE_PATTERN = [
    DataProfile.MONO_UP,
    DataProfile.MONO_DOWN,
    DataProfile.SINGLE_PEAK,
    DataProfile.SINGLE_VALLEY,
    DataProfile.ZIGZAG,
    DataProfile.FLAT,
    DataProfile.RANDOM_WALK,
]

LABEL_PROFILE_PATTERN = [
    LabelProfile.SHORT,
    LabelProfile.MEDIUM,
    LabelProfile.MEDIUM,
    LabelProfile.LONG,
]

STYLE_PROFILE_PATTERN = [
    StyleProfile.CLEAN,
    StyleProfile.GRID,
    StyleProfile.MARKERS,
    StyleProfile.GRID_MARKERS,
]

ANNOTATION_PROFILE_PATTERN = [
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.ENDPOINT_LABELS,
]

LAYOUT_PROFILE_PATTERN = [
    LayoutProfile.SQUARE,
    LayoutProfile.WIDE,
    LayoutProfile.WIDE,
]

Y_SCALE_PROFILE_PATTERN = [
    YScaleProfile.NEAR_FLAT,
    YScaleProfile.SMALL_RANGE,
    YScaleProfile.MEDIUM_RANGE,
    YScaleProfile.LARGE_RANGE,
]

SHORT_SERIES_NAMES = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "North",
    "South",
    "Core",
    "Edge",
]

MEDIUM_SERIES_NAMES = [
    "Monthly Signups",
    "Daily Sales",
    "Usage Rate",
    "Support Tickets",
    "Conversion Rate",
    "Active Users",
    "Revenue Run Rate",
    "Response Time",
]

LONG_SERIES_NAMES = [
    "North America Revenue",
    "Europe Middle East and Africa Revenue",
    "Average Session Duration",
    "Customer Satisfaction Score",
    "Quarterly Fulfillment Lead Time",
    "Monthly Returning User Share",
]

SHORT_CATEGORY_LABELS = list("ABCDEFGHJKLMNPQRST")

MEDIUM_CATEGORY_LABELS = [
    "North",
    "South",
    "East",
    "West",
    "Retail",
    "Growth",
    "Core",
    "Prime",
    "Legacy",
]

LONG_CATEGORY_LABELS = [
    "North America",
    "Latin America",
    "Asia-Pacific",
    "Europe, Middle East, and Africa",
    "Self-Serve Customers",
    "Enterprise Accounts",
]

LINE_TITLES = [
    "Monthly Signups",
    "Weekly Active Users",
    "Fulfillment Time Trend",
    "Average Order Value",
]

BAR_TITLES = [
    "Revenue by Region",
    "Units Sold by Segment",
    "Ticket Volume by Queue",
    "Conversion Rate by Channel",
]

Y_AXIS_LABELS = [
    "Value",
    "Units",
    "Score",
    "Rate",
]

COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
]

FIGURE_SIZES: dict[ImageSizeProfile, tuple[tuple[float, float], int]] = {
    ImageSizeProfile.SMALL_SQUARE: ((4.0, 4.0), 120),
    ImageSizeProfile.MEDIUM_SQUARE: ((5.2, 5.2), 140),
    ImageSizeProfile.MEDIUM_WIDE: ((6.4, 4.0), 150),
    ImageSizeProfile.LARGE_WIDE: ((7.6, 4.6), 160),
    ImageSizeProfile.MEDIUM_TALL: ((4.8, 6.6), 150),
}

LAYOUT_TO_IMAGE_SIZES: dict[LayoutProfile, list[ImageSizeProfile]] = {
    LayoutProfile.SQUARE: [
        ImageSizeProfile.SMALL_SQUARE,
        ImageSizeProfile.MEDIUM_SQUARE,
    ],
    LayoutProfile.WIDE: [
        ImageSizeProfile.MEDIUM_WIDE,
        ImageSizeProfile.LARGE_WIDE,
    ],
    LayoutProfile.TALL: [
        ImageSizeProfile.MEDIUM_TALL,
    ],
}


def sample_chart_spec(rng: random.Random, example_index: int) -> ChartSpec:
    chart_type = _pattern_choice(CHART_TYPE_PATTERN, example_index)
    data_profile = _pattern_choice(DATA_PROFILE_PATTERN, example_index)
    label_profile = _pattern_choice(LABEL_PROFILE_PATTERN, example_index)
    style_profile = _pattern_choice(STYLE_PROFILE_PATTERN, example_index)
    annotation_profile = _pattern_choice(ANNOTATION_PROFILE_PATTERN, example_index)
    layout_profile = _pattern_choice(LAYOUT_PROFILE_PATTERN, example_index)
    y_scale_profile = _pattern_choice(Y_SCALE_PROFILE_PATTERN, example_index)
    image_size_profile = _pattern_choice(
        LAYOUT_TO_IMAGE_SIZES[layout_profile],
        example_index,
    )

    num_points = sample_num_points(
        rng=rng,
        chart_type=chart_type,
        label_profile=label_profile,
        annotation_profile=annotation_profile,
        image_size_profile=image_size_profile,
    )
    return ChartSpec(
        chart_type=chart_type,
        data_profile=data_profile,
        x_mode=XMode.NUMERIC_EVEN if chart_type == ChartType.LINE else XMode.CATEGORICAL_ORDERED,
        label_profile=label_profile,
        style_profile=style_profile,
        annotation_profile=annotation_profile,
        layout_profile=layout_profile,
        image_size_profile=image_size_profile,
        y_scale_profile=y_scale_profile,
        num_points=num_points,
        seed=rng.randrange(1_000_000_000),
    )


def sample_num_points(
    *,
    rng: random.Random,
    chart_type: ChartType,
    label_profile: LabelProfile,
    annotation_profile: AnnotationProfile,
    image_size_profile: ImageSizeProfile,
) -> int:
    low = 4
    high = 12 if chart_type == ChartType.LINE else 10
    if label_profile == LabelProfile.LONG:
        high = min(high, 7)
    if annotation_profile != AnnotationProfile.NONE:
        high = min(high, 6)
    if image_size_profile == ImageSizeProfile.SMALL_SQUARE:
        high = min(high, 6)
    return rng.randint(low, high)


def sample_series_name(rng: random.Random, label_profile: LabelProfile) -> str:
    pools = {
        LabelProfile.SHORT: SHORT_SERIES_NAMES,
        LabelProfile.MEDIUM: MEDIUM_SERIES_NAMES,
        LabelProfile.LONG: LONG_SERIES_NAMES,
    }
    return rng.choice(pools[label_profile])


def sample_category_labels(
    rng: random.Random,
    label_profile: LabelProfile,
    count: int,
) -> list[str]:
    pools = {
        LabelProfile.SHORT: SHORT_CATEGORY_LABELS,
        LabelProfile.MEDIUM: MEDIUM_CATEGORY_LABELS,
        LabelProfile.LONG: LONG_CATEGORY_LABELS,
    }
    pool = pools[label_profile]
    if count <= len(pool):
        return rng.sample(pool, count)
    return [pool[index % len(pool)] for index in range(count)]


def sample_title(rng: random.Random, chart_type: ChartType) -> str | None:
    if rng.random() < 0.45:
        return None
    pool = LINE_TITLES if chart_type == ChartType.LINE else BAR_TITLES
    return rng.choice(pool)


def sample_y_axis_label(rng: random.Random) -> str:
    return rng.choice(Y_AXIS_LABELS)


def sample_color(rng: random.Random) -> str:
    return rng.choice(COLORS)


def _pattern_choice(pattern: Sequence[T], example_index: int) -> T:
    return pattern[example_index % len(pattern)]
