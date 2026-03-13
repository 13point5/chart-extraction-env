from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

from .models import (
    AnnotationProfile,
    ChartSpec,
    ChartType,
    DataProfile,
    DatasetVariant,
    ImageSizeProfile,
    LabelProfile,
    LayoutProfile,
    StyleProfile,
    XMode,
    YScaleProfile,
)

T = TypeVar("T")

V1_CHART_TYPE_PATTERN = [
    ChartType.LINE,
    ChartType.BAR,
]

V1_DATA_PROFILE_PATTERN = [
    DataProfile.MONO_UP,
    DataProfile.MONO_DOWN,
    DataProfile.SINGLE_PEAK,
    DataProfile.SINGLE_VALLEY,
    DataProfile.ZIGZAG,
    DataProfile.FLAT,
    DataProfile.RANDOM_WALK,
]

V1_LABEL_PROFILE_PATTERN = [
    LabelProfile.SHORT,
    LabelProfile.MEDIUM,
    LabelProfile.MEDIUM,
    LabelProfile.LONG,
]

V1_STYLE_PROFILE_PATTERN = [
    StyleProfile.CLEAN,
    StyleProfile.GRID,
    StyleProfile.MARKERS,
    StyleProfile.GRID_MARKERS,
]

V1_ANNOTATION_PROFILE_PATTERN = [
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.ENDPOINT_LABELS,
]

V1_LAYOUT_PROFILE_PATTERN = [
    LayoutProfile.SQUARE,
    LayoutProfile.WIDE,
    LayoutProfile.WIDE,
]

V1_Y_SCALE_PROFILE_PATTERN = [
    YScaleProfile.NEAR_FLAT,
    YScaleProfile.SMALL_RANGE,
    YScaleProfile.MEDIUM_RANGE,
    YScaleProfile.LARGE_RANGE,
]

DENSE_CHART_TYPE_PATTERN = [
    ChartType.LINE,
    ChartType.LINE,
    ChartType.LINE,
    ChartType.LINE,
    ChartType.LINE,
    ChartType.LINE,
    ChartType.LINE,
    ChartType.BAR,
    ChartType.BAR,
    ChartType.BAR,
]

DENSE_LINE_PROFILE_PATTERN = [
    DataProfile.NOISY_TREND_UP,
    DataProfile.NOISY_TREND_DOWN,
    DataProfile.RANDOM_WALK,
    DataProfile.SEASONAL,
    DataProfile.SPIKY,
    DataProfile.LOSS_DECAY,
    DataProfile.GROWTH_SATURATING,
    DataProfile.STEP_CHANGE,
    DataProfile.NOISY_TREND_UP,
    DataProfile.RANDOM_WALK,
]

DENSE_BAR_PROFILE_PATTERN = [
    DataProfile.BUSINESS_RANDOM,
    DataProfile.BUSINESS_SORTED,
    DataProfile.BUSINESS_OUTLIER,
    DataProfile.BUSINESS_RANDOM,
    DataProfile.BUSINESS_SORTED,
    DataProfile.FLAT,
]

DENSE_LABEL_PROFILE_PATTERN = [
    LabelProfile.SHORT,
    LabelProfile.SHORT,
    LabelProfile.MEDIUM,
    LabelProfile.MEDIUM,
    LabelProfile.LONG,
]

DENSE_LINE_STYLE_PATTERN = [
    StyleProfile.ML_DENSE,
    StyleProfile.ML_DENSE,
    StyleProfile.BUSINESS_GRID,
    StyleProfile.GRID,
    StyleProfile.CLEAN,
]

DENSE_BAR_STYLE_PATTERN = [
    StyleProfile.BUSINESS_GRID,
    StyleProfile.BUSINESS_GRID,
    StyleProfile.BUSINESS_LABELS,
    StyleProfile.CLEAN,
]

DENSE_LINE_ANNOTATION_PATTERN = [
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.ENDPOINT_LABELS,
]

DENSE_BAR_ANNOTATION_PATTERN = [
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.NONE,
    AnnotationProfile.ENDPOINT_LABELS,
]

DENSE_LINE_LAYOUT_PATTERN = [
    LayoutProfile.WIDE,
    LayoutProfile.WIDE,
    LayoutProfile.WIDE,
    LayoutProfile.TALL,
]

DENSE_BAR_LAYOUT_PATTERN = [
    LayoutProfile.WIDE,
    LayoutProfile.WIDE,
    LayoutProfile.SQUARE,
]

DENSE_LINE_Y_SCALE_PATTERN = [
    YScaleProfile.NEAR_FLAT,
    YScaleProfile.SMALL_RANGE,
    YScaleProfile.SMALL_RANGE,
    YScaleProfile.MEDIUM_RANGE,
    YScaleProfile.MEDIUM_RANGE,
    YScaleProfile.LARGE_RANGE,
]

DENSE_BAR_Y_SCALE_PATTERN = [
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

V1_SHORT_CATEGORY_LABELS = list("ABCDEFGHJKLMNPQRST")

V1_MEDIUM_CATEGORY_LABELS = [
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

V1_LONG_CATEGORY_LABELS = [
    "North America",
    "Latin America",
    "Asia-Pacific",
    "Europe, Middle East, and Africa",
    "Self-Serve Customers",
    "Enterprise Accounts",
]

DENSE_SHORT_CATEGORY_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
]

DENSE_MEDIUM_CATEGORY_LABELS = [
    "North America",
    "EMEA",
    "APAC",
    "LatAm",
    "Enterprise",
    "SMB",
    "Retail",
    "Partners",
    "Search",
    "Direct",
    "Email",
    "Field Sales",
]

DENSE_LONG_CATEGORY_LABELS = [
    "North America Enterprise",
    "Europe Mid-Market",
    "Asia-Pacific Self-Serve",
    "Latin America Expansion",
    "Digital Acquisition Team",
    "Marketplace Partnerships",
    "Customer Support Queue",
    "Strategic Accounts Group",
]

V1_LINE_TITLES = [
    "Monthly Signups",
    "Weekly Active Users",
    "Fulfillment Time Trend",
    "Average Order Value",
]

V1_BAR_TITLES = [
    "Revenue by Region",
    "Units Sold by Segment",
    "Ticket Volume by Queue",
    "Conversion Rate by Channel",
]

DENSE_LINE_TITLE_POOLS: dict[DataProfile, list[str]] = {
    DataProfile.NOISY_TREND_UP: [
        "Validation Accuracy Over Time",
        "Daily Active Users",
        "Inference Throughput",
    ],
    DataProfile.NOISY_TREND_DOWN: [
        "Support Backlog",
        "Median Latency",
        "Defect Rate",
    ],
    DataProfile.LOSS_DECAY: [
        "Training Loss",
        "Validation Loss",
        "Reconstruction Error",
    ],
    DataProfile.GROWTH_SATURATING: [
        "Accuracy by Epoch",
        "Recall by Iteration",
        "Coverage by Step",
    ],
    DataProfile.SEASONAL: [
        "Traffic Volume",
        "Orders per Day",
        "Weekly Conversion Rate",
    ],
    DataProfile.SPIKY: [
        "CPU Utilization",
        "Request Volume",
        "Alert Volume",
    ],
    DataProfile.STEP_CHANGE: [
        "Deployment Impact",
        "Retention After Launch",
        "Spend After Campaign Shift",
    ],
    DataProfile.RANDOM_WALK: [
        "Validation Metric",
        "Daily Revenue Trend",
        "Response Time",
    ],
}

DENSE_BAR_TITLES = [
    "Revenue by Region",
    "Units Sold by Segment",
    "Ticket Volume by Queue",
    "Pipeline by Channel",
    "Bookings by Product Line",
]

V1_Y_AXIS_LABELS = [
    "Value",
    "Units",
    "Score",
    "Rate",
]

DENSE_LINE_Y_AXIS_LABELS: dict[DataProfile, list[str]] = {
    DataProfile.NOISY_TREND_UP: ["Accuracy", "Users", "Throughput"],
    DataProfile.NOISY_TREND_DOWN: ["Latency (ms)", "Rate", "Open Tickets"],
    DataProfile.LOSS_DECAY: ["Loss", "Error", "Objective"],
    DataProfile.GROWTH_SATURATING: ["Accuracy", "Recall", "Coverage"],
    DataProfile.SEASONAL: ["Volume", "Orders", "Conversion Rate"],
    DataProfile.SPIKY: ["Utilization", "Requests", "Events"],
    DataProfile.STEP_CHANGE: ["Rate", "Spend", "Users"],
    DataProfile.RANDOM_WALK: ["Metric", "Revenue", "Score"],
}

DENSE_BAR_Y_AXIS_LABELS = [
    "Revenue ($K)",
    "Units Sold",
    "Pipeline ($K)",
    "Ticket Volume",
    "Bookings",
]

V1_LINE_X_AXIS_LABELS = ["Index"]
DENSE_LINE_X_AXIS_LABELS: dict[DataProfile, list[str]] = {
    DataProfile.NOISY_TREND_UP: ["Day", "Week", "Observation"],
    DataProfile.NOISY_TREND_DOWN: ["Day", "Week", "Observation"],
    DataProfile.LOSS_DECAY: ["Epoch", "Iteration", "Training Step"],
    DataProfile.GROWTH_SATURATING: ["Epoch", "Iteration", "Training Step"],
    DataProfile.SEASONAL: ["Day", "Week", "Observation"],
    DataProfile.SPIKY: ["Minute", "Step", "Observation"],
    DataProfile.STEP_CHANGE: ["Day", "Week", "Observation"],
    DataProfile.RANDOM_WALK: ["Step", "Day", "Observation"],
}

V1_BAR_X_AXIS_LABELS = ["Category"]
DENSE_BAR_X_AXIS_LABELS = [
    "Region",
    "Segment",
    "Channel",
    "Product Line",
    "Month",
]

COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#9c755f",
    "#bab0ab",
]

FIGURE_SIZES: dict[ImageSizeProfile, tuple[tuple[float, float], int]] = {
    ImageSizeProfile.SMALL_SQUARE: ((4.0, 4.0), 120),
    ImageSizeProfile.MEDIUM_SQUARE: ((5.2, 5.2), 140),
    ImageSizeProfile.MEDIUM_WIDE: ((6.4, 4.0), 150),
    ImageSizeProfile.LARGE_WIDE: ((7.6, 4.6), 160),
    ImageSizeProfile.XL_WIDE: ((10.4, 5.4), 170),
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

DENSE_LAYOUT_TO_IMAGE_SIZES: dict[LayoutProfile, list[ImageSizeProfile]] = {
    LayoutProfile.SQUARE: [
        ImageSizeProfile.MEDIUM_SQUARE,
    ],
    LayoutProfile.WIDE: [
        ImageSizeProfile.MEDIUM_WIDE,
        ImageSizeProfile.LARGE_WIDE,
        ImageSizeProfile.XL_WIDE,
    ],
    LayoutProfile.TALL: [
        ImageSizeProfile.MEDIUM_TALL,
    ],
}


def sample_chart_spec(
    rng: random.Random,
    example_index: int,
    *,
    variant: DatasetVariant = DatasetVariant.V1,
) -> ChartSpec:
    if variant == DatasetVariant.DENSE_NOISY_V1:
        return _sample_dense_chart_spec(rng, example_index)
    return _sample_v1_chart_spec(rng, example_index)


def _sample_v1_chart_spec(rng: random.Random, example_index: int) -> ChartSpec:
    chart_type = _pattern_choice(V1_CHART_TYPE_PATTERN, example_index)
    data_profile = _pattern_choice(V1_DATA_PROFILE_PATTERN, example_index)
    label_profile = _pattern_choice(V1_LABEL_PROFILE_PATTERN, example_index)
    style_profile = _pattern_choice(V1_STYLE_PROFILE_PATTERN, example_index)
    annotation_profile = _pattern_choice(V1_ANNOTATION_PROFILE_PATTERN, example_index)
    layout_profile = _pattern_choice(V1_LAYOUT_PROFILE_PATTERN, example_index)
    y_scale_profile = _pattern_choice(V1_Y_SCALE_PROFILE_PATTERN, example_index)
    image_size_profile = _pattern_choice(
        LAYOUT_TO_IMAGE_SIZES[layout_profile],
        example_index,
    )
    num_points = sample_num_points(
        rng=rng,
        variant=DatasetVariant.V1,
        chart_type=chart_type,
        label_profile=label_profile,
        style_profile=style_profile,
        annotation_profile=annotation_profile,
        image_size_profile=image_size_profile,
        example_index=example_index,
    )
    return ChartSpec(
        variant=DatasetVariant.V1,
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


def _sample_dense_chart_spec(rng: random.Random, example_index: int) -> ChartSpec:
    chart_type = _pattern_choice(DENSE_CHART_TYPE_PATTERN, example_index)
    label_profile = _pattern_choice(DENSE_LABEL_PROFILE_PATTERN, example_index)
    if chart_type == ChartType.LINE:
        data_profile = _pattern_choice(DENSE_LINE_PROFILE_PATTERN, example_index)
        style_profile = _pattern_choice(DENSE_LINE_STYLE_PATTERN, example_index)
        annotation_profile = _pattern_choice(DENSE_LINE_ANNOTATION_PATTERN, example_index)
        layout_profile = _pattern_choice(DENSE_LINE_LAYOUT_PATTERN, example_index)
        y_scale_profile = _pattern_choice(DENSE_LINE_Y_SCALE_PATTERN, example_index)
    else:
        data_profile = _pattern_choice(DENSE_BAR_PROFILE_PATTERN, example_index)
        style_profile = _pattern_choice(DENSE_BAR_STYLE_PATTERN, example_index)
        annotation_profile = _pattern_choice(DENSE_BAR_ANNOTATION_PATTERN, example_index)
        layout_profile = _pattern_choice(DENSE_BAR_LAYOUT_PATTERN, example_index)
        y_scale_profile = _pattern_choice(DENSE_BAR_Y_SCALE_PATTERN, example_index)

    image_size_profile = _pattern_choice(
        DENSE_LAYOUT_TO_IMAGE_SIZES[layout_profile],
        example_index,
    )
    num_points = sample_num_points(
        rng=rng,
        variant=DatasetVariant.DENSE_NOISY_V1,
        chart_type=chart_type,
        label_profile=label_profile,
        style_profile=style_profile,
        annotation_profile=annotation_profile,
        image_size_profile=image_size_profile,
        example_index=example_index,
    )
    return ChartSpec(
        variant=DatasetVariant.DENSE_NOISY_V1,
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
    variant: DatasetVariant,
    chart_type: ChartType,
    label_profile: LabelProfile,
    style_profile: StyleProfile,
    annotation_profile: AnnotationProfile,
    image_size_profile: ImageSizeProfile,
    example_index: int,
) -> int:
    if variant == DatasetVariant.DENSE_NOISY_V1:
        if chart_type == ChartType.LINE:
            point_bucket = _pattern_choice(
                ["compact", "medium", "medium", "large", "large", "xlarge", "large", "xlarge"],
                example_index,
            )
            bucket_ranges = {
                "compact": (32, 80),
                "medium": (81, 180),
                "large": (181, 340),
                "xlarge": (341, 500),
            }
            low, high = bucket_ranges[point_bucket]
            if image_size_profile == ImageSizeProfile.MEDIUM_WIDE:
                high = min(high, 380)
            if image_size_profile == ImageSizeProfile.MEDIUM_TALL:
                high = min(high, 260)
            if annotation_profile != AnnotationProfile.NONE:
                high = min(high, 96)
            if style_profile == StyleProfile.GRID_MARKERS:
                high = min(high, 96)
            low = min(low, high)
            return rng.randint(low, high)

        point_bucket = _pattern_choice(
            ["small", "medium", "medium", "large", "small", "medium"],
            example_index,
        )
        bucket_ranges = {
            "small": (6, 12),
            "medium": (13, 24),
            "large": (25, 60),
        }
        low, high = bucket_ranges[point_bucket]
        if label_profile == LabelProfile.LONG:
            high = min(high, 16)
        if annotation_profile != AnnotationProfile.NONE:
            high = min(high, 14)
        if image_size_profile == ImageSizeProfile.MEDIUM_SQUARE:
            high = min(high, 18)
        low = min(low, high)
        return rng.randint(low, high)

    low = 4
    high = 12 if chart_type == ChartType.LINE else 10
    if label_profile == LabelProfile.LONG:
        high = min(high, 7)
    if annotation_profile != AnnotationProfile.NONE:
        high = min(high, 6)
    if image_size_profile == ImageSizeProfile.SMALL_SQUARE:
        high = min(high, 6)
    if chart_type == ChartType.LINE and style_profile == StyleProfile.GRID_MARKERS:
        high = min(high, 10)
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
    *,
    variant: DatasetVariant = DatasetVariant.V1,
) -> list[str]:
    if variant == DatasetVariant.DENSE_NOISY_V1:
        pools = {
            LabelProfile.SHORT: DENSE_SHORT_CATEGORY_LABELS,
            LabelProfile.MEDIUM: DENSE_MEDIUM_CATEGORY_LABELS,
            LabelProfile.LONG: DENSE_LONG_CATEGORY_LABELS,
        }
    else:
        pools = {
            LabelProfile.SHORT: V1_SHORT_CATEGORY_LABELS,
            LabelProfile.MEDIUM: V1_MEDIUM_CATEGORY_LABELS,
            LabelProfile.LONG: V1_LONG_CATEGORY_LABELS,
        }
    return _sample_unique_labels(rng, pools[label_profile], count)


def sample_title(
    rng: random.Random,
    chart_type: ChartType,
    *,
    data_profile: DataProfile,
    variant: DatasetVariant = DatasetVariant.V1,
) -> str | None:
    if variant == DatasetVariant.DENSE_NOISY_V1:
        if rng.random() < 0.18:
            return None
        if chart_type == ChartType.LINE:
            pool = DENSE_LINE_TITLE_POOLS.get(data_profile, ["Metric Over Time"])
            return rng.choice(pool)
        return rng.choice(DENSE_BAR_TITLES)

    if rng.random() < 0.45:
        return None
    pool = V1_LINE_TITLES if chart_type == ChartType.LINE else V1_BAR_TITLES
    return rng.choice(pool)


def sample_x_axis_label(
    rng: random.Random,
    chart_type: ChartType,
    *,
    data_profile: DataProfile,
    variant: DatasetVariant = DatasetVariant.V1,
) -> str | None:
    if variant == DatasetVariant.DENSE_NOISY_V1:
        if chart_type == ChartType.LINE:
            pool = DENSE_LINE_X_AXIS_LABELS.get(data_profile, ["Index"])
            return rng.choice(pool)
        return rng.choice(DENSE_BAR_X_AXIS_LABELS)
    if chart_type == ChartType.LINE:
        return rng.choice(V1_LINE_X_AXIS_LABELS)
    return rng.choice(V1_BAR_X_AXIS_LABELS)


def sample_y_axis_label(
    rng: random.Random,
    *,
    chart_type: ChartType,
    data_profile: DataProfile,
    variant: DatasetVariant = DatasetVariant.V1,
) -> str:
    if variant == DatasetVariant.DENSE_NOISY_V1:
        if chart_type == ChartType.LINE:
            pool = DENSE_LINE_Y_AXIS_LABELS.get(data_profile, ["Metric"])
            return rng.choice(pool)
        return rng.choice(DENSE_BAR_Y_AXIS_LABELS)
    return rng.choice(V1_Y_AXIS_LABELS)


def sample_color(rng: random.Random) -> str:
    return rng.choice(COLORS)


def _sample_unique_labels(rng: random.Random, pool: Sequence[str], count: int) -> list[str]:
    if count <= len(pool):
        return rng.sample(list(pool), count)

    labels: list[str] = []
    shuffled_pool = list(pool)
    rng.shuffle(shuffled_pool)
    for index in range(count):
        base_label = shuffled_pool[index % len(shuffled_pool)]
        cycle = index // len(shuffled_pool)
        if cycle == 0:
            labels.append(base_label)
        else:
            labels.append(f"{base_label} {cycle + 1}")
    return labels


def _pattern_choice(pattern: Sequence[T], example_index: int) -> T:
    return pattern[example_index % len(pattern)]
