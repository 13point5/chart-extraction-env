from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"


class XType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class DataProfile(str, Enum):
    MONO_UP = "mono_up"
    MONO_DOWN = "mono_down"
    SINGLE_PEAK = "single_peak"
    SINGLE_VALLEY = "single_valley"
    ZIGZAG = "zigzag"
    FLAT = "flat"
    RANDOM_WALK = "random_walk"


class XMode(str, Enum):
    NUMERIC_EVEN = "numeric_even"
    CATEGORICAL_ORDERED = "categorical_ordered"


class LabelProfile(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class StyleProfile(str, Enum):
    CLEAN = "clean"
    GRID = "grid"
    MARKERS = "markers"
    GRID_MARKERS = "grid_markers"


class AnnotationProfile(str, Enum):
    NONE = "none"
    ENDPOINT_LABELS = "endpoint_labels"


class LayoutProfile(str, Enum):
    SQUARE = "square"
    WIDE = "wide"
    TALL = "tall"


class ImageSizeProfile(str, Enum):
    SMALL_SQUARE = "small_square"
    MEDIUM_SQUARE = "medium_square"
    MEDIUM_WIDE = "medium_wide"
    LARGE_WIDE = "large_wide"
    MEDIUM_TALL = "medium_tall"


class YScaleProfile(str, Enum):
    NEAR_FLAT = "near_flat"
    SMALL_RANGE = "small_range"
    MEDIUM_RANGE = "medium_range"
    LARGE_RANGE = "large_range"


class ChartSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    chart_type: ChartType
    data_profile: DataProfile
    x_mode: XMode
    label_profile: LabelProfile
    style_profile: StyleProfile
    annotation_profile: AnnotationProfile
    layout_profile: LayoutProfile
    image_size_profile: ImageSizeProfile
    y_scale_profile: YScaleProfile
    num_points: int = Field(ge=4, le=40)
    seed: int


class SeriesPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float | str
    y: float


class SeriesData(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    points: list[SeriesPoint]


class CanonicalAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    chart_type: ChartType
    x_type: XType
    series: list[SeriesData]


class RenderPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    figure_size_inches: tuple[float, float]
    dpi: int
    show_grid: bool
    show_markers: bool
    rotate_x_labels: bool
    color_hex: str
    line_width: float = 2.2
    marker_size: float = 6.0
    title: str | None = None
    x_label: str | None = None
    y_label: str


class ChartRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: ChartSpec
    render_plan: RenderPlan
    series_name: str
    answer: CanonicalAnswer


class ExampleInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = "v1"
    split: str
    chart_type: ChartType
    data_profile: DataProfile
    x_mode: XMode
    label_profile: LabelProfile
    style_profile: StyleProfile
    annotation_profile: AnnotationProfile
    layout_profile: LayoutProfile
    image_size_profile: ImageSizeProfile
    y_scale_profile: YScaleProfile
    num_points: int
    figure_size_inches: tuple[float, float]
    dpi: int
    seed: int


class GeneratedExample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    example_id: str
    image_bytes: bytes
    answer: CanonicalAnswer
    info: ExampleInfo


class DatasetBuildConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_size: int = Field(default=256, ge=1)
    validation_size: int = Field(default=64, ge=1)
    test_size: int = Field(default=64, ge=1)
    seed: int = 7
    version: str = "v1"
