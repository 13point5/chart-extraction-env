from typing import TypedDict

from schemas import CanonicalChart


class RubricState(TypedDict, total=False):
    parsed_answer: CanonicalChart | None
    series_point_count_ratio_raw: float
    series_point_value_raw: float
