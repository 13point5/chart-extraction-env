from typing import TypedDict

from schemas import CanonicalChart


class RubricState(TypedDict, total=False):
    parsed_answer: CanonicalChart | None
