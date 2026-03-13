from typing import TypedDict

from schema import ChartExtraction_V1


class RubricState(TypedDict, total=False):
    parsed_answer: ChartExtraction_V1 | None
