import json

from pydantic import ValidationError
import verifiers as vf

from schema import ChartExtraction_V1
from .state import RubricState


parser = vf.XMLParser(["answer"])


async def cache_parsed_answer(completion, parser, state: RubricState) -> float:
    if "parsed_answer" in state:
        return 1.0 if isinstance(state["parsed_answer"], ChartExtraction_V1) else 0.0

    answer_text = parser.parse_answer(completion)
    if answer_text is None:
        state["parsed_answer"] = None
        return 0.0

    try:
        parsed_answer = ChartExtraction_V1.model_validate(json.loads(answer_text))
    except (json.JSONDecodeError, ValidationError):
        parsed_answer = None

    state["parsed_answer"] = parsed_answer
    return 1.0 if isinstance(parsed_answer, ChartExtraction_V1) else 0.0
