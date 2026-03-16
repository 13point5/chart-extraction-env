import json

from pydantic import ValidationError

from schemas import CanonicalChart, SchemaVersion, parse_chart_extraction
from .state import RubricState


def make_cache_parsed_answer(schema_version: SchemaVersion):
    async def cache_parsed_answer(completion, parser, state: RubricState) -> float:
        if "parsed_answer" in state:
            return 1.0 if isinstance(state["parsed_answer"], CanonicalChart) else 0.0

        answer_text = parser.parse_answer(completion)
        if answer_text is None:
            state["parsed_answer"] = None
            return 0.0

        try:
            versioned_answer = parse_chart_extraction(json.loads(answer_text), schema_version=schema_version)
            parsed_answer = versioned_answer.to_canonical()
        except (json.JSONDecodeError, ValidationError):
            parsed_answer = None

        state["parsed_answer"] = parsed_answer
        return 1.0 if isinstance(parsed_answer, CanonicalChart) else 0.0

    return cache_parsed_answer
