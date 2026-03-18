import verifiers as vf

from prompts import SystemPromptVersion
from schemas import SchemaVersion
from .parsing import make_cache_parsed_answer
from .rewards.series_points import series_point_count_ratio
from .rewards.series_point_values import series_point_value
from .rewards.series_name_f1 import series_name_f1


def build_rubric(
    schema_version: SchemaVersion = "v1",
    system_prompt: SystemPromptVersion = "v1",
) -> tuple[vf.XMLParser, vf.Rubric]:
    if system_prompt == "v1":
        parser = vf.XMLParser(["answer"], answer_field="answer")
    elif system_prompt == "v2":
        parser = vf.XMLParser(["reasoning", "answer"], answer_field="answer")
    else:
        raise ValueError(f"Unsupported system_prompt: {system_prompt}")

    rubric = vf.Rubric(parser=parser)
    rubric.parallelize_scoring = False
    rubric.class_objects["logger"] = rubric.logger

    rubric.add_reward_func(make_cache_parsed_answer(schema_version), weight=0.0)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=1.0)
    rubric.add_reward_func(series_name_f1, weight=1.0)
    rubric.add_reward_func(series_point_count_ratio, weight=2.0)
    rubric.add_reward_func(series_point_value, weight=2.0)

    return parser, rubric
