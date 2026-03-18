import verifiers as vf

from dataset_transform import load_chart_extraction_dataset
from prompts import SystemPromptVersion
from rubric import build_rubric
from schemas import SchemaVersion


def load_environment(
    schema_version: SchemaVersion = "v1",
    system_prompt: SystemPromptVersion = "v1",
    max_examples: int | None = None,
) -> vf.Environment:
    """
    Loads the chart extraction environment.
    """

    train_dataset = load_chart_extraction_dataset(
        split="train",
        schema_version=schema_version,
        system_prompt=system_prompt,
        max_examples=max_examples,
    )
    eval_dataset = load_chart_extraction_dataset(
        split="test",
        schema_version=schema_version,
        system_prompt=system_prompt,
        max_examples=max_examples,
    )
    parser, rubric = build_rubric(
        schema_version=schema_version,
        system_prompt=system_prompt,
    )

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        map_kwargs={"load_from_cache_file": False},
    )
