import verifiers as vf

from dataset_transform import load_chart_extraction_dataset
from prompts import SystemPromptVersion
from rubric import build_rubric
from rubric.rewards.series_point_values import DEFAULT_OKS_K, DEFAULT_OKS_THRESHOLD
from schemas import SchemaVersion


def load_environment(
    schema_version: SchemaVersion = "v2",
    system_prompt: SystemPromptVersion = "v1",
    max_examples: int | None = None,
    disabled_rewards: list[str] | None = None,
    series_point_value_oks_k: float = DEFAULT_OKS_K,
    series_point_value_oks_threshold: float = DEFAULT_OKS_THRESHOLD,
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
        series_point_value_oks_k=series_point_value_oks_k,
        series_point_value_oks_threshold=series_point_value_oks_threshold,
        disabled_rewards=disabled_rewards,
    )

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        map_kwargs={"load_from_cache_file": False},
    )
