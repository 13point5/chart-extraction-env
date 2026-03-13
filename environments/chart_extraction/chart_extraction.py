import verifiers as vf

from dataset_transform import USER_PROMPT_TEXT, load_chart_extraction_dataset
from prompts import DEFAULT_SYSTEM_PROMPT_V1
from rubric import parser, rubric


def load_environment(
    split: str = "test",
    num_examples: int = -1,
    **kwargs,
) -> vf.Environment:
    """
    Loads the chart extraction environment.
    """

    dataset = load_chart_extraction_dataset(
        instruction_text=USER_PROMPT_TEXT,
        split=split,
        num_examples=num_examples,
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=DEFAULT_SYSTEM_PROMPT_V1,
        parser=parser,
        rubric=rubric,
    )
