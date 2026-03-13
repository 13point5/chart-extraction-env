import verifiers as vf

from dataset_transform import USER_PROMPT_TEXT, load_chart_extraction_dataset
from prompts import DEFAULT_SYSTEM_PROMPT_V1
from rubric import rubric


def load_environment(max_examples: int = -1, **kwargs) -> vf.Environment:
    """
    Loads the chart extraction environment.
    """

    instruction_text = f"{DEFAULT_SYSTEM_PROMPT_V1}\n\n{USER_PROMPT_TEXT}"
    dataset = load_chart_extraction_dataset(
        instruction_text=instruction_text,
        max_examples=max_examples,
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
    )
