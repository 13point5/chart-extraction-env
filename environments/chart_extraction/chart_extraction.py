import verifiers as vf

from dataset_transform import load_chart_extraction_dataset
from rubric import parser, rubric


def load_environment(
    split: str = "test",
) -> vf.Environment:
    """
    Loads the chart extraction environment.
    """

    dataset = load_chart_extraction_dataset(split=split)

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
    )
