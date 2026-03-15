import verifiers as vf

from dataset_transform import load_chart_extraction_dataset
from rubric import parser, rubric


def load_environment() -> vf.Environment:
    """
    Loads the chart extraction environment.
    """

    train_dataset = load_chart_extraction_dataset(split="train")
    eval_dataset = load_chart_extraction_dataset(split="test")

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
    )
