import verifiers as vf

from dataset_transform import load_chart_extraction_dataset
from rubric import build_rubric
from schemas import SchemaVersion


def load_environment(schema_version: SchemaVersion = "v1") -> vf.Environment:
    """
    Loads the chart extraction environment.
    """

    train_dataset = load_chart_extraction_dataset(split="train", schema_version=schema_version)
    eval_dataset = load_chart_extraction_dataset(split="test", schema_version=schema_version)
    parser, rubric = build_rubric(schema_version)

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
    )
