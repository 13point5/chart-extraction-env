from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset, load_from_disk
from huggingface_hub import HfApi

from .generator import build_split_examples, canonical_answer_json, canonical_info_json
from .models import DatasetVariant, GeneratedExample

DATASET_FEATURES = Features(
    {
        "id": Value("string"),
        "image": Image(),
        "answer": Value("string"),
        "info": Value("string"),
    }
)


def build_dataset_dict(split_examples: dict[str, list[GeneratedExample]]) -> DatasetDict:
    datasets = {}
    for split, examples in split_examples.items():
        rows = [example_to_row(example) for example in examples]
        datasets[split] = Dataset.from_list(rows, features=DATASET_FEATURES)
    return DatasetDict(datasets)


def example_to_row(example: GeneratedExample) -> dict[str, object]:
    return {
        "id": example.example_id,
        "image": {"bytes": example.image_bytes, "path": None},
        "answer": canonical_answer_json(example.answer),
        "info": canonical_info_json(example.info),
    }


def save_dataset_dict(dataset_dict: DatasetDict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))


def load_chart_dataset(
    *,
    split: str,
    local_path: str | None = None,
    repo_id: str | None = None,
    seed: int = 7,
    default_examples: int = 64,
    variant: DatasetVariant = DatasetVariant.V1,
    chart_type_filter: str | None = None,
) -> Dataset:
    if local_path:
        loaded = load_from_disk(local_path)
        if isinstance(loaded, DatasetDict):
            dataset = loaded[split]
        else:
            dataset = loaded
        return filter_chart_dataset(dataset, chart_type_filter)

    if repo_id:
        dataset = load_dataset(repo_id, split=split)
        return filter_chart_dataset(dataset, chart_type_filter)

    examples = build_split_examples(
        split=split,
        num_examples=default_examples,
        seed=seed,
        variant=variant,
        version=variant.value,
    )
    dataset = Dataset.from_list([example_to_row(example) for example in examples], features=DATASET_FEATURES)
    return filter_chart_dataset(dataset, chart_type_filter)


def filter_chart_dataset(dataset: Dataset, chart_type_filter: str | None) -> Dataset:
    if not chart_type_filter:
        return dataset
    allowed_chart_types = {
        item.strip()
        for item in chart_type_filter.split(",")
        if item.strip()
    }
    if not allowed_chart_types:
        return dataset
    return dataset.filter(
        lambda example: json.loads(example["info"])["chart_type"] in allowed_chart_types,
        desc=f"Filter chart_type in {sorted(allowed_chart_types)}",
    )


def push_dataset_dict(
    dataset_dict: DatasetDict,
    *,
    repo_id: str,
    private: bool,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset_dict.push_to_hub(repo_id, private=private)


def save_preview_bundle(
    *,
    output_dir: Path,
    split_examples: dict[str, list[GeneratedExample]],
    preview_count: int,
) -> None:
    preview_dir = output_dir / "preview_images"
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# Dataset Preview",
        "",
        "This preview bundle is generated from the first few examples of each split.",
        "",
    ]
    for split, examples in split_examples.items():
        manifest_lines.extend([f"## {split}", ""])
        for example in examples[:preview_count]:
            image_name = f"{example.example_id}.png"
            image_path = preview_dir / image_name
            image_path.write_bytes(example.image_bytes)
            manifest_lines.append(f"### {example.example_id}")
            manifest_lines.append("")
            manifest_lines.append(f"![{example.example_id}](preview_images/{image_name})")
            manifest_lines.append("")
            manifest_lines.append("```json")
            manifest_lines.append(json.dumps(example.info.model_dump(mode='json'), indent=2))
            manifest_lines.append("```")
            manifest_lines.append("")
    (output_dir / "preview.md").write_text("\n".join(manifest_lines), encoding="utf-8")
