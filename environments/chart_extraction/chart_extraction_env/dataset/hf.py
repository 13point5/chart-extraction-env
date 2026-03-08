from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset, load_from_disk
from huggingface_hub import HfApi

from .generator import build_split_examples, canonical_answer_json, canonical_info_json
from .models import GeneratedExample

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
) -> Dataset:
    if local_path:
        loaded = load_from_disk(local_path)
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded

    if repo_id:
        return load_dataset(repo_id, split=split)

    examples = build_split_examples(split=split, num_examples=default_examples, seed=seed)
    return Dataset.from_list([example_to_row(example) for example in examples], features=DATASET_FEATURES)


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
