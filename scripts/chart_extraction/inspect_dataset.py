from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a chart extraction dataset.")
    parser.add_argument("--local-path", default="")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--preview-rows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_chart_dataset(local_path=args.local_path, repo_id=args.repo_id)
    for split, split_dataset in dataset.items():
        print(f"\n## {split} ({len(split_dataset)} rows)")
        info_rows = [json.loads(payload) for payload in split_dataset["info"]]
        for field in ["chart_type", "data_profile", "style_profile", "label_profile", "image_size_profile"]:
            counter = Counter(str(row[field]) for row in info_rows)
            print(f"{field}: {dict(counter)}")

        for row_index in range(min(args.preview_rows, len(split_dataset))):
            row = split_dataset[row_index]
            info = json.loads(row["info"])
            answer = json.loads(row["answer"])
            print(f"example[{row_index}] id={row['id']} info={info} answer={answer}")


def load_chart_dataset(*, local_path: str, repo_id: str) -> DatasetDict:
    if local_path:
        loaded = load_from_disk(local_path)
        if not isinstance(loaded, DatasetDict):
            raise ValueError("Expected a DatasetDict at the local path.")
        return loaded
    if repo_id:
        return load_dataset(repo_id)
    raise ValueError("Pass either --local-path or --repo-id.")


if __name__ == "__main__":
    main()
