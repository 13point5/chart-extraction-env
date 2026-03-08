from __future__ import annotations

import argparse
from pathlib import Path

from chart_extraction_env.dataset import build_dataset_dict, build_split_examples, save_dataset_dict
from chart_extraction_env.dataset.hf import push_dataset_dict, save_preview_bundle
from chart_extraction_env.dataset.models import DatasetBuildConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic chart extraction dataset.")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chart_extraction/dataset_v1"),
    )
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--repo-id", type=str, default="")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetBuildConfig(
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_examples = {
        "train": build_split_examples(
            split="train",
            num_examples=config.train_size,
            seed=config.seed,
        ),
        "validation": build_split_examples(
            split="validation",
            num_examples=config.validation_size,
            seed=config.seed,
        ),
        "test": build_split_examples(
            split="test",
            num_examples=config.test_size,
            seed=config.seed,
        ),
    }
    dataset_dict = build_dataset_dict(split_examples)
    save_dataset_dict(dataset_dict, args.output_dir)
    save_preview_bundle(
        output_dir=args.output_dir,
        split_examples=split_examples,
        preview_count=args.preview_count,
    )

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is set")
        push_dataset_dict(dataset_dict, repo_id=args.repo_id, private=args.private)

    print(f"Saved dataset to {args.output_dir}")
    if args.push:
        print(f"Pushed dataset to {args.repo_id}")


if __name__ == "__main__":
    main()
