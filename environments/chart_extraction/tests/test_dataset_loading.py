from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chart_extraction_env.dataset.generator import build_split_examples
from chart_extraction_env.dataset.hf import build_dataset_dict, load_chart_dataset, save_dataset_dict
from chart_extraction_env.dataset.models import DatasetVariant


class DatasetLoadingTests(unittest.TestCase):
    def test_chart_type_filter_returns_line_only_rows(self) -> None:
        split_examples = {
            "validation": build_split_examples(
                split="validation",
                num_examples=20,
                seed=19,
                variant=DatasetVariant.DENSE_NOISY_V1,
                version="dense_noisy_test",
            )
        }
        dataset_dict = build_dataset_dict(split_examples)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_dataset_dict(dataset_dict, output_dir)
            filtered = load_chart_dataset(
                split="validation",
                local_path=str(output_dir),
                chart_type_filter="line",
            )

        self.assertGreater(len(filtered), 0)
        for row in filtered:
            info = json.loads(row["info"])
            self.assertEqual(info["chart_type"], "line")


if __name__ == "__main__":
    unittest.main()
