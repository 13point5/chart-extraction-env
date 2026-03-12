from __future__ import annotations

import unittest
import random

from chart_extraction_env.dataset.generator import build_recipe, build_split_examples
from chart_extraction_env.dataset.models import ChartType, DatasetVariant, LabelProfile
from chart_extraction_env.dataset.render import render_chart
from chart_extraction_env.dataset.sampling import sample_category_labels


class DatasetGenerationTests(unittest.TestCase):
    def test_dense_variant_reaches_large_line_point_counts(self) -> None:
        recipes = [
            build_recipe(
                seed=100 + example_index,
                example_index=example_index,
                variant=DatasetVariant.DENSE_NOISY_V1,
            )
            for example_index in range(48)
        ]

        line_point_counts = [
            recipe.spec.num_points
            for recipe in recipes
            if recipe.spec.chart_type == ChartType.LINE
        ]

        self.assertTrue(line_point_counts)
        self.assertGreaterEqual(max(line_point_counts), 300)
        self.assertLessEqual(max(line_point_counts), 500)

    def test_dense_variant_uses_unique_bar_labels_for_large_counts(self) -> None:
        labels = sample_category_labels(
            rng=random.Random(7),
            label_profile=LabelProfile.MEDIUM,
            count=40,
            variant=DatasetVariant.DENSE_NOISY_V1,
        )

        self.assertEqual(len(labels), 40)
        self.assertEqual(len(set(labels)), 40)

    def test_dense_variant_examples_record_version_and_variant(self) -> None:
        examples = build_split_examples(
            split="validation",
            num_examples=3,
            seed=17,
            variant=DatasetVariant.DENSE_NOISY_V1,
            version="dense-noisy-local-v1",
        )

        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertEqual(example.info.variant, DatasetVariant.DENSE_NOISY_V1)
            self.assertEqual(example.info.version, "dense-noisy-local-v1")

    def test_dense_variant_large_line_chart_renders(self) -> None:
        recipes = [
            build_recipe(
                seed=500 + example_index,
                example_index=example_index,
                variant=DatasetVariant.DENSE_NOISY_V1,
            )
            for example_index in range(64)
        ]
        recipe = next(
            recipe
            for recipe in recipes
            if recipe.spec.chart_type == ChartType.LINE and recipe.spec.num_points >= 300
        )

        image_bytes = render_chart(recipe)

        self.assertGreater(len(image_bytes), 1_000)


if __name__ == "__main__":
    unittest.main()
