from __future__ import annotations

import unittest

from models.hybrid_recommender import HybridRecommender
from models.ltr_model import FEATURE_COLUMNS, build_ranking_dataset


class HybridPipelineTests(unittest.TestCase):
    def test_feature_matrix_contains_required_columns(self) -> None:
        bundle = build_ranking_dataset(
            seed=42,
            cf_top_k=5,
            content_top_k=5,
            popularity_top_k=5,
            random_negatives=5,
        )
        train_frame = bundle.split_frame("train")
        required = set(FEATURE_COLUMNS + ["label", "user_id", "product_id", "split"])
        self.assertTrue(required.issubset(train_frame.columns))
        self.assertEqual(int(train_frame["label"].max()), 1)
        self.assertEqual(int(train_frame["label"].min()), 0)

    def test_hybrid_recommend_returns_unique_ranked_items(self) -> None:
        recommender = HybridRecommender(load_ranker=False)
        recommendations = recommender.hybrid_recommend(user_id=1276, top_k=5)

        self.assertLessEqual(len(recommendations), 5)
        self.assertGreater(len(recommendations), 0)
        product_ids = [row["product_id"] for row in recommendations]
        self.assertEqual(len(product_ids), len(set(product_ids)))
        self.assertEqual(recommendations[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()

