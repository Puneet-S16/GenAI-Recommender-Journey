from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
from xgboost import XGBRanker

from models.recommender_utils import (
    EXPERIMENT_RESULTS_PATH,
    LTR_METADATA_PATH,
    LTR_MODEL_PATH,
    Catalog,
    CollaborativeFilterScorer,
    ContentScorer,
    ItemStatistics,
    PreferenceProfileModel,
    build_candidate_feature_frame,
    build_interactions,
    dedupe_preserve_order,
    load_behavior,
    normalize_signal_weights,
)


DEFAULT_FEATURE_COLUMNS = [
    "cf_score",
    "embedding_similarity",
    "price_delta_from_user_avg",
    "price_sensitivity_match",
    "category_match_score",
    "material_preference_score",
    "style_affinity_score",
    "historical_ctr",
    "purchase_history_count",
    "interaction_history_count",
    "hybrid_signal_score",
]


class HybridRecommender:
    def __init__(self, *, load_ranker: bool = True) -> None:
        self.catalog = Catalog.from_local_artifacts()
        self.behavior = load_behavior()
        self.interactions = build_interactions(self.behavior)
        self.cf_model = CollaborativeFilterScorer().fit(self.interactions)
        self.profile_model = PreferenceProfileModel().fit(self.interactions, self.catalog.products)
        self.item_stats = ItemStatistics().fit(self.interactions)
        self.content_model = ContentScorer(self.catalog)

        self.feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        self.signal_weights = normalize_signal_weights()
        self.similarity_threshold = 0.0
        self.ranker: XGBRanker | None = None

        if LTR_METADATA_PATH.exists():
            metadata = json.loads(LTR_METADATA_PATH.read_text(encoding="utf-8"))
            self.feature_columns = list(metadata.get("feature_columns", self.feature_columns))
            self.signal_weights = normalize_signal_weights(metadata.get("signal_weights"))
            self.similarity_threshold = float(metadata.get("similarity_threshold", 0.0))

        if load_ranker and LTR_MODEL_PATH.exists():
            self.ranker = XGBRanker()
            self.ranker.load_model(str(LTR_MODEL_PATH))

    def _candidate_pool(self, user_id: int, *, query: str | None = None, pool_size: int = 50) -> list[int]:
        purchased_items = set(self.profile_model.purchased_items.get(int(user_id), set()))

        cf_candidates = self.cf_model.recommend(
            int(user_id),
            top_n=max(pool_size // 2, 10),
            exclude_product_ids=purchased_items,
        )
        content_candidates = self.content_model.recommend_from_history(
            self.profile_model.get_seed_items(int(user_id)),
            top_n=max(pool_size // 2, 10),
            exclude_product_ids=purchased_items,
            min_similarity=0.0,
        )
        query_candidates = (
            self.content_model.recommend_from_query(
                query,
                top_n=max(pool_size // 2, 10),
                exclude_product_ids=purchased_items,
                min_similarity=0.0,
            )
            if query
            else []
        )
        popularity_candidates = self.item_stats.recommend(
            top_n=max(pool_size // 4, 10),
            exclude_product_ids=purchased_items,
        )
        return dedupe_preserve_order(cf_candidates + content_candidates + query_candidates + popularity_candidates)[:pool_size]

    def _score_candidates(self, user_id: int, *, query: str | None = None, pool_size: int = 50) -> pd.DataFrame:
        candidate_product_ids = self._candidate_pool(user_id, query=query, pool_size=pool_size)
        if not candidate_product_ids:
            return pd.DataFrame()

        query_scores = None
        if query:
            query_scores = self.content_model.score_series_from_query(query, candidate_product_ids)

        feature_frame = build_candidate_feature_frame(
            user_id=int(user_id),
            candidate_product_ids=candidate_product_ids,
            catalog=self.catalog,
            cf_model=self.cf_model,
            content_model=self.content_model,
            profile_model=self.profile_model,
            item_stats=self.item_stats,
            signal_weights=self.signal_weights,
            similarity_threshold=self.similarity_threshold,
            query_scores=query_scores,
        )
        if self.ranker is not None:
            feature_frame["ltr_score"] = self.ranker.predict(feature_frame[self.feature_columns].astype(float))
            return feature_frame.sort_values(
                ["ltr_score", "hybrid_signal_score", "historical_ctr"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

        feature_frame["ltr_score"] = feature_frame["hybrid_signal_score"]
        return feature_frame.sort_values(
            ["hybrid_signal_score", "historical_ctr", "cf_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def hybrid_recommend(
        self,
        user_id: int,
        query: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ranked = self._score_candidates(int(user_id), query=query, pool_size=max(top_k * 10, 40))
        if ranked.empty:
            return []

        output = ranked.head(top_k).copy()
        output.insert(0, "rank", range(1, len(output) + 1))
        columns = [
            "rank",
            "product_id",
            "title",
            "category",
            "material",
            "style",
            "price",
            "cf_score",
            "embedding_similarity",
            "price_sensitivity_match",
            "historical_ctr",
            "hybrid_signal_score",
            "ltr_score",
        ]
        return output[columns].to_dict(orient="records")


_RECOMMENDER: HybridRecommender | None = None


def get_recommender(*, load_ranker: bool = True) -> HybridRecommender:
    global _RECOMMENDER
    if _RECOMMENDER is None:
        _RECOMMENDER = HybridRecommender(load_ranker=load_ranker)
    return _RECOMMENDER


def hybrid_recommend(user_id: int, query: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    return get_recommender(load_ranker=True).hybrid_recommend(user_id=user_id, query=query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hybrid recommendations for a user.")
    parser.add_argument("--user-id", type=int, required=True, help="User id to score.")
    parser.add_argument("--query", help="Optional free-text query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommendations = hybrid_recommend(args.user_id, query=args.query, top_k=args.top_k)
    print(json.dumps(recommendations, indent=2))

    if EXPERIMENT_RESULTS_PATH.exists():
        results = json.loads(EXPERIMENT_RESULTS_PATH.read_text(encoding="utf-8"))
        metrics = results.get("evaluation", {}).get("ltr_hybrid", {})
        if metrics:
            print(
                "\nModel summary | "
                f"Precision@5={metrics.get('precision@5', 0):.4f} | "
                f"NDCG@5={metrics.get('ndcg@5', 0):.4f}"
            )


if __name__ == "__main__":
    main()

