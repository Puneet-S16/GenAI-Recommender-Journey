from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    SentenceTransformer = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "models" / "model_checkpoints"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EVALUATION_DIR = ROOT_DIR / "evaluation"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

PRODUCTS_PATH = DATA_DIR / "products_clean.parquet"
BEHAVIOR_PATH = DATA_DIR / "user_behavior_clean.parquet"
USER_FEATURES_PATH = DATA_DIR / "user_features.parquet"
EMBEDDINGS_PATH = CHECKPOINT_DIR / "product_embeddings.npy"
EMBEDDING_METADATA_PATH = CHECKPOINT_DIR / "embedding_metadata.json"
LTR_MODEL_PATH = CHECKPOINT_DIR / "ltr_ranker.json"
LTR_METADATA_PATH = CHECKPOINT_DIR / "ltr_metadata.json"
OPTUNA_DB_PATH = EXPERIMENTS_DIR / "optuna_study.db"
EXPERIMENT_RESULTS_PATH = EXPERIMENTS_DIR / "experiment_results.json"
EVALUATION_REPORT_PATH = EVALUATION_DIR / "hybrid_evaluation_report.pdf"
NOTEBOOK_PATH = NOTEBOOKS_DIR / "hybrid_analysis.ipynb"

EVENT_WEIGHTS = {
    "view": 3.0,
    "wishlist": 5.0,
    "cart": 7.0,
    "purchase": 20.0,
}
DEFAULT_SIGNAL_WEIGHTS = {
    "cf": 0.35,
    "embedding": 0.30,
    "price": 0.10,
    "style": 0.10,
    "ctr": 0.15,
}


def ensure_directories() -> None:
    for path in (DATA_DIR, CHECKPOINT_DIR, EXPERIMENTS_DIR, EVALUATION_DIR, NOTEBOOKS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        cast_value = int(value)
        if cast_value in seen:
            continue
        seen.add(cast_value)
        ordered.append(cast_value)
    return ordered


def normalize_signal_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    raw = dict(DEFAULT_SIGNAL_WEIGHTS)
    if weights:
        raw.update({key: float(value) for key, value in weights.items()})

    total = sum(max(value, 0.0) for value in raw.values())
    if total <= 0.0:
        return dict(DEFAULT_SIGNAL_WEIGHTS)
    return {key: max(value, 0.0) / total for key, value in raw.items()}


def _clean_text(value: object, default: str = "Unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default

    text = " ".join(str(value).replace("_", " ").split())
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def _normalize_label(value: object) -> str:
    return _clean_text(value).title()


def _build_description(row: pd.Series) -> str:
    price = float(row["price"])
    price_text = str(int(price)) if price.is_integer() else f"{price:.2f}"
    return (
        f"{row['style']} {row['material']} {row['category']} in "
        f"{row['color']} color. Price: {price_text}."
    )


def load_products() -> pd.DataFrame:
    products = pd.read_parquet(PRODUCTS_PATH).copy()
    products["product_id"] = products["product_id"].astype(int)
    products.rename(
        columns={
            "name": "title",
            "description": "catalog_description",
        },
        inplace=True,
    )

    for column in ("title", "catalog_description"):
        products[column] = products[column].map(_clean_text)
    for column in ("category", "material", "color", "style"):
        products[column] = products[column].map(_normalize_label)

    products["price"] = products["price"].astype(float)
    products["rating"] = products["rating"].astype(float)
    products["description"] = products.apply(_build_description, axis=1)
    return products[
        [
            "product_id",
            "title",
            "category",
            "material",
            "color",
            "style",
            "price",
            "rating",
            "description",
            "catalog_description",
        ]
    ]


def load_behavior() -> pd.DataFrame:
    behavior = pd.read_parquet(BEHAVIOR_PATH).copy()
    behavior["user_id"] = behavior["user_id"].astype(int)
    behavior["product_id"] = behavior["product_id"].astype(int)
    behavior["timestamp"] = pd.to_datetime(behavior["timestamp"], errors="coerce")
    behavior["event_type"] = behavior["event_type"].map(lambda value: _clean_text(value, default="view").lower())
    return behavior


def load_user_features() -> pd.DataFrame:
    features = pd.read_parquet(USER_FEATURES_PATH).copy()
    features["user_id"] = features["user_id"].astype(int)
    return features


def build_interactions(behavior: pd.DataFrame) -> pd.DataFrame:
    interactions = behavior.copy()
    interactions["score"] = interactions["event_type"].map(EVENT_WEIGHTS).fillna(0.0).astype(float)
    return interactions


def load_embedding_metadata() -> dict[str, object]:
    if not EMBEDDING_METADATA_PATH.exists():
        return {}
    return json.loads(EMBEDDING_METADATA_PATH.read_text(encoding="utf-8"))


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


@lru_cache(maxsize=2)
def get_text_encoder(model_name: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed; query-time embedding generation is unavailable."
        )
    return SentenceTransformer(model_name)


def _resolve_embedding_model_name(metadata: dict[str, object] | None = None) -> str:
    payload = metadata or load_embedding_metadata()
    embedding_config = payload.get("embedding_config")
    if isinstance(embedding_config, dict):
        resolved = embedding_config.get("resolved_model_name")
        if isinstance(resolved, str) and resolved:
            return resolved
    resolved = payload.get("resolved_model_name")
    if isinstance(resolved, str) and resolved:
        return resolved
    return "all-MiniLM-L6-v2"


def encode_query_text(query_text: str, model_name: str | None = None) -> np.ndarray:
    metadata = load_embedding_metadata()
    resolved_model = model_name or _resolve_embedding_model_name(metadata)
    encoder = get_text_encoder(resolved_model)
    vector = encoder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vector, dtype="float32")


def load_embeddings(products: pd.DataFrame | None = None) -> np.ndarray:
    if EMBEDDINGS_PATH.exists():
        return _normalize_embeddings(np.load(EMBEDDINGS_PATH))

    products_frame = load_products() if products is None else products
    model_name = _resolve_embedding_model_name()
    encoder = get_text_encoder(model_name)
    vectors = encoder.encode(
        products_frame["description"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = _normalize_embeddings(np.asarray(vectors, dtype="float32"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings


def load_catalog_products() -> pd.DataFrame:
    parquet_products = load_products()
    metadata = load_embedding_metadata()
    metadata_products = metadata.get("products")
    if not isinstance(metadata_products, list):
        return parquet_products.sort_values("product_id").reset_index(drop=True)

    metadata_frame = pd.DataFrame(metadata_products).copy()
    if metadata_frame.empty:
        return parquet_products.sort_values("product_id").reset_index(drop=True)

    metadata_frame["product_id"] = metadata_frame["product_id"].astype(int)
    for column in ("title", "category", "material", "color", "style", "description"):
        if column in metadata_frame.columns:
            metadata_frame[column] = metadata_frame[column].map(_clean_text)
    for column in ("category", "material", "color", "style"):
        if column in metadata_frame.columns:
            metadata_frame[column] = metadata_frame[column].map(_normalize_label)

    merged = metadata_frame.merge(
        parquet_products[["product_id", "catalog_description", "title"]],
        on="product_id",
        how="left",
        suffixes=("", "_parquet"),
    )
    merged["title"] = merged["title"].where(merged["title"].notna(), merged["title_parquet"])
    merged.drop(columns=["title_parquet"], inplace=True)
    if "catalog_description" not in merged.columns:
        merged["catalog_description"] = ""
    merged["catalog_description"] = merged["catalog_description"].map(_clean_text)
    merged["description"] = merged["description"].where(
        merged["description"].notna(),
        merged.apply(_build_description, axis=1),
    )
    merged["price"] = merged["price"].astype(float)
    merged["rating"] = merged["rating"].astype(float)
    return merged[
        [
            "product_id",
            "title",
            "category",
            "material",
            "color",
            "style",
            "price",
            "rating",
            "description",
            "catalog_description",
        ]
    ].reset_index(drop=True)


@dataclass(slots=True)
class Catalog:
    products: pd.DataFrame
    embeddings: np.ndarray
    product_id_to_index: dict[int, int]
    embedding_model_name: str

    @classmethod
    def from_local_artifacts(cls) -> "Catalog":
        ensure_directories()
        metadata = load_embedding_metadata()
        products = load_catalog_products()
        embeddings = load_embeddings(products)
        if len(products) != embeddings.shape[0]:
            raise ValueError(
                "Embedding count does not match product count. "
                f"Found {embeddings.shape[0]} embeddings for {len(products)} products."
            )
        product_id_to_index = {
            int(product_id): index
            for index, product_id in enumerate(products["product_id"].tolist())
        }
        return cls(
            products=products.reset_index(drop=True),
            embeddings=_normalize_embeddings(embeddings),
            product_id_to_index=product_id_to_index,
            embedding_model_name=_resolve_embedding_model_name(metadata),
        )

    @property
    def product_ids(self) -> list[int]:
        return self.products["product_id"].astype(int).tolist()

    def lookup(self, product_ids: Iterable[int]) -> pd.DataFrame:
        requested = pd.Index([int(product_id) for product_id in product_ids], dtype="int64", name="product_id")
        frame = self.products.set_index("product_id").reindex(requested)
        return frame.reset_index()


class CollaborativeFilterScorer:
    def __init__(self) -> None:
        self.user_item_matrix: pd.DataFrame | None = None
        self.predicted_matrix: pd.DataFrame | None = None
        self.popularity_scores: pd.Series | None = None
        self.global_min = 0.0
        self.global_max = 1.0

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilterScorer":
        matrix = interactions.pivot_table(
            index="user_id",
            columns="product_id",
            values="score",
            aggfunc="sum",
            fill_value=0.0,
        )
        self.user_item_matrix = matrix

        user_vectors = matrix.to_numpy(dtype=float)
        similarity = cosine_similarity(user_vectors)
        np.fill_diagonal(similarity, 0.0)
        denominator = np.abs(similarity).sum(axis=1, keepdims=True)
        predicted = similarity @ user_vectors / np.clip(denominator, 1e-8, None)
        predicted = np.where(user_vectors > 0.0, -np.inf, predicted)
        self.predicted_matrix = pd.DataFrame(
            predicted,
            index=matrix.index,
            columns=matrix.columns,
        )

        popularity = interactions.groupby("product_id")["score"].sum().sort_values(ascending=False)
        self.popularity_scores = popularity
        finite_values = predicted[np.isfinite(predicted)]
        if finite_values.size:
            self.global_min = float(finite_values.min())
            self.global_max = float(finite_values.max())
        return self

    def _scale(self, values: np.ndarray) -> np.ndarray:
        if self.global_max <= self.global_min:
            return np.zeros_like(values, dtype=float)
        return (values - self.global_min) / (self.global_max - self.global_min)

    def _popularity_series(self, candidate_ids: list[int]) -> pd.Series:
        if self.popularity_scores is None:
            return pd.Series(0.0, index=candidate_ids, dtype=float)
        popularity = self.popularity_scores.reindex(candidate_ids).fillna(0.0)
        if popularity.max() <= popularity.min():
            return pd.Series(0.0, index=candidate_ids, dtype=float)
        scaled = (popularity - popularity.min()) / (popularity.max() - popularity.min())
        return scaled.astype(float)

    def score_series(self, user_id: int, candidate_ids: Iterable[int]) -> pd.Series:
        candidate_list = [int(product_id) for product_id in candidate_ids]
        popularity_fallback = self._popularity_series(candidate_list)

        if self.predicted_matrix is None or user_id not in self.predicted_matrix.index:
            popularity_fallback.name = "cf_score"
            return popularity_fallback

        predictions = self.predicted_matrix.loc[user_id].reindex(candidate_list)
        values = predictions.to_numpy(dtype=float)
        output = popularity_fallback.to_numpy(dtype=float, copy=True)
        finite_mask = np.isfinite(values)
        if finite_mask.any():
            output[finite_mask] = self._scale(values[finite_mask])
        series = pd.Series(output, index=candidate_list, dtype=float)
        series.name = "cf_score"
        return series

    def recommend(
        self,
        user_id: int,
        *,
        top_n: int = 20,
        exclude_product_ids: Iterable[int] | None = None,
    ) -> list[int]:
        excluded = {int(product_id) for product_id in (exclude_product_ids or [])}

        if self.predicted_matrix is not None and user_id in self.predicted_matrix.index:
            predictions = self.predicted_matrix.loc[user_id].replace([-np.inf, np.inf], np.nan).dropna()
            predictions = predictions.sort_values(ascending=False)
            return [
                int(product_id)
                for product_id in predictions.index.tolist()
                if int(product_id) not in excluded
            ][:top_n]

        if self.popularity_scores is None:
            return []
        return [
            int(product_id)
            for product_id in self.popularity_scores.index.tolist()
            if int(product_id) not in excluded
        ][:top_n]


class PreferenceProfileModel:
    def __init__(self) -> None:
        self.global_avg_price = 0.0
        self.avg_price_by_user: dict[int, float] = {}
        self.price_std_by_user: dict[int, float] = {}
        self.preference_tables: dict[str, pd.DataFrame] = {}
        self.purchase_history: dict[int, list[int]] = {}
        self.high_intent_history: dict[int, list[int]] = {}
        self.interaction_history: dict[int, list[int]] = {}
        self.purchased_items: dict[int, set[int]] = {}
        self.purchase_counts: dict[int, int] = {}
        self.interaction_counts: dict[int, int] = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> "PreferenceProfileModel":
        joined = interactions.merge(
            products[["product_id", "category", "material", "style", "price"]],
            on="product_id",
            how="left",
        ).copy()
        joined["timestamp"] = pd.to_datetime(joined["timestamp"], errors="coerce")
        joined["score"] = joined["score"].astype(float)

        self.global_avg_price = float(products["price"].mean())
        for user_id, group in joined.groupby("user_id"):
            prices = group["price"].to_numpy(dtype=float)
            weights = group["score"].to_numpy(dtype=float)
            if prices.size == 0 or np.allclose(weights.sum(), 0.0):
                self.avg_price_by_user[int(user_id)] = self.global_avg_price
                self.price_std_by_user[int(user_id)] = max(self.global_avg_price * 0.15, 1.0)
                continue

            weighted_avg = float(np.average(prices, weights=weights))
            variance = float(np.average((prices - weighted_avg) ** 2, weights=weights))
            self.avg_price_by_user[int(user_id)] = weighted_avg
            self.price_std_by_user[int(user_id)] = max(math.sqrt(max(variance, 0.0)), weighted_avg * 0.15, 1.0)

        for attribute in ("category", "material", "style"):
            pivot = joined.pivot_table(
                index="user_id",
                columns=attribute,
                values="score",
                aggfunc="sum",
                fill_value=0.0,
            )
            row_sums = pivot.sum(axis=1).replace(0.0, 1.0)
            self.preference_tables[attribute] = pivot.div(row_sums, axis=0)

        purchases = joined[joined["event_type"] == "purchase"].sort_values("timestamp")
        high_intent = joined[joined["event_type"].isin(["purchase", "cart", "wishlist"])].sort_values("timestamp")
        all_interactions = joined.sort_values("timestamp")

        self.purchase_history = {
            int(user_id): dedupe_preserve_order(group["product_id"].tolist())
            for user_id, group in purchases.groupby("user_id")
        }
        self.high_intent_history = {
            int(user_id): dedupe_preserve_order(group["product_id"].tolist())
            for user_id, group in high_intent.groupby("user_id")
        }
        self.interaction_history = {
            int(user_id): dedupe_preserve_order(group["product_id"].tolist())
            for user_id, group in all_interactions.groupby("user_id")
        }
        self.purchased_items = {
            int(user_id): set(product_ids)
            for user_id, product_ids in self.purchase_history.items()
        }
        self.purchase_counts = {
            int(user_id): len(product_ids)
            for user_id, product_ids in self.purchase_history.items()
        }
        self.interaction_counts = {
            int(user_id): len(product_ids)
            for user_id, product_ids in self.interaction_history.items()
        }
        return self

    def get_seed_items(self, user_id: int, *, limit: int = 5) -> list[int]:
        user_id = int(user_id)
        for mapping in (self.purchase_history, self.high_intent_history, self.interaction_history):
            product_ids = mapping.get(user_id, [])
            if product_ids:
                return product_ids[-limit:]
        return []

    def user_avg_price(self, user_id: int) -> float:
        return float(self.avg_price_by_user.get(int(user_id), self.global_avg_price))

    def price_match(self, user_id: int, price: float) -> float:
        user_id = int(user_id)
        average_price = self.user_avg_price(user_id)
        tolerance = float(self.price_std_by_user.get(user_id, max(average_price * 0.15, 1.0)))
        delta = abs(float(price) - average_price)
        return float(math.exp(-delta / max(tolerance, 1.0)))

    def attribute_score(self, user_id: int, attribute: str, value: str) -> float:
        table = self.preference_tables.get(attribute)
        if table is None or int(user_id) not in table.index or value not in table.columns:
            return 0.0
        return float(table.at[int(user_id), value])

    def purchase_count(self, user_id: int) -> int:
        return int(self.purchase_counts.get(int(user_id), 0))

    def interaction_count(self, user_id: int) -> int:
        return int(self.interaction_counts.get(int(user_id), 0))


class ItemStatistics:
    def __init__(self) -> None:
        self.frame: pd.DataFrame | None = None
        self.rank_order: list[int] = []

    def fit(self, interactions: pd.DataFrame) -> "ItemStatistics":
        counts = interactions.pivot_table(
            index="product_id",
            columns="event_type",
            values="score",
            aggfunc="size",
            fill_value=0,
        )
        for column in EVENT_WEIGHTS:
            if column not in counts.columns:
                counts[column] = 0.0
        counts = counts[list(EVENT_WEIGHTS)]

        total_events = counts.sum(axis=1).astype(float)
        positive_events = counts["wishlist"] + counts["cart"] + counts["purchase"]
        historical_ctr = (positive_events + 1.0) / (total_events + 2.0)
        purchase_rate = (counts["purchase"] + 1.0) / (total_events + 2.0)
        interaction_score = interactions.groupby("product_id")["score"].sum().reindex(counts.index).fillna(0.0)

        frame = pd.DataFrame(
            {
                "historical_ctr": historical_ctr.astype(float),
                "purchase_rate": purchase_rate.astype(float),
                "interaction_score": interaction_score.astype(float),
            }
        )
        if frame["interaction_score"].max() > frame["interaction_score"].min():
            frame["popularity_score"] = (
                (frame["interaction_score"] - frame["interaction_score"].min())
                / (frame["interaction_score"].max() - frame["interaction_score"].min())
            )
        else:
            frame["popularity_score"] = 0.0

        self.frame = frame
        self.rank_order = (
            frame.sort_values(
                ["historical_ctr", "purchase_rate", "interaction_score"],
                ascending=False,
            )
            .index.astype(int)
            .tolist()
        )
        return self

    def ctr_series(self, candidate_ids: Iterable[int]) -> pd.Series:
        candidate_list = [int(product_id) for product_id in candidate_ids]
        if self.frame is None:
            return pd.Series(0.0, index=candidate_list, dtype=float)
        series = self.frame["historical_ctr"].reindex(candidate_list).fillna(float(self.frame["historical_ctr"].mean()))
        series.name = "historical_ctr"
        return series.astype(float)

    def recommend(self, *, top_n: int = 20, exclude_product_ids: Iterable[int] | None = None) -> list[int]:
        excluded = {int(product_id) for product_id in (exclude_product_ids or [])}
        return [product_id for product_id in self.rank_order if product_id not in excluded][:top_n]


class ContentScorer:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.embeddings = _normalize_embeddings(catalog.embeddings)
        self.similarity_matrix = self.embeddings @ self.embeddings.T

    def _indices(self, product_ids: Iterable[int]) -> list[int]:
        return [
            self.catalog.product_id_to_index[int(product_id)]
            for product_id in product_ids
            if int(product_id) in self.catalog.product_id_to_index
        ]

    def score_frame_from_history(
        self,
        product_ids: Iterable[int],
        candidate_ids: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        candidate_list = self.catalog.product_ids if candidate_ids is None else [int(product_id) for product_id in candidate_ids]
        seed_indices = self._indices(product_ids)
        if not seed_indices:
            return pd.DataFrame(
                {
                    "product_id": candidate_list,
                    "embedding_similarity": np.zeros(len(candidate_list), dtype=float),
                    "embedding_mean_similarity": np.zeros(len(candidate_list), dtype=float),
                }
            )

        candidate_indices = self._indices(candidate_list)
        similarities = self.similarity_matrix[np.ix_(seed_indices, candidate_indices)]
        return pd.DataFrame(
            {
                "product_id": candidate_list,
                "embedding_similarity": similarities.max(axis=0).astype(float),
                "embedding_mean_similarity": similarities.mean(axis=0).astype(float),
            }
        )

    def score_series_from_query(
        self,
        query_text: str,
        candidate_ids: Iterable[int] | None = None,
    ) -> pd.Series:
        candidate_list = self.catalog.product_ids if candidate_ids is None else [int(product_id) for product_id in candidate_ids]
        candidate_indices = self._indices(candidate_list)
        query_vector = encode_query_text(query_text, model_name=self.catalog.embedding_model_name)
        scores = self.embeddings[candidate_indices] @ query_vector
        return pd.Series(scores.astype(float), index=candidate_list, name="query_similarity")

    def recommend_from_history(
        self,
        product_ids: Iterable[int],
        *,
        top_n: int = 20,
        exclude_product_ids: Iterable[int] | None = None,
        min_similarity: float = 0.0,
    ) -> list[int]:
        excluded = {int(product_id) for product_id in (exclude_product_ids or [])}
        scores = self.score_frame_from_history(product_ids)
        filtered = scores[
            (~scores["product_id"].isin(excluded))
            & (scores["embedding_similarity"] >= float(min_similarity))
        ]
        return (
            filtered.sort_values(
                ["embedding_similarity", "embedding_mean_similarity", "product_id"],
                ascending=[False, False, True],
            )["product_id"]
            .astype(int)
            .tolist()[:top_n]
        )

    def recommend_from_query(
        self,
        query_text: str,
        *,
        top_n: int = 20,
        exclude_product_ids: Iterable[int] | None = None,
        min_similarity: float = 0.0,
    ) -> list[int]:
        excluded = {int(product_id) for product_id in (exclude_product_ids or [])}
        scores = self.score_series_from_query(query_text)
        filtered = scores[(~scores.index.isin(excluded)) & (scores >= float(min_similarity))]
        return filtered.sort_values(ascending=False).index.astype(int).tolist()[:top_n]


def build_candidate_feature_frame(
    *,
    user_id: int,
    candidate_product_ids: Iterable[int],
    catalog: Catalog,
    cf_model: CollaborativeFilterScorer,
    content_model: ContentScorer,
    profile_model: PreferenceProfileModel,
    item_stats: ItemStatistics,
    signal_weights: dict[str, float] | None = None,
    similarity_threshold: float = 0.0,
    query_scores: pd.Series | None = None,
) -> pd.DataFrame:
    candidate_ids = dedupe_preserve_order(candidate_product_ids)
    candidate_frame = catalog.lookup(candidate_ids).copy()
    candidate_frame["user_id"] = int(user_id)

    cf_scores = cf_model.score_series(user_id, candidate_ids)
    history_scores = content_model.score_frame_from_history(profile_model.get_seed_items(user_id), candidate_ids)
    history_scores.set_index("product_id", inplace=True)

    query_series = (
        pd.Series(0.0, index=candidate_ids, dtype=float)
        if query_scores is None
        else query_scores.reindex(candidate_ids).fillna(0.0).astype(float)
    )
    history_series = history_scores["embedding_similarity"].reindex(candidate_ids).fillna(0.0).astype(float)
    combined_embedding = np.maximum(history_series.to_numpy(dtype=float), query_series.to_numpy(dtype=float))

    signal_config = normalize_signal_weights(signal_weights)
    candidate_frame["cf_score"] = cf_scores.reindex(candidate_ids).to_numpy(dtype=float)
    candidate_frame["embedding_similarity"] = combined_embedding
    candidate_frame["query_similarity"] = query_series.reindex(candidate_ids).to_numpy(dtype=float)

    user_average_price = profile_model.user_avg_price(user_id)
    candidate_frame["price_delta_from_user_avg"] = candidate_frame["price"].astype(float) - user_average_price
    candidate_frame["price_sensitivity_match"] = candidate_frame["price"].map(
        lambda price: profile_model.price_match(user_id, float(price))
    )
    candidate_frame["category_match_score"] = candidate_frame["category"].map(
        lambda value: profile_model.attribute_score(user_id, "category", str(value))
    )
    candidate_frame["material_preference_score"] = candidate_frame["material"].map(
        lambda value: profile_model.attribute_score(user_id, "material", str(value))
    )
    candidate_frame["style_affinity_score"] = candidate_frame["style"].map(
        lambda value: profile_model.attribute_score(user_id, "style", str(value))
    )
    candidate_frame["historical_ctr"] = item_stats.ctr_series(candidate_ids).reindex(candidate_ids).to_numpy(dtype=float)
    candidate_frame["purchase_history_count"] = profile_model.purchase_count(user_id)
    candidate_frame["interaction_history_count"] = profile_model.interaction_count(user_id)

    gated_embedding = np.where(
        candidate_frame["embedding_similarity"].to_numpy(dtype=float) >= float(similarity_threshold),
        candidate_frame["embedding_similarity"].to_numpy(dtype=float),
        0.0,
    )
    candidate_frame["hybrid_signal_score"] = (
        signal_config["cf"] * candidate_frame["cf_score"].to_numpy(dtype=float)
        + signal_config["embedding"] * gated_embedding
        + signal_config["price"] * candidate_frame["price_sensitivity_match"].to_numpy(dtype=float)
        + signal_config["style"] * candidate_frame["style_affinity_score"].to_numpy(dtype=float)
        + signal_config["ctr"] * candidate_frame["historical_ctr"].to_numpy(dtype=float)
    )
    return candidate_frame
