from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Install it with "
        "`pip install sentence-transformers numpy pandas pyarrow`."
    ) from exc


MODEL_REGISTRY = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
DEFAULT_MODEL_KEY = "all-MiniLM-L6-v2"
DEFAULT_PCA_DIMENSIONS = (64, 128)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PRODUCTS_DATA_PATH = DATA_DIR / "products_clean.parquet"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "product_embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "embedding_metadata.json"
EVALUATION_PATH = EMBEDDINGS_DIR / "embedding_evaluation.json"
HF_CACHE_DIR = Path(tempfile.gettempdir()) / "genai_recommender_hf_cache"
SOURCE_SCHEMA_VERSION = 3
REQUIRED_COLUMNS = {
    "product_id",
    "name",
    "category",
    "material",
    "color",
    "style",
    "price",
    "rating",
    "description",
}
ProductRecord = dict[str, object]


def configure_model_cache() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(HF_CACHE_DIR / "sentence_transformers"),
    )


def resolve_model_key(model_name: str | None = None) -> tuple[str, str]:
    requested = (model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_KEY)).strip()
    if not requested:
        requested = DEFAULT_MODEL_KEY

    requested_lower = requested.lower()
    for key, resolved_name in MODEL_REGISTRY.items():
        if requested == key or requested == resolved_name:
            return key, resolved_name
        if requested_lower == key.lower() or requested_lower == resolved_name.lower():
            return key, resolved_name

    supported = ", ".join(MODEL_REGISTRY)
    raise ValueError(f"Unsupported embedding model: {requested}. Choose from {supported}.")


def get_embedding_config(model_name: str | None = None) -> dict[str, object]:
    model_key, resolved_model_name = resolve_model_key(model_name)
    return {
        "model_key": model_key,
        "resolved_model_name": resolved_model_name,
        "normalize_embeddings": True,
    }


def get_data_source_info(data_path: Path = PRODUCTS_DATA_PATH) -> dict[str, object]:
    resolved_path = data_path.resolve()
    stats = resolved_path.stat()
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "path": str(resolved_path),
        "modified_time_ns": stats.st_mtime_ns,
        "size_bytes": stats.st_size,
    }


def _clean_text(value: object, default: str = "Unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default

    text = " ".join(str(value).replace("_", " ").split())
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def _normalize_label(value: object) -> str:
    return _clean_text(value).title()


def _normalize_price(value: object) -> float:
    return round(float(value), 2)


def _normalize_rating(value: object) -> float:
    return round(float(value), 2)


def _format_price(price: float) -> str:
    if float(price).is_integer():
        return str(int(price))
    return f"{price:.2f}"


def build_description(product: ProductRecord) -> str:
    return (
        f"{product['style']} {product['material']} {product['category']} "
        f"in {product['color']} color. Price: {_format_price(float(product['price']))}."
    )


def normalize_product_record(row: dict[str, object]) -> ProductRecord:
    product: ProductRecord = {
        "product_id": str(row["product_id"]),
        "title": _clean_text(row["name"]),
        "category": _normalize_label(row["category"]),
        "material": _normalize_label(row["material"]),
        "color": _normalize_label(row["color"]),
        "style": _normalize_label(row["style"]),
        "price": _normalize_price(row["price"]),
        "rating": _normalize_rating(row["rating"]),
        "catalog_description": _clean_text(row["description"], default=""),
    }
    product["description"] = build_description(product)
    return product


def load_products_from_data(data_path: Path = PRODUCTS_DATA_PATH) -> list[ProductRecord]:
    if not data_path.exists():
        raise FileNotFoundError(f"Product dataset not found at {data_path}")

    dataframe = pd.read_parquet(data_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    return [
        normalize_product_record(record)
        for record in dataframe.to_dict(orient="records")
    ]


def prepare_product_data(
    products: list[ProductRecord],
) -> tuple[list[str], list[ProductRecord]]:
    descriptions: list[str] = []
    metadata: list[ProductRecord] = []

    for product in products:
        description = build_description(product)
        descriptions.append(description)
        metadata.append({**product, "description": description})

    return descriptions, metadata


def metadata_matches_data_source(
    metadata: dict[str, object],
    data_path: Path = PRODUCTS_DATA_PATH,
    model_name: str | None = None,
) -> bool:
    data_source = metadata.get("data_source")
    if not isinstance(data_source, dict):
        return False

    expected_source = get_data_source_info(data_path)
    for key, expected_value in expected_source.items():
        if data_source.get(key) != expected_value:
            return False

    product_count = metadata.get("product_count")
    if not isinstance(product_count, int) or product_count <= 0:
        return False

    embedding_config = metadata.get("embedding_config")
    if not isinstance(embedding_config, dict):
        return False

    expected_config = get_embedding_config(model_name)
    for key, expected_value in expected_config.items():
        if embedding_config.get(key) != expected_value:
            return False

    return True


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str = DEFAULT_MODEL_KEY) -> SentenceTransformer:
    _, resolved_model_name = resolve_model_key(model_name)
    configure_model_cache()
    return SentenceTransformer(resolved_model_name)


def prepare_texts_for_model(
    texts: list[str],
    model_name: str | None = None,
    *,
    is_query: bool = False,
) -> list[str]:
    _, resolved_model_name = resolve_model_key(model_name)
    normalized_texts = [_clean_text(text, default="") for text in texts]

    if "bge-" in resolved_model_name.lower() and is_query:
        prefix = "Represent this sentence for searching relevant passages: "
        return [prefix + text for text in normalized_texts]

    return normalized_texts


def generate_embeddings(
    descriptions: list[str],
    model_name: str | None = None,
    *,
    is_query: bool = False,
) -> np.ndarray:
    texts = prepare_texts_for_model(descriptions, model_name=model_name, is_query=is_query)
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = get_embedding_model(model_name or DEFAULT_MODEL_KEY)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def summarize_scores(scores: np.ndarray) -> dict[str, object]:
    values = np.asarray(scores, dtype="float32")
    if values.size == 0:
        return {"count": 0}

    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "count": int(values.size),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "p10": round(float(percentiles[0]), 4),
        "p25": round(float(percentiles[1]), 4),
        "p50": round(float(percentiles[2]), 4),
        "p75": round(float(percentiles[3]), 4),
        "p90": round(float(percentiles[4]), 4),
    }


def build_relevance_masks(metadata: list[ProductRecord]) -> dict[str, np.ndarray]:
    categories = np.asarray([product["category"] for product in metadata], dtype=object)
    materials = np.asarray([product["material"] for product in metadata], dtype=object)
    styles = np.asarray([product["style"] for product in metadata], dtype=object)
    colors = np.asarray([product["color"] for product in metadata], dtype=object)

    same_category = categories[:, None] == categories[None, :]
    same_material = materials[:, None] == materials[None, :]
    same_style = styles[:, None] == styles[None, :]
    same_color = colors[:, None] == colors[None, :]

    diagonal = np.eye(len(metadata), dtype=bool)
    structured_relevance = same_category & (same_material | same_style | same_color)
    exact_attribute_match = same_category & same_material & same_style & same_color
    different_category = ~same_category

    for mask in (
        same_category,
        same_material,
        same_style,
        same_color,
        structured_relevance,
        exact_attribute_match,
        different_category,
    ):
        mask[diagonal] = False

    attribute_overlap = (
        same_category.astype(np.int16)
        + same_material.astype(np.int16)
        + same_style.astype(np.int16)
        + same_color.astype(np.int16)
    )
    attribute_overlap[diagonal] = 0

    return {
        "same_category": same_category,
        "same_material": same_material,
        "same_style": same_style,
        "same_color": same_color,
        "structured_relevance": structured_relevance,
        "exact_attribute_match": exact_attribute_match,
        "different_category": different_category,
        "attribute_overlap": attribute_overlap,
    }


def compute_similarity_statistics(
    embeddings: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    similarity_matrix = embeddings @ embeddings.T
    upper_triangle = np.triu_indices(similarity_matrix.shape[0], k=1)

    def select(mask_name: str) -> np.ndarray:
        return similarity_matrix[upper_triangle][masks[mask_name][upper_triangle]]

    return {
        "overall_pairs": summarize_scores(similarity_matrix[upper_triangle]),
        "same_category_pairs": summarize_scores(select("same_category")),
        "structured_relevance_pairs": summarize_scores(select("structured_relevance")),
        "exact_attribute_match_pairs": summarize_scores(select("exact_attribute_match")),
        "different_category_pairs": summarize_scores(select("different_category")),
    }


def _mean_reciprocal_rank(relevance_hits: np.ndarray) -> float:
    reciprocal_ranks = np.zeros(relevance_hits.shape[0], dtype="float32")
    for row_index, hits in enumerate(relevance_hits):
        positions = np.flatnonzero(hits)
        if positions.size:
            reciprocal_ranks[row_index] = 1.0 / float(positions[0] + 1)
    return float(reciprocal_ranks.mean())


def _mean_ndcg(
    rankings: np.ndarray,
    graded_relevance: np.ndarray,
    top_k: int,
) -> float:
    discounts = 1.0 / np.log2(np.arange(top_k, dtype="float32") + 2.0)
    scores: list[float] = []

    for row_index, ranked_indices in enumerate(rankings):
        gains = graded_relevance[row_index, ranked_indices[:top_k]].astype("float32")
        dcg = float(np.sum((2.0**gains - 1.0) * discounts[: gains.size]))

        ideal_gains = np.sort(graded_relevance[row_index])[::-1][:top_k].astype("float32")
        ideal_dcg = float(np.sum((2.0**ideal_gains - 1.0) * discounts[: ideal_gains.size]))
        scores.append(0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg)

    return float(np.mean(scores))


def evaluate_retrieval(
    embeddings: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    top_k_values: tuple[int, ...] = (5, 10),
) -> dict[str, object]:
    similarity_matrix = embeddings @ embeddings.T
    np.fill_diagonal(similarity_matrix, -np.inf)

    rankings = np.argsort(-similarity_matrix, axis=1)
    row_indices = np.arange(rankings.shape[0])[:, None]
    primary_relevance = masks["structured_relevance"]
    category_relevance = masks["same_category"]
    total_primary_relevant = primary_relevance.sum(axis=1)
    total_exact_attribute_matches = masks["exact_attribute_match"].sum(axis=1)
    valid_queries = total_primary_relevant > 0

    metrics: dict[str, object] = {
        "relevance_definition": (
            "Relevant neighbors share the same category and at least one of "
            "style, material, or color."
        ),
        "queries_with_relevant_neighbors": int(valid_queries.sum()),
        "average_relevant_neighbors_per_query": round(
            float(total_primary_relevant.mean()), 4
        ),
        "average_exact_attribute_matches_per_query": round(
            float(total_exact_attribute_matches.mean()), 4
        ),
    }

    for top_k in top_k_values:
        ranked_indices = rankings[:, :top_k]
        primary_hits = primary_relevance[row_indices, ranked_indices]
        category_hits = category_relevance[row_indices, ranked_indices]
        graded_hits = masks["attribute_overlap"][row_indices, ranked_indices]
        top_scores = similarity_matrix[row_indices, ranked_indices]

        precision = primary_hits.mean(axis=1)
        recall = np.zeros(rankings.shape[0], dtype="float32")
        recall[valid_queries] = (
            primary_hits.sum(axis=1)[valid_queries] / total_primary_relevant[valid_queries]
        )

        metrics[f"precision_at_{top_k}"] = round(float(precision.mean()), 4)
        metrics[f"recall_at_{top_k}"] = round(
            float(recall[valid_queries].mean()) if valid_queries.any() else 0.0,
            4,
        )
        metrics[f"category_match_rate_at_{top_k}"] = round(
            float(category_hits.mean()),
            4,
        )
        metrics[f"mean_similarity_at_{top_k}"] = round(float(top_scores.mean()), 4)
        metrics[f"mean_attribute_overlap_at_{top_k}"] = round(
            float(graded_hits.mean()),
            4,
        )

    max_top_k = max(top_k_values)
    relevance_hits = primary_relevance[row_indices, rankings[:, :max_top_k]]
    metrics[f"mrr_at_{max_top_k}"] = round(_mean_reciprocal_rank(relevance_hits), 4)
    metrics[f"ndcg_at_{max_top_k}"] = round(
        _mean_ndcg(rankings, masks["attribute_overlap"], max_top_k),
        4,
    )
    return metrics


def reduce_embeddings_with_pca(
    embeddings: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(embeddings, dtype="float32")
    max_components = min(matrix.shape[0] - 1, matrix.shape[1])
    if n_components <= 1 or n_components >= max_components:
        raise ValueError(
            f"PCA components must be between 2 and {max_components - 1} for this dataset."
        )

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    projection = right_vectors[:n_components]
    reduced = centered @ projection.T
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.clip(norms, 1e-12, None)

    explained_variance = singular_values**2
    explained_ratio = float(
        explained_variance[:n_components].sum() / explained_variance.sum()
    )
    return reduced.astype("float32"), explained_ratio


def evaluate_dimensionality_reduction(
    embeddings: np.ndarray,
    masks: dict[str, np.ndarray],
    dimensions: tuple[int, ...] = DEFAULT_PCA_DIMENSIONS,
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for dimension in dimensions:
        if dimension >= embeddings.shape[1]:
            continue

        reduced_embeddings, explained_ratio = reduce_embeddings_with_pca(
            embeddings,
            dimension,
        )
        evaluations.append(
            {
                "method": "PCA",
                "target_dimension": dimension,
                "explained_variance_ratio": round(explained_ratio, 4),
                "retrieval_metrics": evaluate_retrieval(reduced_embeddings, masks),
            }
        )

    return evaluations


def benchmark_embedding_models(
    descriptions: list[str],
    metadata: list[ProductRecord],
    model_keys: list[str],
) -> dict[str, object]:
    masks = build_relevance_masks(metadata)
    benchmarks: dict[str, object] = {}

    for model_key in model_keys:
        try:
            embeddings = generate_embeddings(descriptions, model_name=model_key)
            benchmarks[model_key] = {
                "status": "ok",
                "embedding_dimension": int(embeddings.shape[1]),
                "retrieval_metrics": evaluate_retrieval(embeddings, masks),
            }
        except Exception as exc:  # pragma: no cover - depends on local model availability
            benchmarks[model_key] = {
                "status": "failed",
                "error": str(exc),
            }

    return benchmarks


def save_outputs(
    embeddings: np.ndarray,
    metadata: list[ProductRecord],
    model_name: str | None = None,
    source_path: Path = PRODUCTS_DATA_PATH,
) -> dict[str, object]:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    embedding_config = get_embedding_config(model_name)
    payload: dict[str, object] = {
        "model_name": embedding_config["model_key"],
        "resolved_model_name": embedding_config["resolved_model_name"],
        "normalize_embeddings": True,
        "embedding_dimension": int(embeddings.shape[1]),
        "product_count": len(metadata),
        "data_source": get_data_source_info(source_path),
        "embedding_config": embedding_config,
        "products": metadata,
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_evaluation_report(report: dict[str, object]) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_evaluation_report(
    embeddings: np.ndarray,
    metadata: list[ProductRecord],
    *,
    model_name: str | None = None,
    benchmark_models: list[str] | None = None,
) -> dict[str, object]:
    embedding_config = get_embedding_config(model_name)
    masks = build_relevance_masks(metadata)
    report: dict[str, object] = {
        "model_name": embedding_config["model_key"],
        "resolved_model_name": embedding_config["resolved_model_name"],
        "product_count": len(metadata),
        "embedding_dimension": int(embeddings.shape[1]),
        "similarity_score_distribution": compute_similarity_statistics(embeddings, masks),
        "retrieval_metrics": evaluate_retrieval(embeddings, masks),
        "dimensionality_reduction_experiments": evaluate_dimensionality_reduction(
            embeddings,
            masks,
        ),
    }

    if benchmark_models:
        descriptions = [product["description"] for product in metadata]
        report["model_benchmarks"] = benchmark_embedding_models(
            descriptions,
            metadata,
            benchmark_models,
        )

    return report


def build_and_save_embeddings(
    products: list[ProductRecord] | None = None,
    source_path: Path = PRODUCTS_DATA_PATH,
    *,
    model_name: str | None = None,
    benchmark_models: list[str] | None = None,
    save_evaluation: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    catalog = load_products_from_data(source_path) if products is None else products
    descriptions, metadata = prepare_product_data(catalog)
    embeddings = generate_embeddings(descriptions, model_name=model_name)
    payload = save_outputs(embeddings, metadata, model_name=model_name, source_path=source_path)

    if save_evaluation:
        report = build_evaluation_report(
            embeddings,
            metadata,
            model_name=model_name,
            benchmark_models=benchmark_models,
        )
        save_evaluation_report(report)

    return embeddings, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate product embeddings and evaluation.")
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_KEY),
        help="Embedding model to use. Supported: " + ", ".join(MODEL_REGISTRY),
    )
    parser.add_argument(
        "--benchmark-models",
        nargs="*",
        default=None,
        help="Optional list of additional embedding models to benchmark.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip embedding evaluation and only save vectors plus metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings, payload = build_and_save_embeddings(
        model_name=args.model,
        benchmark_models=args.benchmark_models,
        save_evaluation=not args.skip_evaluation,
    )
    print(f"Loaded products from: {PRODUCTS_DATA_PATH}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    if not args.skip_evaluation:
        print(f"Saved embedding evaluation to: {EVALUATION_PATH}")
    print(
        "Embedded "
        f"{payload['product_count']} products with shape {tuple(embeddings.shape)} "
        f"using {payload['model_name']}."
    )


if __name__ == "__main__":
    main()
