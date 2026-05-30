from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    import faiss
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: faiss-cpu. Install it with `pip install faiss-cpu numpy`."
    ) from exc

from embedder import (
    DEFAULT_MODEL_KEY,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    PRODUCTS_DATA_PATH,
    build_and_save_embeddings,
    generate_embeddings,
    metadata_matches_data_source,
)


BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = BASE_DIR / "vector_db"
INDEX_DIR = VECTOR_DB_DIR / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"
SEARCH_DEMO_PATH = VECTOR_DB_DIR / "search_demo_results.json"
HNSW_PARAMETER_CANDIDATES = (
    {"hnsw_m": 16, "ef_construction": 80, "ef_search": 32},
    {"hnsw_m": 32, "ef_construction": 120, "ef_search": 64},
    {"hnsw_m": 48, "ef_construction": 200, "ef_search": 128},
)
DEFAULT_SEARCH_QUERIES = (
    {"query": "modern fabric sofa", "filters": {"category": "Sofa"}},
    {"query": "industrial wood wardrobe", "filters": {"category": "Wardrobe"}},
    {
        "query": "minimalist grey chair",
        "filters": {"category": "Chair", "min_price": 100, "max_price": 1200},
    },
)
SearchResult = dict[str, object]


def ensure_embedding_artifacts(model_name: str | None = None) -> None:
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        build_and_save_embeddings(source_path=PRODUCTS_DATA_PATH, model_name=model_name)
        return

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build_and_save_embeddings(source_path=PRODUCTS_DATA_PATH, model_name=model_name)
        return

    if not metadata_matches_data_source(metadata, PRODUCTS_DATA_PATH, model_name=model_name):
        build_and_save_embeddings(source_path=PRODUCTS_DATA_PATH, model_name=model_name)


def load_embeddings() -> np.ndarray:
    ensure_embedding_artifacts()
    return np.asarray(np.load(EMBEDDINGS_PATH), dtype="float32")


def load_metadata() -> dict[str, object]:
    ensure_embedding_artifacts()
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_index_metadata() -> dict[str, object]:
    return json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))


def get_products() -> list[dict[str, object]]:
    metadata = load_metadata()
    return list(metadata["products"])


def get_embeddings_source_info() -> dict[str, object]:
    resolved_path = EMBEDDINGS_PATH.resolve()
    stats = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "modified_time_ns": stats.st_mtime_ns,
        "size_bytes": stats.st_size,
    }


def get_active_model_key(metadata: dict[str, object] | None = None) -> str:
    payload = metadata or load_metadata()
    embedding_config = payload.get("embedding_config")
    if isinstance(embedding_config, dict):
        model_key = embedding_config.get("model_key")
        if isinstance(model_key, str) and model_key:
            return model_key

    model_name = payload.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name

    return DEFAULT_MODEL_KEY


def build_hnsw_index(
    embeddings: np.ndarray,
    *,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
):
    index = faiss.index_factory(
        embeddings.shape[1],
        f"HNSW{hnsw_m},Flat",
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)
    return index


def compute_exact_neighbor_indices(embeddings: np.ndarray, top_k: int) -> np.ndarray:
    similarity_matrix = embeddings @ embeddings.T
    np.fill_diagonal(similarity_matrix, -np.inf)
    return np.argsort(-similarity_matrix, axis=1)[:, :top_k]


def compute_neighbor_recall(
    approx_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    recalls: list[float] = []
    for row_index, truth_indices in enumerate(ground_truth):
        truth = {int(index) for index in truth_indices.tolist()}
        predicted = [
            int(index)
            for index in approx_indices[row_index].tolist()
            if index != row_index and index != -1
        ][: len(truth_indices)]
        if not truth:
            continue
        recalls.append(len(truth.intersection(predicted)) / len(truth))
    return float(np.mean(recalls)) if recalls else 0.0


def optimize_hnsw_parameters(
    embeddings: np.ndarray,
    *,
    top_k: int = 10,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if embeddings.shape[0] <= 2:
        selected = dict(HNSW_PARAMETER_CANDIDATES[0])
        return selected, [selected]

    effective_top_k = min(top_k, embeddings.shape[0] - 1)
    ground_truth = compute_exact_neighbor_indices(embeddings, effective_top_k)
    evaluation_results: list[dict[str, object]] = []

    for candidate in HNSW_PARAMETER_CANDIDATES:
        build_started = perf_counter()
        index = build_hnsw_index(embeddings, **candidate)
        build_time_ms = (perf_counter() - build_started) * 1000.0

        search_started = perf_counter()
        _, approx_indices = index.search(embeddings, effective_top_k + 1)
        total_search_time_ms = (perf_counter() - search_started) * 1000.0

        evaluation_results.append(
            {
                **candidate,
                f"recall_at_{effective_top_k}": round(
                    compute_neighbor_recall(approx_indices, ground_truth),
                    4,
                ),
                "build_time_ms": round(build_time_ms, 3),
                "average_query_time_ms": round(
                    total_search_time_ms / embeddings.shape[0],
                    4,
                ),
            }
        )

    recall_key = f"recall_at_{effective_top_k}"
    selected = max(
        evaluation_results,
        key=lambda item: (item[recall_key], -item["average_query_time_ms"], -item["build_time_ms"]),
    )
    selected_config = {
        "hnsw_m": int(selected["hnsw_m"]),
        "ef_construction": int(selected["ef_construction"]),
        "ef_search": int(selected["ef_search"]),
    }
    return selected_config, evaluation_results


def save_index_metadata(
    *,
    embeddings: np.ndarray,
    metadata: dict[str, object],
    selected_parameters: dict[str, object],
    candidate_evaluation: list[dict[str, object]],
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_type": "faiss_hnsw_flat",
        "product_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_source": get_embeddings_source_info(),
        "embedding_model": get_active_model_key(metadata),
        "selected_parameters": selected_parameters,
        "candidate_evaluation": candidate_evaluation,
    }
    INDEX_METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_and_save_index(embeddings: np.ndarray | None = None) -> str:
    matrix = np.asarray(embeddings if embeddings is not None else load_embeddings(), dtype="float32")
    metadata = load_metadata()
    selected_parameters, candidate_evaluation = optimize_hnsw_parameters(matrix)
    index = build_hnsw_index(matrix, **selected_parameters)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    save_index_metadata(
        embeddings=matrix,
        metadata=metadata,
        selected_parameters=selected_parameters,
        candidate_evaluation=candidate_evaluation,
    )
    return str(INDEX_PATH)


def index_metadata_matches_embeddings(index_metadata: dict[str, object]) -> bool:
    expected_source = get_embeddings_source_info()
    embedding_source = index_metadata.get("embedding_source")
    if not isinstance(embedding_source, dict):
        return False

    for key, expected_value in expected_source.items():
        if embedding_source.get(key) != expected_value:
            return False

    embeddings = load_embeddings()
    if index_metadata.get("product_count") != int(embeddings.shape[0]):
        return False
    if index_metadata.get("embedding_dimension") != int(embeddings.shape[1]):
        return False

    parameters = index_metadata.get("selected_parameters")
    return isinstance(parameters, dict)


def ensure_index_artifacts() -> None:
    ensure_embedding_artifacts()
    if not INDEX_PATH.exists() or not INDEX_METADATA_PATH.exists():
        create_and_save_index()
        return

    try:
        metadata = load_index_metadata()
    except (OSError, json.JSONDecodeError):
        create_and_save_index()
        return

    if not index_metadata_matches_embeddings(metadata):
        create_and_save_index()


def load_index():
    ensure_index_artifacts()
    index = faiss.read_index(str(INDEX_PATH))

    try:
        metadata = load_index_metadata()
    except (OSError, json.JSONDecodeError):
        return index

    parameters = metadata.get("selected_parameters")
    if isinstance(parameters, dict):
        index.hnsw.efSearch = int(parameters.get("ef_search", index.hnsw.efSearch))
    return index


def _normalize_query_vector(query_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(query_vector, dtype="float32")
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    return vector / np.clip(norms, 1e-12, None)


def _normalize_filter_value(value: object) -> str:
    return str(value).strip().lower()


def build_filters(
    *,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    style: str | None = None,
    material: str | None = None,
    color: str | None = None,
) -> dict[str, object] | None:
    filters: dict[str, object] = {}
    if category:
        filters["category"] = category
    if min_price is not None:
        filters["min_price"] = float(min_price)
    if max_price is not None:
        filters["max_price"] = float(max_price)
    if style:
        filters["style"] = style
    if material:
        filters["material"] = material
    if color:
        filters["color"] = color
    return filters or None


def product_matches_filters(
    product: dict[str, object],
    filters: dict[str, object] | None,
) -> bool:
    if not filters:
        return True

    for attribute in ("category", "style", "material", "color"):
        filter_value = filters.get(attribute)
        if filter_value is None:
            continue

        if _normalize_filter_value(product.get(attribute, "")) != _normalize_filter_value(filter_value):
            return False

    price = float(product["price"])
    min_price = filters.get("min_price")
    max_price = filters.get("max_price")

    if min_price is not None and price < float(min_price):
        return False
    if max_price is not None and price > float(max_price):
        return False
    return True


def search_index(
    query_vector: np.ndarray,
    *,
    top_k: int = 3,
    exclude_indices: set[int] | None = None,
    filters: dict[str, object] | None = None,
    similarity_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    index = load_index()
    products = get_products()
    query = _normalize_query_vector(query_vector)
    excluded = exclude_indices or set()
    available_count = max(index.ntotal - len(excluded), 0)
    target_top_k = min(top_k, available_count)
    if target_top_k <= 0:
        return (
            np.asarray([], dtype="float32"),
            np.asarray([], dtype="int64"),
        )

    candidate_count = index.ntotal if filters or similarity_threshold is not None else min(
        index.ntotal,
        max(target_top_k * 10, target_top_k + len(excluded)),
    )
    scores, indices = index.search(query, candidate_count)

    filtered_scores: list[float] = []
    filtered_indices: list[int] = []

    for score, index_value in zip(scores[0], indices[0]):
        if index_value == -1 or index_value in excluded:
            continue
        if similarity_threshold is not None and float(score) < similarity_threshold:
            continue

        product = products[int(index_value)]
        if not product_matches_filters(product, filters):
            continue

        filtered_scores.append(float(score))
        filtered_indices.append(int(index_value))
        if len(filtered_indices) == target_top_k:
            break

    return (
        np.asarray(filtered_scores, dtype="float32"),
        np.asarray(filtered_indices, dtype="int64"),
    )


def build_result_records(scores: np.ndarray, indices: np.ndarray) -> list[SearchResult]:
    products = get_products()
    results: list[SearchResult] = []

    for rank, (score, product_index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        product = products[product_index]
        results.append(
            {
                "rank": rank,
                "score": round(float(score), 4),
                "product_id": product["product_id"],
                "title": product["title"],
                "category": product["category"],
                "style": product["style"],
                "color": product["color"],
                "material": product["material"],
                "price": product["price"],
                "rating": product["rating"],
                "description": product["description"],
                "catalog_description": product["catalog_description"],
            }
        )

    return results


def search_by_text(
    query_text: str,
    *,
    top_k: int = 3,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    similarity_threshold: float | None = None,
    style: str | None = None,
    material: str | None = None,
    color: str | None = None,
) -> list[SearchResult]:
    metadata = load_metadata()
    model_key = get_active_model_key(metadata)
    filters = build_filters(
        category=category,
        min_price=min_price,
        max_price=max_price,
        style=style,
        material=material,
        color=color,
    )
    query_embedding = generate_embeddings([query_text], model_name=model_key, is_query=True)[0]
    scores, indices = search_index(
        query_embedding,
        top_k=top_k,
        filters=filters,
        similarity_threshold=similarity_threshold,
    )
    return build_result_records(scores, indices)


def find_product_index(product_id: str) -> int:
    for index, product in enumerate(get_products()):
        if product["product_id"] == str(product_id):
            return index
    raise ValueError(f"Unknown product_id: {product_id}")


def recommend_similar_products(
    product_id: str,
    *,
    top_k: int = 3,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    similarity_threshold: float | None = None,
) -> list[SearchResult]:
    product_index = find_product_index(product_id)
    embeddings = load_embeddings()
    filters = build_filters(
        category=category,
        min_price=min_price,
        max_price=max_price,
    )
    scores, indices = search_index(
        embeddings[product_index],
        top_k=top_k,
        exclude_indices={product_index},
        filters=filters,
        similarity_threshold=similarity_threshold,
    )
    return build_result_records(scores, indices)


def run_search_demo(*, top_k: int = 3) -> list[dict[str, object]]:
    demo_results: list[dict[str, object]] = []
    for sample in DEFAULT_SEARCH_QUERIES:
        filters = sample.get("filters", {})
        results = search_by_text(
            sample["query"],
            top_k=top_k,
            category=filters.get("category"),
            min_price=filters.get("min_price"),
            max_price=filters.get("max_price"),
            style=filters.get("style"),
            material=filters.get("material"),
            color=filters.get("color"),
        )
        demo_results.append(
            {
                "query": sample["query"],
                "filters": filters,
                "results": results,
            }
        )

    SEARCH_DEMO_PATH.write_text(json.dumps(demo_results, indent=2), encoding="utf-8")
    return demo_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query the FAISS vector index.")
    parser.add_argument("--query", help="Optional single query to search.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return.")
    parser.add_argument("--category", help="Optional category filter.")
    parser.add_argument("--min-price", type=float, help="Optional minimum price filter.")
    parser.add_argument("--max-price", type=float, help="Optional maximum price filter.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional similarity threshold for returned results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_and_save_index()

    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved index metadata to: {INDEX_METADATA_PATH}")

    if args.query:
        results = search_by_text(
            args.query,
            top_k=args.top_k,
            category=args.category,
            min_price=args.min_price,
            max_price=args.max_price,
            similarity_threshold=args.threshold,
        )
        print(f"\nQuery: {args.query}")
        for result in results:
            print(
                f"{result['rank']}. {result['title']} ({result['product_id']}) | "
                f"score={result['score']:.4f} | {result['description']}"
            )
        return

    demo_results = run_search_demo(top_k=args.top_k)
    print(f"Saved search demo to: {SEARCH_DEMO_PATH}")
    for demo in demo_results:
        print(f"\nQuery: {demo['query']} | filters={demo['filters']}")
        for result in demo["results"]:
            print(
                f"{result['rank']}. {result['title']} ({result['product_id']}) | "
                f"score={result['score']:.4f} | {result['description']}"
            )


if __name__ == "__main__":
    main()
