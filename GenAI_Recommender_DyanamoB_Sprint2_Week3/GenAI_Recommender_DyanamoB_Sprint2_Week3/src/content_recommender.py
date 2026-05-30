from __future__ import annotations

import argparse
import json

from vector_store import (
    VECTOR_DB_DIR,
    build_filters,
    ensure_index_artifacts,
    find_product_index,
    get_products,
    load_embeddings,
    recommend_similar_products,
    run_search_demo,
    search_by_text,
    search_index,
)


RECOMMENDATION_DEMO_PATH = VECTOR_DB_DIR / "recommendation_demo.json"


def recommend_from_query(
    query_text: str,
    *,
    top_k: int = 3,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    similarity_threshold: float | None = None,
) -> list[dict[str, object]]:
    ensure_index_artifacts()
    return search_by_text(
        query_text,
        top_k=top_k,
        category=category,
        min_price=min_price,
        max_price=max_price,
        similarity_threshold=similarity_threshold,
    )


def recommend_from_product(
    product_id: str,
    *,
    top_k: int = 3,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    similarity_threshold: float | None = None,
) -> list[dict[str, object]]:
    ensure_index_artifacts()
    return recommend_similar_products(
        product_id,
        top_k=top_k,
        category=category,
        min_price=min_price,
        max_price=max_price,
        similarity_threshold=similarity_threshold,
    )


def recommend_for_user(
    purchased_product_ids: list[str],
    *,
    top_k: int = 5,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    similarity_threshold: float | None = None,
) -> list[dict[str, object]]:
    ensure_index_artifacts()
    products = get_products()
    embeddings = load_embeddings()
    purchased_ids = [str(product_id) for product_id in purchased_product_ids]
    purchased_indices = [find_product_index(product_id) for product_id in purchased_ids]
    purchased_index_set = set(purchased_indices)
    filters = build_filters(
        category=category,
        min_price=min_price,
        max_price=max_price,
    )

    aggregated_candidates: dict[int, dict[str, object]] = {}
    candidate_pool = min(len(products), max(top_k * 10, top_k + len(purchased_indices)))

    for purchased_index in purchased_indices:
        scores, indices = search_index(
            embeddings[purchased_index],
            top_k=candidate_pool,
            exclude_indices=purchased_index_set,
            filters=filters,
            similarity_threshold=similarity_threshold,
        )
        source_product_id = products[purchased_index]["product_id"]

        for score, candidate_index in zip(scores.tolist(), indices.tolist()):
            candidate = aggregated_candidates.setdefault(
                int(candidate_index),
                {
                    "score_max": float("-inf"),
                    "score_sum": 0.0,
                    "match_count": 0,
                    "matched_from_product_ids": [],
                },
            )
            candidate["score_max"] = max(candidate["score_max"], float(score))
            candidate["score_sum"] += float(score)
            candidate["match_count"] += 1
            candidate["matched_from_product_ids"].append(source_product_id)

    ranked_candidates = sorted(
        aggregated_candidates.items(),
        key=lambda item: (
            -float(item[1]["score_max"]),
            -int(item[1]["match_count"]),
            -(float(item[1]["score_sum"]) / max(int(item[1]["match_count"]), 1)),
            item[0],
        ),
    )

    recommendations: list[dict[str, object]] = []
    for rank, (candidate_index, aggregate) in enumerate(ranked_candidates[:top_k], start=1):
        product = products[int(candidate_index)]
        mean_score = float(aggregate["score_sum"]) / max(int(aggregate["match_count"]), 1)
        recommendations.append(
            {
                "rank": rank,
                "score": round(float(aggregate["score_max"]), 4),
                "mean_score": round(mean_score, 4),
                "support_count": int(aggregate["match_count"]),
                "matched_from_product_ids": aggregate["matched_from_product_ids"],
                "product_id": product["product_id"],
                "title": product["title"],
                "category": product["category"],
                "style": product["style"],
                "color": product["color"],
                "material": product["material"],
                "price": product["price"],
                "rating": product["rating"],
                "description": product["description"],
            }
        )

    return recommendations


def build_sample_user_profiles(limit: int = 3) -> list[dict[str, object]]:
    category_groups: dict[str, list[str]] = {}
    for product in get_products():
        category_groups.setdefault(str(product["category"]), []).append(str(product["product_id"]))

    profiles: list[dict[str, object]] = []
    for category, product_ids in sorted(
        category_groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        if len(product_ids) < 2:
            continue

        profiles.append(
            {
                "user_id": f"{category.lower()}_shopper",
                "purchased_product_ids": product_ids[:2],
                "preferred_category": category,
            }
        )
        if len(profiles) == limit:
            break

    return profiles


def print_results(title: str, results: list[dict[str, object]]) -> None:
    print(f"\n{title}")
    if not results:
        print("No results found.")
        return

    for result in results:
        support = result.get("support_count")
        support_text = f" | support={support}" if support is not None else ""
        print(
            f"{result['rank']}. {result['title']} ({result['product_id']}) | "
            f"score={result['score']:.4f}{support_text} | "
            f"{result['category']} | {result['material']} | {result['price']}"
        )


def run_recommendation_demo(*, top_k: int = 3) -> dict[str, object]:
    search_demo = run_search_demo(top_k=top_k)
    sample_users = build_sample_user_profiles()
    user_recommendations: list[dict[str, object]] = []

    for profile in sample_users:
        user_recommendations.append(
            {
                "user_id": profile["user_id"],
                "purchased_product_ids": profile["purchased_product_ids"],
                "preferred_category": profile["preferred_category"],
                "recommendations": recommend_for_user(
                    profile["purchased_product_ids"],
                    top_k=top_k,
                    category=profile["preferred_category"],
                ),
            }
        )

    demo_payload = {
        "search_demo": search_demo,
        "sample_user_recommendations": user_recommendations,
    }
    RECOMMENDATION_DEMO_PATH.write_text(
        json.dumps(demo_payload, indent=2),
        encoding="utf-8",
    )
    return demo_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run content-based recommendation demos.")
    parser.add_argument("--query", help="Optional text query to recommend from.")
    parser.add_argument("--product-id", help="Optional product id to find similar items.")
    parser.add_argument(
        "--user-products",
        nargs="*",
        help="Optional list of purchased product ids for user-level recommendations.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of recommendations.")
    parser.add_argument("--category", help="Optional category filter.")
    parser.add_argument("--min-price", type=float, help="Optional minimum price filter.")
    parser.add_argument("--max-price", type=float, help="Optional maximum price filter.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional similarity threshold for returned recommendations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_index_artifacts()

    if args.query:
        print_results(
            f"Content-based recommendations for query: {args.query}",
            recommend_from_query(
                args.query,
                top_k=args.top_k,
                category=args.category,
                min_price=args.min_price,
                max_price=args.max_price,
                similarity_threshold=args.threshold,
            ),
        )
        return

    if args.product_id:
        print_results(
            f"Similar products for product {args.product_id}",
            recommend_from_product(
                args.product_id,
                top_k=args.top_k,
                category=args.category,
                min_price=args.min_price,
                max_price=args.max_price,
                similarity_threshold=args.threshold,
            ),
        )
        return

    if args.user_products:
        print_results(
            f"Recommendations for purchased products: {args.user_products}",
            recommend_for_user(
                args.user_products,
                top_k=args.top_k,
                category=args.category,
                min_price=args.min_price,
                max_price=args.max_price,
                similarity_threshold=args.threshold,
            ),
        )
        return

    demo_payload = run_recommendation_demo(top_k=args.top_k)
    print(f"Saved recommendation demo to: {RECOMMENDATION_DEMO_PATH}")
    for entry in demo_payload["sample_user_recommendations"]:
        print_results(
            (
                f"User {entry['user_id']} purchased {entry['purchased_product_ids']} "
                f"({entry['preferred_category']})"
            ),
            entry["recommendations"],
        )


if __name__ == "__main__":
    main()
