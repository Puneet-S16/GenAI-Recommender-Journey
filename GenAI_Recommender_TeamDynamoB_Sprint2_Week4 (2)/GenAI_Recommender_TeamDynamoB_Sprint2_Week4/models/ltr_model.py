from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nbformat as nbf
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRanker

from models.recommender_utils import (
    DEFAULT_SIGNAL_WEIGHTS,
    EVALUATION_REPORT_PATH,
    EXPERIMENT_RESULTS_PATH,
    LTR_METADATA_PATH,
    LTR_MODEL_PATH,
    NOTEBOOK_PATH,
    OPTUNA_DB_PATH,
    Catalog,
    CollaborativeFilterScorer,
    ContentScorer,
    ItemStatistics,
    PreferenceProfileModel,
    build_candidate_feature_frame,
    build_interactions,
    dedupe_preserve_order,
    ensure_directories,
    load_behavior,
    normalize_signal_weights,
    save_json,
)


FEATURE_COLUMNS = [
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
BASELINE_SCORE_COLUMNS = {
    "cf_only": "cf_score",
    "content_only": "embedding_similarity",
    "heuristic_hybrid": "hybrid_signal_score",
}


@dataclass(slots=True)
class RankingDataBundle:
    ranking_frame: pd.DataFrame
    holdout_targets: pd.DataFrame
    data_summary: dict[str, Any]

    def split_frame(self, split: str) -> pd.DataFrame:
        frame = self.ranking_frame[self.ranking_frame["split"] == split].copy()
        return frame.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def _choose_holdout_targets(behavior: pd.DataFrame) -> pd.DataFrame:
    purchases = behavior[behavior["event_type"] == "purchase"].sort_values(["user_id", "timestamp"])
    rows: list[dict[str, Any]] = []

    for user_id, group in purchases.groupby("user_id"):
        if group["product_id"].nunique() < 2:
            continue

        counts = group["product_id"].value_counts()
        selected = None
        for row in group.sort_values("timestamp", ascending=False).itertuples(index=False):
            if int(counts.loc[int(row.product_id)]) == 1:
                selected = row
                break
        if selected is None:
            continue

        rows.append(
            {
                "user_id": int(user_id),
                "target_product_id": int(selected.product_id),
                "cutoff_timestamp": pd.Timestamp(selected.timestamp),
            }
        )

    targets = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    return targets


def _assign_user_splits(
    targets: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    assigned = targets.copy()
    user_ids = assigned["user_id"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(user_ids)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_map: dict[int, str] = {}
    for position, user_id in enumerate(shuffled):
        if position < train_end:
            split_map[int(user_id)] = "train"
        elif position < val_end:
            split_map[int(user_id)] = "val"
        else:
            split_map[int(user_id)] = "test"

    assigned["split"] = assigned["user_id"].map(split_map)
    return assigned


def _truncate_behavior(behavior: pd.DataFrame, holdout_targets: pd.DataFrame) -> pd.DataFrame:
    cutoff_frame = holdout_targets[["user_id", "cutoff_timestamp"]]
    merged = behavior.merge(cutoff_frame, on="user_id", how="left")
    keep_mask = merged["cutoff_timestamp"].isna() | (merged["timestamp"] < merged["cutoff_timestamp"])
    truncated = merged.loc[keep_mask, behavior.columns].copy()
    return truncated.reset_index(drop=True)


def build_ranking_dataset(
    *,
    seed: int = 42,
    cf_top_k: int = 25,
    content_top_k: int = 25,
    popularity_top_k: int = 10,
    random_negatives: int = 25,
) -> RankingDataBundle:
    ensure_directories()
    behavior = load_behavior()
    holdout_targets = _assign_user_splits(_choose_holdout_targets(behavior), seed=seed)
    observed_behavior = _truncate_behavior(behavior, holdout_targets)
    interactions = build_interactions(observed_behavior)

    catalog = Catalog.from_local_artifacts()
    cf_model = CollaborativeFilterScorer().fit(interactions)
    profile_model = PreferenceProfileModel().fit(interactions, catalog.products)
    item_stats = ItemStatistics().fit(interactions)
    content_model = ContentScorer(catalog)

    popularity_candidates = item_stats.recommend(top_n=popularity_top_k)
    all_product_ids = catalog.product_ids
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    candidate_sizes: list[int] = []

    for target in holdout_targets.itertuples(index=False):
        user_id = int(target.user_id)
        positive_item = int(target.target_product_id)
        purchased_before = set(profile_model.purchased_items.get(user_id, set()))
        exclude_items = purchased_before.difference({positive_item})

        cf_candidates = cf_model.recommend(
            user_id,
            top_n=cf_top_k,
            exclude_product_ids=exclude_items,
        )
        content_candidates = content_model.recommend_from_history(
            profile_model.get_seed_items(user_id),
            top_n=content_top_k,
            exclude_product_ids=exclude_items,
            min_similarity=0.0,
        )
        hard_candidates = dedupe_preserve_order(cf_candidates + content_candidates + popularity_candidates)

        hard_candidate_set = set(hard_candidates)
        random_pool = [
            product_id
            for product_id in all_product_ids
            if product_id not in exclude_items
            and product_id != positive_item
            and product_id not in hard_candidate_set
        ]
        random_sample = (
            rng.choice(random_pool, size=min(random_negatives, len(random_pool)), replace=False).tolist()
            if random_pool
            else []
        )
        candidate_product_ids = dedupe_preserve_order([positive_item] + hard_candidates + random_sample)

        feature_frame = build_candidate_feature_frame(
            user_id=user_id,
            candidate_product_ids=candidate_product_ids,
            catalog=catalog,
            cf_model=cf_model,
            content_model=content_model,
            profile_model=profile_model,
            item_stats=item_stats,
            signal_weights=DEFAULT_SIGNAL_WEIGHTS,
            similarity_threshold=0.0,
        )
        feature_frame["label"] = (feature_frame["product_id"].astype(int) == positive_item).astype(int)
        feature_frame["split"] = str(target.split)
        feature_frame["target_product_id"] = positive_item

        rows.append(feature_frame)
        candidate_sizes.append(len(candidate_product_ids))

    ranking_frame = pd.concat(rows, ignore_index=True)
    ranking_frame = ranking_frame.sort_values(["split", "user_id", "product_id"]).reset_index(drop=True)

    split_counts = holdout_targets["split"].value_counts().sort_index().to_dict()
    data_summary = {
        "eligible_users": int(len(holdout_targets)),
        "observed_interactions": int(len(observed_behavior)),
        "ranking_rows": int(len(ranking_frame)),
        "products": int(len(catalog.products)),
        "candidate_pool_mean": round(float(np.mean(candidate_sizes)), 2),
        "candidate_pool_min": int(min(candidate_sizes)),
        "candidate_pool_max": int(max(candidate_sizes)),
        "split_user_counts": {str(key): int(value) for key, value in split_counts.items()},
    }
    return RankingDataBundle(
        ranking_frame=ranking_frame,
        holdout_targets=holdout_targets,
        data_summary=data_summary,
    )


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("user_id", sort=False).size().astype(int).tolist()


def _sorted_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def _prepare_ranker_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    ordered = _sorted_for_ranking(frame)
    features = ordered[FEATURE_COLUMNS].astype(float)
    labels = ordered["label"].astype(int)
    return features, labels, _group_sizes(ordered)


def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(relevances.size, dtype=float) + 2.0)
    return float(np.sum((2.0**relevances - 1.0) * discounts))


def evaluate_grouped_ranking(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    grouped = frame.groupby("user_id", sort=False)

    for top_k in top_k_values:
        precision_scores: list[float] = []
        ndcg_scores: list[float] = []
        hit_rates: list[float] = []

        for _, group in grouped:
            ranked = group.sort_values(score_column, ascending=False).head(top_k)
            labels = ranked["label"].to_numpy(dtype=float)
            precision_scores.append(float(labels.sum()) / float(top_k))
            hit_rates.append(float(labels.sum() > 0))

            ideal = np.sort(group["label"].to_numpy(dtype=float))[::-1][:top_k]
            ideal_dcg = _dcg(ideal)
            ndcg_scores.append(0.0 if ideal_dcg == 0.0 else _dcg(labels) / ideal_dcg)

        metrics[f"precision@{top_k}"] = round(float(np.mean(precision_scores)), 4)
        metrics[f"ndcg@{top_k}"] = round(float(np.mean(ndcg_scores)), 4)
        metrics[f"hit_rate@{top_k}"] = round(float(np.mean(hit_rates)), 4)

    return metrics


def mean_intra_list_diversity(
    frame: pd.DataFrame,
    *,
    score_column: str,
    catalog: Catalog,
    top_k: int = 5,
) -> float:
    embedding_lookup = catalog.embeddings
    product_id_to_index = catalog.product_id_to_index
    scores: list[float] = []

    for _, group in frame.groupby("user_id", sort=False):
        ranked = group.sort_values(score_column, ascending=False).head(top_k)
        product_ids = ranked["product_id"].astype(int).tolist()
        indices = [product_id_to_index[product_id] for product_id in product_ids if product_id in product_id_to_index]
        if len(indices) < 2:
            scores.append(0.0)
            continue
        embeddings = embedding_lookup[indices]
        similarity = embeddings @ embeddings.T
        upper = similarity[np.triu_indices(len(indices), k=1)]
        scores.append(float(np.mean(1.0 - upper)))

    return round(float(np.mean(scores)) if scores else 0.0, 4)


def _signal_config_from_trial(trial: optuna.Trial) -> tuple[dict[str, float], float]:
    raw_weights = {
        "cf": trial.suggest_float("weight_cf", 0.05, 1.0),
        "embedding": trial.suggest_float("weight_embedding", 0.05, 1.0),
        "price": trial.suggest_float("weight_price", 0.05, 0.6),
        "style": trial.suggest_float("weight_style", 0.05, 0.6),
        "ctr": trial.suggest_float("weight_ctr", 0.05, 0.8),
    }
    threshold = trial.suggest_float("similarity_threshold", 0.10, 0.60)
    return normalize_signal_weights(raw_weights), float(threshold)


def _apply_signal_config(
    frame: pd.DataFrame,
    *,
    signal_weights: dict[str, float],
    similarity_threshold: float,
) -> pd.DataFrame:
    working = frame.copy()
    gated_embedding = np.where(
        working["embedding_similarity"].to_numpy(dtype=float) >= float(similarity_threshold),
        working["embedding_similarity"].to_numpy(dtype=float),
        0.0,
    )
    working["hybrid_signal_score"] = (
        signal_weights["cf"] * working["cf_score"].to_numpy(dtype=float)
        + signal_weights["embedding"] * gated_embedding
        + signal_weights["price"] * working["price_sensitivity_match"].to_numpy(dtype=float)
        + signal_weights["style"] * working["style_affinity_score"].to_numpy(dtype=float)
        + signal_weights["ctr"] * working["historical_ctr"].to_numpy(dtype=float)
    )
    return working


def _build_ranker(
    *,
    params: dict[str, Any],
    random_state: int = 42,
) -> XGBRanker:
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@5",
        tree_method="hist",
        random_state=random_state,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        verbosity=0,
    )


def tune_and_train(
    bundle: RankingDataBundle,
    *,
    n_trials: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    train_frame = bundle.split_frame("train")
    val_frame = bundle.split_frame("val")
    test_frame = bundle.split_frame("test")

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("Train/val/test splits must all be non-empty.")

    study = optuna.create_study(
        study_name="hybrid_ranking_study",
        direction="maximize",
        storage=f"sqlite:///{OPTUNA_DB_PATH.as_posix()}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        signal_weights, similarity_threshold = _signal_config_from_trial(trial)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 320),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }

        trial_train = _apply_signal_config(
            train_frame,
            signal_weights=signal_weights,
            similarity_threshold=similarity_threshold,
        )
        trial_val = _apply_signal_config(
            val_frame,
            signal_weights=signal_weights,
            similarity_threshold=similarity_threshold,
        )

        train_x, train_y, train_group = _prepare_ranker_inputs(trial_train)
        val_x, val_y, val_group = _prepare_ranker_inputs(trial_val)

        ranker = _build_ranker(params=params, random_state=seed)
        ranker.fit(
            train_x,
            train_y,
            group=train_group,
            eval_set=[(val_x, val_y)],
            eval_group=[val_group],
            verbose=False,
        )

        scored_val = trial_val.copy()
        scored_val["ltr_score"] = ranker.predict(val_x)
        metrics = evaluate_grouped_ranking(scored_val, score_column="ltr_score", top_k_values=(5, 10))
        objective_value = metrics["ndcg@5"]
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("signal_weights", signal_weights)
        trial.set_user_attr("similarity_threshold", similarity_threshold)
        return objective_value

    remaining_trials = max(int(n_trials) - len(study.trials), 0)
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_signal_weights = normalize_signal_weights(dict(best_trial.user_attrs["signal_weights"]))
    best_similarity_threshold = float(best_trial.user_attrs["similarity_threshold"])
    best_model_params = {
        key: best_trial.params[key]
        for key in (
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
        )
    }

    train_val_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    tuned_train_val = _apply_signal_config(
        train_val_frame,
        signal_weights=best_signal_weights,
        similarity_threshold=best_similarity_threshold,
    )
    tuned_test = _apply_signal_config(
        test_frame,
        signal_weights=best_signal_weights,
        similarity_threshold=best_similarity_threshold,
    )

    train_val_x, train_val_y, train_val_group = _prepare_ranker_inputs(tuned_train_val)
    test_x, test_y, test_group = _prepare_ranker_inputs(tuned_test)
    eval_ranker = _build_ranker(params=best_model_params, random_state=seed)
    eval_ranker.fit(
        train_val_x,
        train_val_y,
        group=train_val_group,
        eval_set=[(test_x, test_y)],
        eval_group=[test_group],
        verbose=False,
    )

    scored_test = tuned_test.copy()
    scored_test["ltr_score"] = eval_ranker.predict(test_x)

    catalog = Catalog.from_local_artifacts()
    evaluation = {
        baseline_name: {
            **evaluate_grouped_ranking(scored_test, score_column=column, top_k_values=(5, 10)),
            "diversity@5": mean_intra_list_diversity(scored_test, score_column=column, catalog=catalog, top_k=5),
        }
        for baseline_name, column in BASELINE_SCORE_COLUMNS.items()
    }
    evaluation["ltr_hybrid"] = {
        **evaluate_grouped_ranking(scored_test, score_column="ltr_score", top_k_values=(5, 10)),
        "diversity@5": mean_intra_list_diversity(scored_test, score_column="ltr_score", catalog=catalog, top_k=5),
    }

    best_baseline_precision = max(
        evaluation["cf_only"]["precision@5"],
        evaluation["content_only"]["precision@5"],
        evaluation["heuristic_hybrid"]["precision@5"],
    )
    best_baseline_ndcg = max(
        evaluation["cf_only"]["ndcg@5"],
        evaluation["content_only"]["ndcg@5"],
        evaluation["heuristic_hybrid"]["ndcg@5"],
    )
    evaluation["improvement_vs_best_baseline"] = {
        "precision@5_absolute": round(evaluation["ltr_hybrid"]["precision@5"] - best_baseline_precision, 4),
        "ndcg@5_absolute": round(evaluation["ltr_hybrid"]["ndcg@5"] - best_baseline_ndcg, 4),
        "precision@5_relative_pct": round(
            100.0 * (evaluation["ltr_hybrid"]["precision@5"] - best_baseline_precision) / max(best_baseline_precision, 1e-8),
            2,
        ),
        "ndcg@5_relative_pct": round(
            100.0 * (evaluation["ltr_hybrid"]["ndcg@5"] - best_baseline_ndcg) / max(best_baseline_ndcg, 1e-8),
            2,
        ),
    }

    feature_importance = {
        feature_name: round(float(score), 6)
        for feature_name, score in sorted(
            zip(FEATURE_COLUMNS, eval_ranker.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    all_frame = _apply_signal_config(
        bundle.ranking_frame,
        signal_weights=best_signal_weights,
        similarity_threshold=best_similarity_threshold,
    )
    all_x, all_y, all_group = _prepare_ranker_inputs(all_frame)
    deployment_ranker = _build_ranker(params=best_model_params, random_state=seed)
    deployment_ranker.fit(all_x, all_y, group=all_group, verbose=False)
    deployment_ranker.save_model(str(LTR_MODEL_PATH))

    metadata = {
        "objective": "rank:pairwise",
        "feature_columns": FEATURE_COLUMNS,
        "model_params": best_model_params,
        "signal_weights": best_signal_weights,
        "similarity_threshold": best_similarity_threshold,
        "evaluation": evaluation,
        "feature_importance": feature_importance,
        "data_summary": bundle.data_summary,
    }
    save_json(LTR_METADATA_PATH, metadata)

    result_payload = {
        "data_summary": bundle.data_summary,
        "best_trial": {
            "number": int(best_trial.number),
            "value": float(best_trial.value),
            "params": {key: float(value) if isinstance(value, float) else int(value) for key, value in best_trial.params.items()},
            "metrics": best_trial.user_attrs["metrics"],
        },
        "best_signal_config": {
            "weights": best_signal_weights,
            "similarity_threshold": best_similarity_threshold,
            "scoring_function": (
                "hybrid_signal = w_cf*cf_score + w_embedding*gated_embedding_similarity + "
                "w_price*price_sensitivity_match + w_style*style_affinity_score + w_ctr*historical_ctr"
            ),
        },
        "evaluation": evaluation,
        "feature_importance": feature_importance,
        "artifacts": {
            "ltr_model_path": str(LTR_MODEL_PATH.resolve()),
            "ltr_metadata_path": str(LTR_METADATA_PATH.resolve()),
            "optuna_db_path": str(OPTUNA_DB_PATH.resolve()),
        },
    }
    save_json(EXPERIMENT_RESULTS_PATH, result_payload)
    _generate_pdf_report(result_payload)
    _generate_analysis_notebook(result_payload)
    return result_payload


def _generate_pdf_report(results: dict[str, Any]) -> None:
    ensure_directories()
    evaluation = results["evaluation"]
    feature_importance = results["feature_importance"]

    with PdfPages(EVALUATION_REPORT_PATH) as pdf:
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.axis("off")
        summary_lines = [
            "Hybrid Recommender Evaluation Report",
            "",
            f"Eligible users with next-purchase holdout: {results['data_summary']['eligible_users']}",
            f"Ranking rows: {results['data_summary']['ranking_rows']}",
            f"Average candidate pool size: {results['data_summary']['candidate_pool_mean']}",
            "",
            "Signal sources:",
            "1. Collaborative Filtering score",
            "2. Embedding similarity score",
            "3. Price sensitivity match",
            "4. Style preference match",
            "5. Historical CTR (simulated)",
            "",
            "Scoring function:",
            results["best_signal_config"]["scoring_function"],
            "",
            "Best signal weights:",
            *[
                f"  - {name}: {value:.3f}"
                for name, value in results["best_signal_config"]["weights"].items()
            ],
            f"Similarity threshold: {results['best_signal_config']['similarity_threshold']:.3f}",
            "",
            "Headline test-set lift vs strongest baseline:",
            f"  - Precision@5: {evaluation['improvement_vs_best_baseline']['precision@5_absolute']:+.4f} "
            f"({evaluation['improvement_vs_best_baseline']['precision@5_relative_pct']:+.2f}%)",
            f"  - NDCG@5: {evaluation['improvement_vs_best_baseline']['ndcg@5_absolute']:+.4f} "
            f"({evaluation['improvement_vs_best_baseline']['ndcg@5_relative_pct']:+.2f}%)",
        ]
        ax.text(0.03, 0.97, "\n".join(summary_lines), va="top", ha="left", fontsize=11, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        model_names = ["cf_only", "content_only", "heuristic_hybrid", "ltr_hybrid"]
        display_names = ["CF", "Content", "Heuristic", "LTR Hybrid"]
        precision_values = [evaluation[name]["precision@5"] for name in model_names]
        ndcg_values = [evaluation[name]["ndcg@5"] for name in model_names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(display_names, precision_values, color=["#5B8E7D", "#D98E04", "#C45D3C", "#2F5D8C"])
        axes[0].set_title("Precision@5")
        axes[0].set_ylim(0, max(precision_values) * 1.25)
        axes[1].bar(display_names, ndcg_values, color=["#5B8E7D", "#D98E04", "#C45D3C", "#2F5D8C"])
        axes[1].set_title("NDCG@5")
        axes[1].set_ylim(0, max(ndcg_values) * 1.25)
        for axis in axes:
            axis.grid(axis="y", alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        diversity_values = [evaluation[name]["diversity@5"] for name in model_names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(diversity_values, ndcg_values, s=100, color="#2F5D8C")
        for x_value, y_value, label in zip(diversity_values, ndcg_values, display_names):
            ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel("Diversity@5")
        ax.set_ylabel("NDCG@5")
        ax.set_title("Diversity vs Relevance Trade-off")
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        top_features = list(feature_importance.items())[:10]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [name for name, _ in reversed(top_features)],
            [value for _, value in reversed(top_features)],
            color="#5B8E7D",
        )
        ax.set_title("LTR Feature Importance")
        ax.grid(axis="x", alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _generate_analysis_notebook(results: dict[str, Any]) -> None:
    ensure_directories()
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Hybrid Recommender Analysis\n"
            "This notebook summarizes the Week 4 hybrid recommender pipeline, the tuned LTR setup, "
            "and the evaluation outputs stored in `experiments/experiment_results.json`."
        ),
        nbf.v4.new_markdown_cell(
            "## Signal Design\n"
            "- Collaborative Filtering score\n"
            "- Embedding similarity score\n"
            "- Price sensitivity match\n"
            "- Style preference match\n"
            "- Historical CTR (simulated)\n\n"
            "Hybrid pre-ranking score:\n\n"
            "```text\n"
            f"{results['best_signal_config']['scoring_function']}\n"
            "```"
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "results = json.loads(Path('../experiments/experiment_results.json').read_text())\n"
            "results['evaluation']"
        ),
        nbf.v4.new_code_cell(
            "from models.hybrid_recommender import hybrid_recommend\n\n"
            "hybrid_recommend(user_id=1276, top_k=5)"
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "feature_importance = pd.Series(results['feature_importance']).sort_values()\n"
            "feature_importance.tail(10).plot(kind='barh', figsize=(8, 5), color='#5B8E7D')\n"
            "plt.title('Top LTR Feature Importances')\n"
            "plt.tight_layout()"
        ),
    ]
    NOTEBOOK_PATH.write_text(nbf.writes(notebook), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Week 4 hybrid LTR model.")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_ranking_dataset(seed=args.seed)
    results = tune_and_train(bundle, n_trials=args.trials, seed=args.seed)

    print(f"Saved Optuna study to: {OPTUNA_DB_PATH}")
    print(f"Saved experiment results to: {EXPERIMENT_RESULTS_PATH}")
    print(f"Saved ranking model to: {LTR_MODEL_PATH}")
    print(f"Saved ranking metadata to: {LTR_METADATA_PATH}")
    print(f"Saved evaluation report to: {EVALUATION_REPORT_PATH}")
    print(f"Saved analysis notebook to: {NOTEBOOK_PATH}")
    print(
        "Best test metrics | "
        f"Precision@5={results['evaluation']['ltr_hybrid']['precision@5']:.4f} | "
        f"NDCG@5={results['evaluation']['ltr_hybrid']['ndcg@5']:.4f}"
    )


if __name__ == "__main__":
    main()

