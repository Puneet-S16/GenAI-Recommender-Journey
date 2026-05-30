# Hybrid Recommender - Team DynamoB - Sprint 2 Week 4

This deliverable implements an end-to-end hybrid recommender that combines collaborative filtering, content-based embedding similarity, and a learn-to-rank reranker trained with XGBoost.

## What Is Included

- `models/hybrid_recommender.py`
  - Production-facing `hybrid_recommend(user_id, query=None, top_k=5)` API
  - Candidate generation from CF + content + popularity
  - Deduplication and LTR reranking
- `models/ltr_model.py`
  - Feature matrix builder for next-purchase ranking
  - XGBoost `rank:pairwise` training pipeline
  - Optuna tuning loop
  - Evaluation, PDF report generation, notebook generation
- `models/model_checkpoints/`
  - Precomputed product embeddings
  - Trained LTR ranker checkpoint
  - LTR metadata with tuned weights and feature definitions
- `experiments/`
  - `optuna_study.db`
  - `experiment_results.json`
- `evaluation/`
  - `hybrid_evaluation_report.pdf`
- `notebooks/`
  - `hybrid_analysis.ipynb`
- `data/`
  - Local parquet datasets copied into the Week 4 workspace for reproducibility
- `tests/`
  - Basic unittest coverage for feature-matrix generation and hybrid recommendation output

## Hybrid Recommendation Logic

The hybrid system uses these signal sources:

1. Collaborative Filtering score
2. Embedding similarity score
3. Price sensitivity match
4. Style preference match
5. Historical CTR (simulated)

The heuristic blend used before LTR reranking is:

```text
hybrid_signal = w_cf*cf_score
              + w_embedding*gated_embedding_similarity
              + w_price*price_sensitivity_match
              + w_style*style_affinity_score
              + w_ctr*historical_ctr
```

Best tuned signal weights from Optuna:

- `cf`: `0.1653`
- `embedding`: `0.0747`
- `price`: `0.1316`
- `style`: `0.4565`
- `ctr`: `0.1720`
- Similarity threshold: `0.2319`

## LTR Feature Matrix

Each `(user, item)` row includes:

- `cf_score`
- `embedding_similarity`
- `price_delta_from_user_avg`
- `price_sensitivity_match`
- `category_match_score`
- `material_preference_score`
- `style_affinity_score`
- `historical_ctr`
- `purchase_history_count`
- `interaction_history_count`
- `hybrid_signal_score`

Label:

- `1` if the item is the user’s held-out next purchase
- `0` otherwise

Dataset design:

- 322 users with at least 2 unique purchases
- Chronological next-purchase holdout per user
- Split: 225 train / 48 validation / 49 test users
- Average candidate pool size: 83.47 items per user

## Model Training

- Ranker: `XGBoost`
- Objective: `rank:pairwise`
- Tuning: `Optuna`
- Trials run: `20`

Best model parameters:

- `n_estimators`: `281`
- `learning_rate`: `0.1264`
- `max_depth`: `4`
- `min_child_weight`: `4.9882`
- `subsample`: `0.6976`
- `colsample_bytree`: `0.7094`
- `reg_lambda`: `0.1951`

## Evaluation Summary

Test-set results:

| Model | Precision@5 | NDCG@5 | Diversity@5 |
|------|------:|------:|------:|
| CF only | 0.0041 | 0.0129 | 0.3295 |
| Content only | 0.0000 | 0.0000 | 0.0791 |
| Heuristic hybrid | 0.0000 | 0.0000 | 0.2054 |
| LTR hybrid | 0.0245 | 0.0515 | 0.2978 |

Improvement vs strongest baseline:

- `Precision@5`: `+0.0204` (`+497.56%`)
- `NDCG@5`: `+0.0386` (`+299.22%`)

Additional ranking metrics for the LTR hybrid:

- `Precision@10`: `0.0408`
- `NDCG@10`: `0.1417`
- `HitRate@5`: `0.1224`
- `HitRate@10`: `0.4082`

## Feature Importance

Top learned features from the final evaluation ranker:

1. `cf_score`
2. `embedding_similarity`
3. `historical_ctr`
4. `hybrid_signal_score`
5. `price_sensitivity_match`

Full importances are stored in `experiments/experiment_results.json`.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train/tune/generate artifacts:

```bash
python -m models.ltr_model --trials 20
```

Get recommendations for a user:

```bash
python -m models.hybrid_recommender --user-id 1276 --top-k 5
python -m models.hybrid_recommender --user-id 1276 --query "modern fabric sofa" --top-k 5
```

Run tests:

```bash
python -m unittest tests.test_hybrid_pipeline
```

## Repository Layout

```text
GenAI_Recommender_TeamDynamoB_Sprint2_Week4/
├── data/
├── evaluation/
│   └── hybrid_evaluation_report.pdf
├── experiments/
│   ├── experiment_results.json
│   └── optuna_study.db
├── models/
│   ├── hybrid_recommender.py
│   ├── ltr_model.py
│   ├── recommender_utils.py
│   └── model_checkpoints/
├── notebooks/
│   └── hybrid_analysis.ipynb
├── tests/
│   └── test_hybrid_pipeline.py
├── README.md
└── requirements.txt
```

## Notes

- The project uses precomputed product embeddings copied from the Week 3 content-based recommender deliverable.
- Query-aware recommendations reuse the same embedding model at inference time.
- The PDF report and notebook are generated automatically by the training pipeline.
