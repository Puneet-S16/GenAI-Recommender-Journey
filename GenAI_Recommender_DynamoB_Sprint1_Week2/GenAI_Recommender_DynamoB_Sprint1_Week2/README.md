#  Recommender System (Week 2)
## Overview

This project focuses on building a **Recommender System** using user behavior data. The goal is to suggest relevant products to users based on their past interactions and overall trends in the dataset.

This project builds directly on **Week 1**, where the raw data was collected, cleaned, and transformed into structured `.parquet` files.

---

## Data Source (From Week 1)

We use the processed datasets generated earlier:

* `products_clean.parquet` → Contains product details (name, category, etc.)
* `user_behavior.parquet` → Contains user interactions such as:

  * views
  * add to cart
  * purchases

These datasets are already cleaned and structured, so no heavy preprocessing is required in this stage.

---

## ETL Pipeline (What it does here)

Since the data is already processed, the ETL pipeline in this project is simplified.

### It performs:

* **Loading** data from `.parquet` files
* **Validating** required columns (user_id, product_id)
* **Transforming** user behavior into a numerical score:

  * view → 1
  * cart → 3
  * purchase → 5

This score represents how strongly a user is interested in a product.

---

## Core Concept: Interaction Matrix

The system converts user behavior into a **User-Item Interaction Matrix**:

* Rows → Users
* Columns → Products
* Values → Interaction scores

This matrix is the foundation for all recommendation models.

---

## Models Implemented

### 1. Popularity-Based Model

* Recommends the most popular products overall
* Works well as a baseline
* Does not personalize recommendations

---

### 2. Collaborative Filtering

#### User-Based

* Finds users with similar behavior
* Recommends items liked by similar users

#### Item-Based

* Finds products similar to a given product
* Used for “Similar items” recommendations

---

### 3. Matrix Factorization (ALS)

* Advanced model using latent features
* Learns hidden patterns in user behavior
* Provides better personalized recommendations

---

## Evaluation Metrics

The system includes evaluation functions such as:

* **Precision@K** → Accuracy of top recommendations
* **Recall@K** → Coverage of relevant items
* **MAP@K** → Ranking quality
* **RMSE** → Prediction error

These metrics help compare model performance.

---

##  Testing

Basic unit tests are included to verify:

* Model outputs
* Recommendation size
* Basic functionality

---

## ▶️ How to Run

Install dependencies:

```
pip install pandas scikit-learn scipy implicit
```

Run the project:

```
python main.py
```

---

## Project Structure

```
pipelines/
    etl_pipeline.py
models/
    collaborative_filter.py
    matrix_factorization.py
    popularity_model.py
evaluation/
    metrics.py
docs/
    SRS_Sprint1.pdf
tests/
    test_recommenders.py
main.py(The main file which imports all the files and gets the outputs)
README.md
```

---

## Documentation

* `docs/SRS_Sprint1.pdf` → System design and architecture
* `evaluation/evaluation_report.pdf` → Model performance analysis

---

---

## Summary

This project demonstrates how to build a complete recommender system using structured behavioral data. It covers everything from data handling to multiple recommendation strategies and evaluation, providing a strong foundation for real-world applications.

---
