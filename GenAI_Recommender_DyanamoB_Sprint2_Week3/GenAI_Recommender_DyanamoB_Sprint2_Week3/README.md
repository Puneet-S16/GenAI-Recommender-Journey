# GenAI Recommender System

An embedding-based product search and content recommendation system built using transformer models and vector similarity search. This project converts a structured product catalog into a semantic search engine with personalized recommendation capabilities.

---

## Features

- Semantic product search using natural language queries  
- Embedding generation from product descriptions  
- Fast similarity search using FAISS (HNSW index)  
- Content-based recommendation system  
- Support for multiple embedding models  
- Evaluation metrics for embedding quality and retrieval  

---

## Project Structure


.
├── data/
│ └── products_clean.parquet
├── embeddings/
│ ├── product_embeddings.npy
│ ├── embedding_metadata.json
│ └── embedding_evaluation.json
├── vector_db/
│ ├── faiss_index/
│ │ ├── index.faiss
│ │ └── index_metadata.json
│ ├── search_demo_results.json
│ └── recommendation_demo.json
├── src/
│ ├── embedder.py
│ ├── vector_store.py
│ └── content_recommender.py
└── README.md


---

## Pipeline Overview

### Embedding Generation
- Cleans and normalizes product descriptions  
- Generates sentence embeddings using transformer models  
- Stores embeddings and metadata  

### Vector Indexing
- Builds a FAISS HNSW index for approximate nearest neighbor search  
- Supports filtering by category, price, style, material, and color  

### Recommendation Engine
- Generates recommendations based on embedding similarity  
- Excludes already purchased items  
- Supports query-based and history-based recommendations  

---

## Outputs

| File | Description |
|------|------------|
| product_embeddings.npy | Product embeddings |
| embedding_metadata.json | Metadata and processed descriptions |
| embedding_evaluation.json | Evaluation metrics and similarity analysis |
| index.faiss | FAISS vector index |
| index_metadata.json | Index configuration and parameters |
| search_demo_results.json | Example search outputs |
| recommendation_demo.json | Example recommendations |

---

## Installation


pip install -r requirements.txt


---

## Usage

Run the pipeline:


python src/embedder.py
python src/vector_store.py
python src/content_recommender.py


---

## Model Options

- all-MiniLM-L6-v2  
- all-mpnet-base-v2  
- bge-base-en-v1.5  

Example:


python src/embedder.py --model all-mpnet-base-v2


---

## Example Queries


python src/vector_store.py --query "industrial wood wardrobe" --category Wardrobe


Other examples:
- modern fabric sofa  
- minimalist grey chair  

---

## Recommendation Example


python src/content_recommender.py --user-products 3 6 --top-k 5


---

## Current Configuration

- Embedding Model: all-MiniLM-L6-v2  
- Vector Database: FAISS (HNSW)  
- Filters: category, price range, style, material, color  
- Recommendation Logic: embedding similarity with purchased-item exclusion  

---

## Evaluation

Includes similarity distribution, retrieval metrics, and PCA-based experiments.

---

## Future Improvements

- Hybrid search (keyword + semantic)  
- Real-time API deployment  
- User behavior-based personalization  
- Frontend interface  

---

## License

MIT License