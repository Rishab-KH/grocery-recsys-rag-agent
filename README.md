# Two-Tower Retrieval Example (Instacart)

## Project Overview
This repository implements a recommendation system for the Instacart dataset using a Two-Tower model. The system retrieves and ranks products for users based on their purchase history. The pipeline includes data preprocessing, model training, retrieval, reranking, and evaluation.

## Files
- **data_processing.py**: Handles data loading, filtering, temporal splitting, and mapping creation.
- **model.py**: Defines the Two-Tower model for user and item embeddings.
- **train.py**: Orchestrates the end-to-end pipeline, including training, FAISS index building, retrieval, reranking, and evaluation.
- **evaluate.py**: Provides metrics such as Recall, NDCG, and MRR for evaluating recommendations.

## Experiments Conducted
### 1. Baseline Retrieval
- **Objective**: Train a Two-Tower model to retrieve top-K items for users.
- **Methodology**:
  - Used in-batch negatives for efficient contrastive learning.
  - Normalized embeddings to approximate cosine similarity.
  - Built a FAISS `IndexFlatIP` for fast nearest-neighbor retrieval.
- **Results**:
  - Evaluated using Recall@K and NDCG@K metrics.

### 2. Popularity Baseline
- **Objective**: Compare the Two-Tower model against a popularity-based recommendation system.
- **Methodology**:
  - Ranked items based on their global popularity in the training set.
- **Results**:
  - Provided a baseline for Recall@K and NDCG@K metrics.

### 3. Advanced Reranking
- **Objective**: Improve recommendation quality by reranking retrieved items.
- **Methodology**:
  - Extracted features such as similarity scores, item popularity, and user history.
  - Trained a Gradient Boosting Regressor to predict relevance scores.
  - Combined retrieval scores with reranker predictions for final ranking.
- **Results**:
  - Significant improvement in Recall@K and NDCG@K metrics compared to baseline retrieval.

### 4. Temporal Features
- **Objective**: Incorporate temporal dynamics into the recommendation pipeline.
- **Methodology**:
  - Split data temporally to ensure no information leakage.
  - Evaluated the model on temporally held-out test data.
- **Results**:
  - Improved generalization and robustness of the recommendation system.

## Experiment Results

The following table summarizes the results of all experiments conducted so far:

| Experiment                  | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|-----------------------------|-----------|----------|-----------|----------|
| **Popularity Baseline**     | 0.0699    | 0.0976   | 0.0955    | 0.0974   |
| **Two-Tower Retrieval**     | 0.1065    | 0.1047   | 0.1427    | 0.1140   |
| **Reranked (Two-Tower)**    | **0.1369**    | **0.1423**   | **0.1427**    | **0.1326**   |

### Retrieval (Model)
| Metric                      | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|-----------------------------|-----------|----------|-----------|----------|
| **Two-Tower Baseline**      | 0.0580    | 0.0594   | 0.0811    | 0.0655   |

This table highlights the improvements achieved through reranking and the use of the Two-Tower model compared to the popularity baseline.

## Experiment Tracking with MLflow

This project uses MLflow to track experiments, including parameters, metrics, and artifacts. To use MLflow:

1. Install MLflow:
   ```bash
   pip install mlflow
   ```
2. Start the MLflow UI:
   ```bash
   mlflow ui
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Open the MLflow UI in your browser at `http://localhost:5000` to view experiment results.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   Results will be printed to the console and saved in the `models/` directory.

## Notes
- The pipeline uses in-batch negatives for efficient training.
- FAISS is used for fast nearest-neighbor retrieval on CPU.
- Metadata for each experiment is saved in `metadata.json` within the `models/` directory.

## Future Work
- Transition the pipeline to Google Cloud for faster training and scalability.
- Experiment with additional reranking models, such as neural networks.
- Incorporate more advanced temporal features and user-item interaction dynamics.
