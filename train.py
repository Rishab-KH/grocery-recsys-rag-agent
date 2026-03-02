import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import faiss

import mlflow

from data_processing import (
    load_and_merge_data,
    filter_active_users,
    temporal_train_test_split,
    build_mappings,
    interactions_to_indices,
    get_popularity,
)
from model import TwoTowerModel
from evaluate import evaluate_recommendations
import datetime
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Save metadata for each model run
def save_metadata(version, model_dir, emb_dim, batch_size, epochs, results, rerank_results):
    metadata = {
        'version': version,
        'model_dir': model_dir,
        'emb_dim': emb_dim,
        'batch_size': batch_size,
        'epochs': epochs,
        'results': results,
        'rerank_results': rerank_results,
    }
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f'Metadata saved to {metadata_path}')

class InteractionDataset(Dataset):
    """
    Simple dataset that yields (user_idx, pos_item_idx) pairs for training.

    In-batch negatives: each other item in the batch serves as a negative for a user,
    which is an efficient way to get many negatives without explicit sampling.
    """

    def __init__(self, interactions_df):
        # Expect interactions_df with columns 'user_idx' and 'product_idx'
        self.data = interactions_df[['user_idx', 'product_idx']].drop_duplicates().values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i = self.data[idx]
        return int(u), int(i)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_pipeline(model, dataloader, optimizer, device, temperature=0.07):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(dataloader, desc='train', leave=False):
        user_idx, pos_idx = batch
        user_idx = user_idx.to(device)
        pos_idx = pos_idx.to(device)

        user_emb = model.get_user_embedding(user_idx)  # (B, D)
        pos_emb = model.get_item_embedding(pos_idx)    # (B, D)

        # In-batch negatives: compute similarity of each user to all positive items in batch
        # This yields a B x B score matrix where diagonal are positives.
        logits = user_emb @ pos_emb.t()
        logits = logits / temperature

        labels = torch.arange(logits.size(0), device=device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * user_idx.size(0)
    return total_loss / len(dataloader.dataset)


def build_faiss_index(item_embeddings: np.ndarray):
    d = item_embeddings.shape[1]
    try:
        # We train with dot-product similarity. With L2-normalized embeddings,
        # inner product is equivalent to cosine similarity.
        print("Initializing FAISS IndexFlatIP...")
        index = faiss.IndexFlatIP(d)
        index.add(item_embeddings)
        print("FAISS index built successfully.")
        return index
    except Exception as e:
        print(f"FAISS failed: {e}. Falling back to Scikit-learn.")
        nn = NearestNeighbors(n_neighbors=20, metric='cosine').fit(item_embeddings)
        return nn


def retrieve_topk(index, user_embeddings: np.ndarray, k: int = 20):
    # user_embeddings: (num_users, dim)
    scores, indices = index.search(user_embeddings.astype('float32'), k)
    return indices, scores

def rerank_candidates(
    candidates: np.ndarray,
    cand_scores: np.ndarray,
    test_users: list[int],
    train_idx,
    pop_counts: np.ndarray,
    alpha: float = 0.75,
    beta: float = 0.20,
    gamma: float = 0.05,
):
    """
    Rerank candidates by combining model score, popularity, and a small random factor for tie-breaking.
    """
    """Simple reranker over Top-K candidates.

    Combine:
      - retrieval similarity score (two-tower)
      - global popularity prior
      - user history prior (whether user bought it before)
    """
    # Normalize popularity to [0, 1]
    pop = pop_counts.astype(np.float32)
    pop = np.log1p(pop)
    pop = pop / (pop.max() + 1e-8)

    # Build user history sets only for test users (keeps memory reasonable)
    train_subset = train_idx[train_idx['user_idx'].isin(test_users)]
    user_hist = train_subset.groupby('user_idx')['product_idx'].apply(lambda s: set(s.tolist())).to_dict()

    reranked = {}
    for row_i, u in enumerate(test_users):
        items = candidates[row_i]
        sims = cand_scores[row_i]

        pop_feat = pop[items]
        hist = user_hist.get(u, set())
        hist_feat = np.array([1.0 if it in hist else 0.0 for it in items], dtype=np.float32)

        final = alpha * sims + beta * pop_feat + gamma * hist_feat
        order = np.argsort(-final)  # descending
        reranked[u] = items[order].tolist()
    return reranked

def train_reranker(features, labels):
    """
    Train a reranker using Gradient Boosting.
    Features: User-item pair features (e.g., similarity, popularity, history).
    Labels: Ground truth relevance scores.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(n_estimators=100, max_depth=5))
    ])
    pipeline.fit(features, labels)
    return pipeline


def extract_features(candidates, cand_scores, test_users, train_idx, pop_counts):
    """
    Extract features for reranking.
    Features include similarity scores, popularity, and user history.
    """
    features = []
    labels = []
    train_subset = train_idx[train_idx['user_idx'].isin(test_users)]
    user_hist = train_subset.groupby('user_idx')['product_idx'].apply(lambda s: set(s.tolist())).to_dict()

    for row_i, u in enumerate(test_users):
        items = candidates[row_i]
        sims = cand_scores[row_i]
        pop_feat = pop_counts[items]
        hist = user_hist.get(u, set())
        hist_feat = np.array([1.0 if it in hist else 0.0 for it in items], dtype=np.float32)

        for i, item in enumerate(items):
            features.append([sims[i], pop_feat[i], hist_feat[i]])
            labels.append(1.0 if item in user_hist.get(u, set()) else 0.0)

    return np.array(features), np.array(labels)

def main(data_dir='./data/', emb_dim=128, batch_size=4096, epochs=8, seed=42, resume_if_exists=True):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate unique version identifier (timestamp)
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'./models/version_{version}'
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'model.pt')
    mappings_path = os.path.join(model_dir, 'mappings.pt')

    print('Loading and preprocessing data...')
    orders, interactions, _ = load_and_merge_data(data_dir)
    interactions = filter_active_users(orders, interactions, min_orders=3)
    train_df, test_df = temporal_train_test_split(interactions)

    user2idx, prod2idx = build_mappings(train_df, test_df)
    train_idx = interactions_to_indices(train_df, user2idx, prod2idx)
    test_idx = interactions_to_indices(test_df, user2idx, prod2idx)

    num_users = len(user2idx)
    num_items = len(prod2idx)
    print(f'Num users: {num_users}, num items: {num_items}')

    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(mappings_path):
        print(f'\nLoading pre-trained model from {model_path}...')
        model = TwoTowerModel(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print('\nTraining new model...')
        dataset = InteractionDataset(train_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        model = TwoTowerModel(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, epochs + 1):
            loss = train_pipeline(model, dataloader, optimizer, device)
            print(f'Epoch {epoch}/{epochs} - loss: {loss:.4f}')
        
        # Save model and mappings
        torch.save(model.state_dict(), model_path)
        torch.save({'user2idx': user2idx, 'prod2idx': prod2idx}, mappings_path)
        print(f'\nModel saved to {model_path}')

    # Build FAISS index over item embeddings
    print('Building FAISS index...')
    item_embs = model.get_all_item_embeddings().numpy().astype('float32')
    print(f'Item embeddings shape: {item_embs.shape}')
    # Ensure embeddings are normalized (they should be)
    faiss.normalize_L2(item_embs)
    index = build_faiss_index(item_embs)
    print('Index built successfully.')

    # Prepare test users: unique user_idx in test set
    test_by_user = test_idx.groupby('user_idx')['product_idx'].apply(list).to_dict()
    test_users = list(test_by_user.keys())

    # Compute user embeddings and retrieve
    user_idx_tensor = torch.tensor(test_users, dtype=torch.long)
    with torch.no_grad():
        user_embs = model.get_user_embedding(user_idx_tensor).cpu().numpy().astype('float32')
    faiss.normalize_L2(user_embs)

    K_list = [10, 20]
    recs_idx, recs_scores = retrieve_topk(index, user_embs, k=max(K_list))

    # Map results to dict user_idx -> list of product_idx
    recs = {u: recs_idx[i].tolist() for i, u in enumerate(test_users)}

    # Ground truth dict
    truths = test_by_user

    # Evaluate
    results = evaluate_recommendations(recs, truths, ks=K_list)

    # Popularity baseline
    popular = get_popularity(train_df, prod2idx)
    pop_recs = {u: popular[:max(K_list)] for u in test_users}
    pop_results = evaluate_recommendations(pop_recs, truths, ks=K_list)


    # -------------------------------
    # Popularity baseline
    # -------------------------------
    pop_series = train_df['product_id'].map(prod2idx).dropna().astype(int).value_counts()
    pop_counts = np.zeros(num_items, dtype=np.int64)
    pop_counts[pop_series.index.values] = pop_series.values

    popular = pop_series.index.tolist()
    pop_recs = {u: popular[:max(K_list)] for u in test_users}
    pop_results = evaluate_recommendations(pop_recs, truths, ks=K_list)

    # -------------------------------
    # Train Advanced Reranker
    # -------------------------------
    print('Training advanced reranker...')
    features, labels = extract_features(recs_idx, recs_scores, test_users, train_idx, pop_counts)
    reranker = train_reranker(features, labels)

    # Rerank candidates using the trained reranker
    reranked_recs = {}
    for row_i, u in enumerate(test_users):
        items = recs_idx[row_i]
        item_features = features[row_i * len(items):(row_i + 1) * len(items)]
        scores = reranker.predict(item_features)
        order = np.argsort(-scores)  # descending
        reranked_recs[u] = items[order].tolist()

    rerank_results = evaluate_recommendations(reranked_recs, truths, ks=K_list)

    print('\nRetrieval Results (Model):')
    for k in K_list:
        print(f"Recall@{k}: {results[f'Recall@{k}']:.4f}, NDCG@{k}: {results[f'NDCG@{k}']:.4f}")

    print('\nPopularity Baseline:')
    for k in K_list:
        print(f"Recall@{k}: {pop_results[f'Recall@{k}']:.4f}, NDCG@{k}: {pop_results[f'NDCG@{k}']:.4f}")

    print('\nReranked Results (Two-tower + priors):')
    for k in K_list:
        print(f"Recall@{k}: {rerank_results[f'Recall@{k}']:.4f}, NDCG@{k}: {rerank_results[f'NDCG@{k}']:.4f}")


    # Save metadata
    save_metadata(
        version=version,
        model_dir=model_dir,
        emb_dim=emb_dim,
        batch_size=batch_size,
        epochs=epochs,
        results=results,
        rerank_results=rerank_results,
    )

    # Start MLflow experiment
    mlflow.set_experiment("Two-Tower Recsys")

    with mlflow.start_run():
        mlflow.log_param("embedding_dim", emb_dim)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        # Train the model
        for epoch in range(1, epochs + 1):
            loss = train_pipeline(model, dataloader, optimizer, device)
            print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f}")
            mlflow.log_metric("loss", loss, step=epoch)

        # Save model and mappings
        torch.save(model.state_dict(), model_path)
        torch.save({'user2idx': user2idx, 'prod2idx': prod2idx}, mappings_path)
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.log_artifact(mappings_path, artifact_path="mappings")

        # Log evaluation metrics
        for k in K_list:
            mlflow.log_metric(f"Recall@{k}", results[f"Recall@{k}"])
            mlflow.log_metric(f"NDCG@{k}", results[f"NDCG@{k}"])

        # Log reranking results
        for k in K_list:
            mlflow.log_metric(f"Rerank_Recall@{k}", rerank_results[f"Recall@{k}"])
            mlflow.log_metric(f"Rerank_NDCG@{k}", rerank_results[f"NDCG@{k}"])

if __name__ == '__main__':
    main()
