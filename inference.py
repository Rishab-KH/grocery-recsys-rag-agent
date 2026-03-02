import torch
import faiss
import numpy as np
import pandas as pd
from model import TwoTowerModel
from data_processing import build_mappings, interactions_to_indices
from evaluate import evaluate_recommendations

# Load the trained model and mappings
def load_model_and_mappings(version):
    model_path = f'./models/version_{version}/model.pt'
    mappings_path = f'./models/version_{version}/mappings.pt'

    mappings = torch.load(mappings_path)
    user2idx = mappings['user2idx']
    prod2idx = mappings['prod2idx']

    model = TwoTowerModel(num_users=len(user2idx), num_items=len(prod2idx), emb_dim=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, user2idx, prod2idx

# Build FAISS index for item embeddings
def build_faiss_index(item_embs):
    faiss.normalize_L2(item_embs)
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    return index

# Perform inference for a single user
def infer_for_user(model, user_id, user2idx, prod2idx, index, k=10):
    user_idx = torch.tensor([user2idx[user_id]], dtype=torch.long)
    user_emb = model.get_user_embedding(user_idx).detach().numpy()
    faiss.normalize_L2(user_emb)

    scores, indices = index.search(user_emb, k)
    recommended_products = [list(prod2idx.keys())[list(prod2idx.values()).index(idx)] for idx in indices[0]]
    return recommended_products, scores[0]

# Load data and filter unseen users
def load_data_and_filter_unseen(data_dir, train_users):
    orders = pd.read_csv(f"{data_dir}/orders.csv")
    train_orders = orders[orders['eval_set'] == 'train']
    test_orders = orders[orders['eval_set'] == 'test']

    # Get users in training and testing sets
    train_users_set = set(train_orders['user_id'])
    test_users_set = set(test_orders['user_id'])

    # Filter unseen users (users in test set but not in training set)
    unseen_users = test_users_set - train_users_set

    # Prepare ground truth for unseen users
    test_products = pd.read_csv(f"{data_dir}/order_products__train.csv")
    ground_truth = test_products[test_products['order_id'].isin(test_orders['order_id'])]
    ground_truth = ground_truth.groupby('order_id')['product_id'].apply(list).to_dict()

    return unseen_users, ground_truth

# Main inference function
def main(version, data_dir):
    model, user2idx, prod2idx = load_model_and_mappings(version)

    # Generate item embeddings and build FAISS index
    item_embs = model.get_all_item_embeddings().detach().numpy()
    index = build_faiss_index(item_embs)

    # Load data and filter unseen users
    unseen_users, ground_truth = load_data_and_filter_unseen(data_dir, set(user2idx.keys()))

    # Perform inference for unseen users
    for user_id in unseen_users:
        if user_id not in user2idx:
            continue

        recommended_products, scores = infer_for_user(model, user_id, user2idx, prod2idx, index)
        print(f"Recommended Products for User {user_id}: {recommended_products}")

        # Evaluate results
        recommendations = {user_id: recommended_products}
        user_ground_truth = ground_truth.get(user_id, [])
        results = evaluate_recommendations(recommendations, {user_id: user_ground_truth}, ks=[10, 20])
        print(f"Evaluation Results for User {user_id}: {results}")

if __name__ == "__main__":
    # Example usage
    version = "20260302_123456"  # Replace with your model version
    data_dir = "./data"  # Path to data directory

    main(version, data_dir)