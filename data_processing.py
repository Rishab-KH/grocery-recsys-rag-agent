import os
import pandas as pd
from typing import Tuple, Dict


def load_and_merge_data(data_dir: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load orders and order_products (prior + train) and merge into a single interaction frame.
    """
    orders_path = os.path.join(data_dir, 'orders.csv')
    prior_path = os.path.join(data_dir, 'order_products__prior.csv')
    train_path = os.path.join(data_dir, 'order_products__train.csv')

    orders = pd.read_csv(orders_path)
    prior = pd.read_csv(prior_path)
    train = pd.read_csv(train_path)

    order_products = pd.concat([prior, train], ignore_index=True)

    # Merge orders with products to get user-level interactions
    interactions = orders.merge(order_products, on='order_id', how='inner')
    # Keep only relevant columns
    interactions = interactions[['user_id', 'order_id', 'order_number', 'product_id']]
    return orders, interactions, order_products


def filter_active_users(orders: pd.DataFrame, interactions: pd.DataFrame, min_orders: int = 3):
    """
    Keep only users with at least `min_orders` total orders.
    Temporal split requires a sufficiently long history per user.
    """
    user_max = orders.groupby('user_id')['order_number'].max().reset_index()
    active_users = user_max[user_max['order_number'] >= min_orders]['user_id']
    interactions = interactions[interactions['user_id'].isin(active_users)].copy()
    return interactions


def temporal_train_test_split(interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user, put all orders with order_number < max(order_number) into train,
    and the last order (order_number == max) into test.

    Why temporal split: ensures no future information is used to predict past behavior,
    reflecting realistic next-basket prediction.
    """
    # compute the last order number per user
    last = interactions.groupby('user_id')['order_number'].max().rename('last_order').reset_index()
    df = interactions.merge(last, on='user_id', how='inner')

    train = df[df['order_number'] < df['last_order']].copy()
    test = df[df['order_number'] == df['last_order']].copy()

    # Safety: drop the helper column
    train = train[['user_id', 'order_id', 'order_number', 'product_id']]
    test = test[['user_id', 'order_id', 'order_number', 'product_id']]
    return train, test


def build_mappings(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create user and product id -> compact integer index mappings.
    Only include products that appear in training set for the item vocabulary to avoid leakage.
    """
    users = pd.Index(train['user_id'].unique()).sort_values()
    user2idx = {u: i for i, u in enumerate(users)}

    # Build product vocabulary from training interactions (no leakage)
    products = pd.Index(train['product_id'].unique()).sort_values()
    prod2idx = {p: i for i, p in enumerate(products)}

    return user2idx, prod2idx


def interactions_to_indices(df: pd.DataFrame, user2idx: Dict[int, int], prod2idx: Dict[int, int]) -> pd.DataFrame:
    """
    Map raw ids to indices. Rows with products not in prod2idx are dropped to avoid leakage.
    """
    df = df.copy()
    df['user_idx'] = df['user_id'].map(user2idx)
    df['product_idx'] = df['product_id'].map(prod2idx)
    df = df.dropna(subset=['user_idx', 'product_idx'])
    df['user_idx'] = df['user_idx'].astype(int)
    df['product_idx'] = df['product_idx'].astype(int)
    return df[['user_id', 'order_id', 'order_number', 'product_id', 'user_idx', 'product_idx']]


def get_popularity(train: pd.DataFrame, prod2idx: Dict[int, int]):
    """
    Compute global popularity ranking (by training interactions). Returns list of product_idx sorted by popularity desc.
    """
    counts = train['product_id'].map(prod2idx).dropna().astype(int).value_counts()
    popular = counts.index.tolist()
    return popular
